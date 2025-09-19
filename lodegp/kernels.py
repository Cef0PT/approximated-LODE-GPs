import copy as cp
from einops import rearrange
import re
import itertools
from itertools import zip_longest
from torch.distributions import constraints
import torch
from torch.nn import ParameterDict
from functools import reduce
import gpytorch
from gpytorch.lazy import *
from gpytorch.kernels.kernel import Kernel
from linear_operator.operators import (
    to_linear_operator,
    CatLinearOperator,
    DiagLinearOperator
)
from famgpytorch.functions import ChebyshevHermitePolynomials
from famgpytorch.kernels import approx_rbf_covariance
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from sage.arith.misc import factorial
from sage.symbolic.operators import add_vararg
import numpy as np
import pdb
from gpytorch.constraints import Positive
import random
import einops
torch_operations = {'mul': torch.mul, 'add': torch.add,
                    'pow': torch.pow, 'exp':torch.exp,
                    'sin':torch.sin, 'cos':torch.cos,
                    'log': torch.log}
from .permutation_linear_operator import PermutationLinearOperator

DEBUG =False


class LODE_Kernel(Kernel):
        def __init__(self, covar_description, model_parameters: ParameterDict, active_dims=None, approx=False, number_of_eigenvalues=200, **kwargs):
            super(LODE_Kernel, self).__init__(active_dims=active_dims)
            self.covar_description = covar_description
            self.model_parameters = model_parameters
            self.num_tasks = len(covar_description)
            self.approx = approx
            self.number_of_eigenvalues = number_of_eigenvalues

            # get kernel inputs
            self.kernel_inputs = kwargs.get("approx_kernel_inputs", None)

            self.perm_rows_lin_op = None
            self.perm_cols_lin_op = None

        def num_outputs_per_input(self, x1, x2):
            """
            Given `n` data points `x1` and `m` datapoints `x2`, this multitask
            kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
            """
            return self.num_tasks

        #def forward(self, X, Z=None, common_terms=None):
        def forward(self, x1, x2, diag=False, **params):
            if diag:
                raise NotImplementedError("Diagonal forward not implemented for LODE_Kernel")
            common_terms = params["common_terms"]
            model_parameters = self.model_parameters
            if not x2 is None:
                common_terms["t_diff"] = x1-x2.t()
                common_terms["t_sum"] = x1+x2.t()
                common_terms["t_ones"] = torch.ones_like(x1+x2.t())
                common_terms["t_zeroes"] = torch.zeros_like(x1+x2.t())

            # OLD
            # if self.approx:
            #    # we need one approx se common term for each alpha parameter
            #    for param_name in filter(lambda s: 'alpha' in s, model_parameters.keys()):
            #        # get the param index
            #        idx = re.search(r'[0-9]+', param_name).group(0)
            #        common_terms['approx_se_' + idx] = approx_rbf_covariance(
            #            x1,
            #            x2,
            #            torch.exp(model_parameters['lengthscale_' + idx]),
            #            torch.exp(model_parameters[param_name]),
            #            200
            #        )

            # we need to compute each kernel term on every forward call, however we only need to compute it once
            # -> remember which kernel term was already computed this forward call
            computed_kernel = set()
            K_list = list()
            for rownum, row in enumerate(self.covar_description):
                for cell in row:
                    if self.approx:
                        # add approx se common term for each diffed se kernel
                        for match in re.findall(r'common_terms\["approx_se_([0-9]+)_d1([0-9]+)_d2([0-9]+)"]', cell):
                            if match not in computed_kernel:
                                idx, derive_x1, derive_x2 = match
                                common_terms[f"approx_se_{idx}_d1{derive_x1}_d2{derive_x2}"] = approx_rbf_covariance(
                                    x1,
                                    x2,
                                    torch.exp(model_parameters['lengthscale_' + idx]) + float(1e-07),
                                    torch.exp(model_parameters['alpha_' + idx]) + float(1e-07),
                                    self.number_of_eigenvalues,
                                    diff_order_x1=int(derive_x1),
                                    diff_order_x2=int(derive_x2)
                                )
                                computed_kernel.add(match)
                        for match in re.findall(r'common_terms\["(s_[0-9]+_[0-9]+)_d1([0-9]+)_d2([0-9]+)"]', cell):
                            if match not in computed_kernel:
                                kernel_name, derive_x1, derive_x2 = match
                                kernel_input = self.kernel_inputs[kernel_name]
                                common_terms[f"{kernel_name}_d1{derive_x1}_d2{derive_x2}"] = sinusoidal_kernel_lo(
                                    x1,
                                    x2,
                                    int(kernel_input["j"]),
                                    float(kernel_input["a"]),
                                    float(kernel_input["b"]),
                                    int(derive_x1),
                                    int(derive_x2)
                                )
                                computed_kernel.add(match)
                        for match in re.findall(r'common_terms\["(ex_[0-9]+_[0-9]+)_d1([0-9]+)_d2([0-9]+)"]', cell):
                            if match not in computed_kernel:
                                kernel_name, derive_x1, derive_x2 = match
                                kernel_input = self.kernel_inputs[kernel_name]
                                common_terms[f"{kernel_name}_d1{derive_x1}_d2{derive_x2}"] = exponential_kernel_lo(
                                    x1,
                                    x2,
                                    int(kernel_input["j"]),
                                    float(kernel_input["a"]),
                                    int(derive_x1),
                                    int(derive_x2)
                                )
                                computed_kernel.add(match)

                    # invoke cell (MatmulLinearOperator is bad operand type for unary -)
                    K_list.append(eval('0' + cell if cell[0] == '-' else cell))
            kernel_count = len(self.covar_description)
            # from https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/6
            #if K_list[0].ndim == 1:
            #    K_list = [kk.unsqueeze(1) for kk in K_list]
            if not self.approx:
                return einops.rearrange(K_list, '(t1 t2) h w -> (h t1) (w t2)', t1=kernel_count, t2=kernel_count)

            else:
                # the following does exactly the same as the einops.rearrange(...) call (needed for linear_operator)
                # TODO: Optimize building of list (eval covar_description directly into K_lin_op, will drastically help readability as well)
                # concatenate corresponding linear operator in dim 1 and further concatenate into single linear operator
                K_lin_op = CatLinearOperator(
                    *[
                        CatLinearOperator(
                            *[to_linear_operator(K_list[i + j * kernel_count]) for i in range(kernel_count)],
                            dim=1,
                            output_device=K_list[0].device
                        ) for j in range(kernel_count)
                    ],
                    dim=0,
                    output_device=K_list[0].device
                )

                # prepare linear operators for permutation
                h, w = K_list[0].shape
                perm_rows = torch.tensor([j * h + i for i in range(h) for j in range(kernel_count)]).to(K_list[0].device)
                perm_cols = torch.tensor([j * w + i for i in range(w) for j in range(kernel_count)]).to(K_list[0].device)
                self.perm_rows_lin_op = PermutationLinearOperator(perm_rows, dtype=K_lin_op.dtype)
                self.perm_cols_lin_op = PermutationLinearOperator(perm_cols, dtype=K_lin_op.dtype)

                # permutate rows and columns
                return self.perm_rows_lin_op @ K_lin_op @ self.perm_cols_lin_op.mT


def create_kernel_matrix_from_diagonal(D, approx=False, **kwargs):
    base_kernel = kwargs["base_kernel"] if "base_kernel" in kwargs else "SE_kernel"
    if approx and base_kernel != "SE_kernel":
        raise NotImplementedError("Approximation is only implemented for base_kernel=SE_kernel.")
    if base_kernel == "Matern_kernel_32":
        sqrt_3 = sqrt(3)
        base_kernel_expression = lambda i : globals()[f"signal_variance_{i}"]**2 * (1 + sqrt_3*((abs(t1 - t2)))/globals()[f"lengthscale_{i}"])*exp(-sqrt_3*((abs(t1 - t2)))/globals()[f"lengthscale_{i}"])
    elif base_kernel == "Matern_kernel_52":
        sqrt_5 = sqrt(5)
        base_kernel_expression = lambda i : globals()[f"signal_variance_{i}"]**2 * (1 + sqrt_5*((abs(t1 - t2)))/globals()[f"lengthscale_{i}"] + 5*(t1-t2)**2/(3*globals()[f"lengthscale_{i}"]**2))*exp(-sqrt_5*((abs(t1 - t2)))/globals()[f"lengthscale_{i}"])
    elif base_kernel == "SE_kernel":
        if approx:
            base_kernel_expression = lambda i: globals()[f"signal_variance_{i}"]**2 * globals()[f"approx_se_{i}"]
        else:
            base_kernel_expression = lambda i : globals()[f"signal_variance_{i}"]**2 * exp(-1/2*(t1-t2)**2/globals()[f"lengthscale_{i}"]**2)
    t1, t2 = var("t1, t2")
    translation_dictionary = dict()
    param_dict = torch.nn.ParameterDict()
    #sage_covariance_matrix = [[0 for cell in range(max(len(D.rows()), len(D.columns())))] for row in range(max(len(D.rows()), len(D.columns())))]
    sage_covariance_matrix = [[0 for cell in range(len(D.columns()))] for row in range(len(D.columns()))]
    #for i in range(max(len(D.rows()), len(D.columns()))):
    approx_kernel_inputs = dict() if approx else None
    for i in range(len(D.columns())):
        if i > len(D.diagonal())-1:
            entry = 0
        else:
            entry = D[i][i]
        var(f"LODEGP_kernel_{i}")
        if entry == 0:
            param_dict[f"signal_variance_{i}"] = torch.nn.Parameter(torch.tensor(float(0.)))
            param_dict[f"lengthscale_{i}"] = torch.nn.Parameter(torch.tensor(float(0.)))
            # Create an SE kernel
            var(f"signal_variance_{i}")
            var(f"lengthscale_{i}")
            if approx:
                param_dict[f"alpha_{i}"] = torch.nn.Parameter(torch.tensor(float(0.)))
                var(f"approx_se_{i}")
            #translation_dictionary[f"LODEGP_kernel_{i}"] = globals()[f"signal_variance_{i}"]**2 * exp(-1/2*(t1-t2)**2/globals()[f"lengthscale_{i}"]**2)
            translation_dictionary[f"LODEGP_kernel_{i}"] = base_kernel_expression(i)
        elif entry == 1:
            translation_dictionary[f"LODEGP_kernel_{i}"] = 0 
        else:
            kernel_translation_kernel = 0
            roots = entry.roots(ring=CC)
            roots_copy = cp.deepcopy(roots)
            for rootnum, root in enumerate(roots):
                # Complex root, i.e. sinusoidal exponential
                #if root[0].is_complex():
                param_dict[f"signal_variance_{i}_{rootnum}"] = torch.nn.Parameter(torch.tensor(float(0.)))
                var(f"signal_variance_{i}_{rootnum}")
                if root[0].imag() != 0.0:
                    # Check to prevent conjugates creating additional kernels
                    if not root[0].conjugate() in [r[0] for r in roots_copy]:
                        continue

                    # If it doesn't exist then it's new so find and pop the complex conjugate of the current root
                    roots_copy.remove((root[0].conjugate(), root[1]))
                    roots_copy.remove(root)

                    # Create sinusoidal kernel
                    a = root[0].real()
                    b = root[0].imag()
                    j = root[1]
                    if approx:
                        var(f"s_{i}_{rootnum}")
                        approx_kernel_inputs[f"s_{i}_{rootnum}"] = {"a": a, "b": b, "j": j}
                        kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2 * globals()[f"s_{i}_{rootnum}"]
                    else:
                        var("exponent_runner")
                        kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2*sum(t1**globals()["exponent_runner"] * t2**globals()["exponent_runner"], globals()["exponent_runner"], 0, j-1) *\
                                                        exp(a*(t1 + t2)) * cos(b*(t1-t2))
                else:
                    a = root[0]
                    j = root[1]
                    if approx:
                        var(f"ex_{i}_{rootnum}")
                        approx_kernel_inputs[f"ex_{i}_{rootnum}"] = {"a": a, "j": j}
                        kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2 * globals()[f"ex_{i}_{rootnum}"]
                    else:
                        var("exponent_runner")
                        # Create the exponential kernel functions
                        kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2*sum(t1**globals()["exponent_runner"] * t2**globals()["exponent_runner"], globals()["exponent_runner"], 0, j-1) * exp(a*(t1+t2))
            translation_dictionary[f"LODEGP_kernel_{i}"] = kernel_translation_kernel 
        sage_covariance_matrix[i][i] = globals()[f"LODEGP_kernel_{i}"]
    return sage_covariance_matrix, translation_dictionary, param_dict, approx_kernel_inputs


def _dim_helper(tens):
    while tens.dim() < 2:
        tens = tens.unsqueeze(-1)
    return tens


def exponential_kernel_lo(x1, x2, j, a, diff_order_x1=0, diff_order_x2=0):
    # get inputs in desired dimensions
    x1, x2 = tuple(_dim_helper(tens) for tens in (x1, x2))

    def psi(x, k=0):
        i = torch.arange(j, device=x.device)
        exp_ax = torch.exp(a * x)

        if k == 0:
            return to_linear_operator(x**i * exp_ax)

        else:
            # create summation index m
            m = torch.arange(k + 1, device=x.device).unsqueeze(-1)

            # compute everything independent of x
            k_tens = torch.tensor(k, device=x.device)
            binoms = torch.exp(
                torch.lgamma(k_tens + 1) -
                torch.lgamma(m + 1) +
                torch.lgamma(i + 1) -
                torch.lgamma(k_tens - m + 1) -
                torch.lgamma(i - (k_tens - m) + 1)
            )
            factors_ = a**m * binoms

            # sum over m
            sum_ = 0
            for m_idx in range(k + 1):
                # TODO: try implementation using large tensor and torch.sum() -> will use vastly more memory
                #  , however this might run faster on cuda
                sum_ += x**(i-(k_tens-m_idx)).clamp(min=0) * factors_[m_idx, :]

            return to_linear_operator(exp_ax * sum_)

    psi_x1 = psi(x1, diff_order_x1)
    if torch.equal(x1, x2) and diff_order_x1 == diff_order_x2:
        psi_x2 = psi_x1
    else:
        psi_x2 = psi(x2, diff_order_x2)

    return psi_x1 @ psi_x2.mT


def sinusoidal_kernel_lo(x1, x2, j, a, b, diff_order_x1=0, diff_order_x2=0):
    # get inputs in desired dimensions
    x1, x2 = tuple(_dim_helper(tens) for tens in (x1, x2))

    def psi(x, k=0):
        i = torch.arange(j, device=x.device)
        exp_ax = torch.exp(a * x)

        if k == 0:
            return (to_linear_operator(x**i * exp_ax * torch.cos(b * x)),
                    to_linear_operator(x**i * exp_ax * torch.sin(b * x)))

        else:
            k_tens = torch.tensor(k, device=x.device)
            # sum over l
            sum_l_c = 0
            sum_l_s = 0
            cos_bx = torch.cos(b*x)
            sin_bx = torch.sin(b*x)
            for l_idx in range(k + 1):
                l_tens = torch.tensor(l_idx, device=x.device)
                # create summation index m
                m = torch.arange(k - l_idx + 1, device=x.device).unsqueeze(-1)

                noms = torch.exp(
                    torch.lgamma(k_tens + 1) -
                    torch.lgamma(l_tens + 1) -
                    torch.lgamma(m + 1) +
                    torch.lgamma(i + 1) -
                    torch.lgamma(k_tens - l_tens - m + 1) -
                    torch.lgamma(i - (k_tens - l_tens - m) + 1)
                )
                factors_ = a**l_idx * noms

                # sum over m
                sum_m_c = 0
                sum_m_s = 0
                for m_idx in range(k-l_idx + 1):
                    if m_idx % 4 == 0:
                        dcos_dx = b**m_idx * cos_bx
                        dsin_dx = b**m_idx * sin_bx
                    elif m_idx % 4 == 1:
                        dcos_dx = -b**m_idx * sin_bx
                        dsin_dx = b**m_idx * cos_bx
                    elif m_idx % 4 == 2:
                        dcos_dx = -b**m_idx * cos_bx
                        dsin_dx = -b**m_idx * sin_bx
                    elif m_idx % 4 == 3:
                        dcos_dx = b**m_idx * sin_bx
                        dsin_dx = -b**m_idx * cos_bx
                    clamped_factors = x**(i-(k_tens-l_idx-m_idx)).clamp(min=0) * factors_[m_idx, :]
                    sum_m_c += clamped_factors * dcos_dx
                    sum_m_s += clamped_factors * dsin_dx

                sum_l_c += sum_m_c
                sum_l_s += sum_m_s

            return (to_linear_operator(exp_ax * sum_l_c),
                    to_linear_operator(exp_ax * sum_l_s))

    psi_c_x1, psi_s_x1 = psi(x1, diff_order_x1)
    if torch.equal(x1, x2) and diff_order_x1 == diff_order_x2:
        psi_c_x2, psi_s_x2 = psi_c_x1, psi_s_x1
    else:
        psi_c_x2, psi_s_x2 = psi(x2, diff_order_x2)

    return  psi_c_x1 @ psi_c_x2.mT + psi_s_x1 @ psi_s_x2.mT


def build_dict_for_SR_expression(expression, dx1, dx2):
    final_dict = {}
    dx1 = var("dx1")
    dx2 = var("dx2")
    for coeff_dx1 in expression.coefficients(dx1):
        final_dict.update({(Integer(coeff_dx1[1]), Integer(coeff_dx2[1])): coeff_dx2[0] for coeff_dx2 in coeff_dx1[0].coefficients(dx2)})
    return final_dict


def add_coeffs_to_name(subbed, name, d1, d2):
    subbed = SR(subbed)
    if "t1" in str(subbed) or "t2" in str(subbed):
        # sanity check
        raise RuntimeError(
            f"Cannot approximate:\n{subbed}")
    # do not diff yet (will be done by kernel), but add coeffs to name
    diffed_name = f"{name}_d1{d1}_d2{d2}"
    var(diffed_name)
    return str(subbed.substitute(globals()[name] == globals()[diffed_name]))


def differentiate_kernel_matrix(K, V, Vt, kernel_translation_dictionary, dx1, dx2, approx=False, **kwargs):
    """
    This code takes the sage covariance matrix and differentiation matrices
    and returns a list of lists containing the results of the `compile` 
    commands that calculate the respective cov. fct. entry
    """
    base_kernel = kwargs["base_kernel"] if "base_kernel" in kwargs else "SE_kernel"
    sage_multiplication_kernel_matrix = matrix(K.base_ring(), len(K[0]), len(K[0]), (V*K*Vt))
    final_kernel_matrix = [[None for i in range(len(K[0]))] for j in range(len(K[0]))]
    for i, row in  enumerate(sage_multiplication_kernel_matrix):
        for j, cell in enumerate(row):
            cell_expression = 0
            diff_dictionary = build_dict_for_SR_expression(cell, dx1, dx2)
            for coeffs in diff_dictionary:
                orig_cell_expression = diff_dictionary[coeffs]
                if approx:
                    # we need to consider each summand separately # TODO: not sure if this is still true
                    if orig_cell_expression.operator() == add_vararg:
                        summands = orig_cell_expression.operands()
                    else:
                        summands = [orig_cell_expression]

                    temp_cell_expression = 0
                    for summand in summands:
                        for kernel_translation in kernel_translation_dictionary:
                            if kernel_translation in str(summand):
                                subbed = summand.substitute(
                                    globals()[kernel_translation] == kernel_translation_dictionary[kernel_translation])

                                expr = str(subbed)
                                if approx:
                                    for match in re.findall(r'approx_se_[0-9]+', expr):
                                        expr = add_coeffs_to_name(expr, match, coeffs[0], coeffs[1])
                                    for match in re.findall(r's_[0-9]+_[0-9]+', expr):
                                        expr = add_coeffs_to_name(expr, match, coeffs[0], coeffs[1])
                                    for match in re.findall(r'ex_[0-9]+_[0-9]+', expr):
                                        expr= add_coeffs_to_name(expr, match, coeffs[0], coeffs[1])

                                    temp_cell_expression += SR(expr)

                                else:
                                    temp_cell_expression += SR(subbed).diff(t1, coeffs[0]).diff(t2, coeffs[1])
                    cell_expression += SR(temp_cell_expression)
                else:
                    # temp_cell_expression = mul([K[i][i] for i, multiplicant in enumerate(summand[3:]) if multiplicant > 0])
                    temp_cell_expression = orig_cell_expression
                    for kernel_translation in kernel_translation_dictionary:
                        if kernel_translation in str(temp_cell_expression):
                            temp_cell_expression = SR(temp_cell_expression)
                            # cell = cell.factor()
                            # replace
                            temp_cell_expression = temp_cell_expression.substitute(
                                globals()[kernel_translation] == kernel_translation_dictionary[kernel_translation])
                        # And now that everything is replaced: diff that bad boy!
                    cell_expression += SR(temp_cell_expression).diff(t1, coeffs[0]).diff(t2, coeffs[1])

            if base_kernel == "Matern_kernel_52" or base_kernel == "Matern_kernel_32":
                var("r")
                var("t1, t2")
                assume(r, "real")
                assume(t1, "real")
                assume(t2, "real")
                final_kernel_matrix[i][j] =cell_expression.subs(t1=r+t2).factor().expand().simplify().factor().subs(r=t1-t2).subs({sqrt(5):sqrt(5).n(), sqrt(3):sqrt(3).n()})
            else:
                final_kernel_matrix[i][j] = cell_expression
    return final_kernel_matrix 


def replace_sum_and_diff(kernelmatrix, sumname="t_sum", diffname="t_diff", onesname="t_ones", zerosname="t_zeroes"):
    result_kernel_matrix = cp.deepcopy(kernelmatrix)
    var(sumname, diffname)
    for i, row in enumerate(kernelmatrix):
        for j, cell in enumerate(row):
            # Check if the cell is just a number
            if type(cell) == sage.symbolic.expression.Expression and not cell.is_numeric():
                #result_kernel_matrix[i][j] = cell.substitute({t1-t2:globals()[diffname], t1+t2:globals()[sumname]})
                result_kernel_matrix[i][j] = cell.substitute({t1:0.5*globals()[sumname] + 0.5*globals()[diffname], t2:0.5*globals()[sumname] - 0.5*globals()[diffname]})
            # This case is assumed to be just a constant, but we require it to be of 
            # the same size as the other covariance submatrices
            else:
                if cell == 0:
                    var(zerosname)
                    result_kernel_matrix[i][j] = globals()[zerosname]
                else:
                    var(onesname)
                    result_kernel_matrix[i][j] = cell * globals()[onesname]
    return result_kernel_matrix


def replace_basic_operations(kernel_string):
    # Define the regex replacement rules for the text
    regex_replacements_multi_group = {
        "exp" : [r'(e\^)\((([^()]*|\(([^()]*|\([^()]*\))*\))*)\)', "torch.exp"],
        # TODO fix this
        "sqrt" : [r'sqrt(\((([^()]*|\(([^()]*|\([^()]*\))*\))*)\))', "torch.sqrt"],
        "exp_singular" : [r'(e\^)([0-9a-zA-Z_]*)', "torch.exp"]
    }
    regex_replacements_single_group = {
        "sin" : [r'sin', "torch.sin"],
        "cos" : [r'cos', "torch.cos"],
        "pow" : [r'\^', "**"]
    }
    # OLD - first try to replace approximated se
    # if approx:
    #     # substitute squared exponential with its approximation
    #     pattern = r'e\^\(-0\.5\*t_diff\^2/(lengthscale_([0-9]+))\^2\)'
    #     m = re.search(pattern, kernel_string)
    #     if not m is None:
    #         # kernel_string = re.sub(pattern, create_squared_exponential_approx_string(200), kernel_string)
    #         kernel_string = re.sub(pattern, r'common_terms["approx_se_\2"]', kernel_string)
    for replace_term in regex_replacements_multi_group:
        m = re.search(regex_replacements_multi_group[replace_term][0], kernel_string)
        if not m is None:
            # There is a second group, i.e. we have exp(something)
            kernel_string = re.sub(regex_replacements_multi_group[replace_term][0], f'{regex_replacements_multi_group[replace_term][1]}'+r"(\2)", kernel_string)
    for replace_term in regex_replacements_single_group:
        m = re.search(regex_replacements_single_group[replace_term][0], kernel_string)
        if not m is None:
            kernel_string = re.sub(regex_replacements_single_group[replace_term][0], f'{regex_replacements_single_group[replace_term][1]}', kernel_string)

    return kernel_string 


def replace_parameters(kernel_string, model_parameters, common_terms = []):
    regex_replace_string = r"(^|[\*\+\/\(\)\-\s])(REPLACE)([\*\+\/\(\)\-\s\.]|$)"
    
    for term in common_terms:
        if term in kernel_string:
            kernel_string = re.sub(regex_replace_string.replace("REPLACE", term), r"\1" + f"common_terms[\"{term}\"]" + r"\3", kernel_string)

    for model_param in model_parameters:
        kernel_string = re.sub(regex_replace_string.replace("REPLACE", model_param), r"\1"+f"(torch.exp(model_parameters[\"{model_param}\"]) + 1e-07)"+r"\3", kernel_string)

    return kernel_string 


def verify_sage_entry(kernel_string, local_vars):
    # This is a call to willingly produce an error if the string is not originally coming from sage
    try:
        if type(kernel_string) == sage.symbolic.expression.Expression:
            kernel_string = kernel_string.simplify()
        kernel_string = str(kernel_string)
        sage_eval(kernel_string, locals = local_vars)
    except Exception as E:
        raise Exception(f"The string was not safe and has not been used to construct the Kernel.\nPlease ensure that only valid operations are part of the kernel and all variables have been declared.\nYour kernel string was:\n'{kernel_string}'")


def replace_approx_terms(kernel_string):
    subbed_se = re.sub('(approx_se_[0-9]+_d1[0-9]+_d2[0-9]+)', r'common_terms["\1"]', kernel_string)
    subbed_ex = re.sub('(ex_[0-9]+_[0-9]+_d1[0-9]+_d2[0-9]+)', r'common_terms["\1"]', subbed_se)
    return re.sub('(s_[0-9]+_[0-9]+_d1[0-9]+_d2[0-9]+)', r'common_terms["\1"]', subbed_ex)


def translate_kernel_matrix_to_gpytorch_kernel(kernelmatrix, paramdict, common_terms=[], approx=False):
    kernel_call_matrix = [[] for i in range(len(kernelmatrix))]
    for rownum, row in enumerate(kernelmatrix):
        for colnum, cell in enumerate(row):
            # First thing I do: Verify that the entry is a valid sage command
            local_vars = {str(v):v for v in SR(cell).variables()}
            verify_sage_entry(cell, local_vars)
            # Now translate the cell to a call
            replaced_op_cell = replace_basic_operations(str(cell))
            replaced_var_cell = replace_parameters(replaced_op_cell, paramdict, common_terms)
            if approx:
                replaced_var_cell = replace_approx_terms(replaced_var_cell)
            #print("DEBUG: replaced_var_cell:")
            #print(replaced_var_cell)
            # kernel_call_matrix[rownum].append(compile(replaced_var_cell, "", "eval"))
            kernel_call_matrix[rownum].append(replaced_var_cell)
    return kernel_call_matrix
