#!/usr/bin/env python3
import logging
from warnings import catch_warnings
from pathlib import Path
import shutil
import re
import gc
import time
import argparse

import torch
import gpytorch
from gpytorch.settings import max_cholesky_size, max_preconditioner_size

from lodegp.LODEGP import LODEGP

DIRECTORY_PATH = Path("./data/measure_run_time")


def log_warnings(rec):
    for w in rec:
        logging.warning(f"{w.category.__name__} - {w.message}",)


def generate_data(nb_data: int) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(2, 12, nb_data)
    y_func = [
        lambda x: 781/8000      * torch.sin(x) / x - 1/20      * torch.cos(x) / x**2 + 1/20      * torch.sin(x) / x**3,
        lambda x: 881/8000      * torch.sin(x) / x - 1/40      * torch.cos(x) / x**2 + 1/40      * torch.sin(x) / x**3,
        lambda x: 688061/800000 * torch.sin(x) / x - 2543/4000 * torch.cos(x) / x**2 + 1743/4000 * torch.sin(x) / x**3
                  - 3/5 * torch.cos(x) / x**4 + 3/5 * torch.sin(x) / x**5
    ]
    train_y = torch.stack([f(train_x) for f in y_func], dim=-1)
    
    return train_x, train_y


def train_gp(train_x: torch.Tensor, train_y: torch.Tensor, nb_eigenvalues: int, device: torch.device, loss_calc: str):
    # define model, optimizer and loss
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = LODEGP(
        train_x, train_y,
        likelihood,
        3,
        approx=nb_eigenvalues!=0, number_of_eigenvalues=nb_eigenvalues,
        ODE_name="Bipendulum",
        system_parameters={"l1": 1.0, "l2": 2.0}
    )
    model.to(device)
    likelihood.train()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if device == "cuda":
        # make sure everything is synchronized on cuda
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # train and measure time
    max_chol_size = train_x.size(0) * 3 + 1 if loss_calc == "Cholesky" else 1
    with catch_warnings(record=True) as warn_record:
        try:
            with max_cholesky_size(max_chol_size), max_preconditioner_size(0):
                start_t = None
                for i in range(1):
                    if i == 0:
                        # start timing after 5 warm up iterations
                        start_t = time.perf_counter()
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()
                time_elapsed = time.perf_counter() - start_t

            log_warnings(warn_record)
            logging.info(f"Training completed in {time_elapsed:.2f} seconds.")
            return time_elapsed, None, model

        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
            # Accept some error and return the error message
            log_warnings(warn_record)
            logging.error(type(e).__name__ + " - " + str(e) + "\n")
            return time.perf_counter() - start_t, str(e), None

        except Exception as e:
            # Unexpected error -> log and raise
            log_warnings(warn_record)
            logging.error(type(e).__name__ + " - " + str(e))
            raise e


def covariance_to_file(train_x, model, fp):
    try:
        with torch.no_grad():
            covar = model(train_x).covariance_matrix
            torch.save(covar, fp)
        logging.info(f"Covariance tensor successfully written to {fp}\n")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # covariance matrix will be dense tensor instead of linear operator -> more memory usage as in training
        logging.error(type(e).__name__ + " - " + str(e) + "\n")


def get_run(run_id, fp):
    with open(fp, "r", encoding="utf-8") as f:
        # discard headers
        f.readline()

        # check every line separately to minimize memory usage
        while line := f.readline():
            # split at every comma, except those in double quotes
            data = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line.replace("\n", ""))
            if run_id == data[0]:
                e = data[2][1:-1]
                return float(data[1]), e if e != "None" else None
        return None


def handle_os_kill():
    # TODO: check csv file / log. if last entry did not finish -> insert run as Exception "killed by OS"
    #  (consider KeyboardInterrupt as not killed!)
    pass


def main():
    handle_os_kill()

    if not csv_file_path.is_file() or run_all:
        with open(csv_file_path, "w", encoding="utf-8") as f:
            # create / truncate file and add headers
            f.write("run_id,run_time,exception")

    # get all possible gp_ids (number data points, number eigenvalues, loss calc method, device)
    cuda_gp_ids = []
    cpu_gp_ids = []
    for nb_eigenvalues in range(0, 301, 50):
        loss_calc_methods = ["mBCG"] if nb_eigenvalues != 0 else ["Cholesky", "mBCG"]
        for loss_calc_method in loss_calc_methods:
            cuda_gp_ids.append(f"n{nb_eigenvalues}_{loss_calc_method}_cuda")
            cpu_gp_ids.append(f"n{nb_eigenvalues}_{loss_calc_method}_cpu")

    for all_gp_ids in [cuda_gp_ids, cpu_gp_ids]:
        # repeat training for increasing amount of data points until all gps fail
        failed = set()
        nb_data = 1000
        while not set(all_gp_ids).issubset(failed):
            for gp_id in all_gp_ids:
                if gp_id not in failed:
                    run_id = f"N{nb_data}_{gp_id}"
                    params = gp_id[1:].split("_")
                    nb_eigenvalues = int(params[0])
                    loss_calc_method = params[1]
                    device = torch.device(params[2])

                    if not run_all and (csv_data := get_run(run_id, csv_file_path)) is not None:
                        if csv_data[1] is not None:
                            failed.add(gp_id)
                        else:
                            succeded = True
                        logging.info(f"Found experiment {run_id} in csv file.\n")
                        continue

                    logging.info(f"Running experiment {run_id}")

                    # generate data
                    train_x, train_y = (data.to(device) for data in generate_data(nb_data))

                    # train model
                    run_time, exc, model = train_gp(train_x, train_y, nb_eigenvalues, device, loss_calc_method)

                    # write results to csv file
                    with open(csv_file_path, "a", encoding="utf-8") as f:
                        f.write(f'\n{run_id},{run_time},"{exc}"')

                    if model is None:
                        # run failed
                        failed.add(gp_id)
                        continue

                    # write covar to file
                    logging.info("Trying to write covariance tensor to file...")
                    covariance_to_file(train_x, model, covar_path / (run_id + ".pt"))

            nb_data += 1000


if __name__ == "__main__":
    # config command-line options
    def len_gt_0(s):
        if len(s) > 0:
            return s
        raise argparse.ArgumentTypeError(f"Output file name must contain at least one character")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-a", "--run_all",
        default=False,
        action="store_true",
        help="Truncate csv file and run all experiments."
    )
    arg_parser.add_argument(
        "--delete-addons",
        default=False,
        action="store_true",
        help="Delete all logs and covariances before running experiments. Will always run all experiments."
    )
    arg_parser.add_argument(
        "-o", "--output_file",
        default="experiments.csv",
        type=len_gt_0,
        help="Output file name."
    )

    # handle command-line arguments
    cmd_args = arg_parser.parse_args()
    run_all = cmd_args.run_all or cmd_args.delete_addons
    csv_file_path = DIRECTORY_PATH / cmd_args.output_file
    if csv_file_path.suffix != ".csv":
        csv_file_path = csv_file_path.parent / f"{csv_file_path.name}.csv"
    log_path = DIRECTORY_PATH / f"{csv_file_path.stem}_logs"
    covar_path = DIRECTORY_PATH / f"{csv_file_path.stem}_covariances"
    if cmd_args.delete_addons:
        if log_path.exists():
            shutil.rmtree(log_path)
        if covar_path.exists():
            shutil.rmtree(covar_path)

    # create directories
    if not log_path.exists():
        log_path.mkdir(parents=True)
    if not covar_path.exists():
        covar_path.mkdir(parents=True)

    # logging config
    utc_string = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    log_file_path = log_path / f"{utc_string}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="{levelname}:{name}: {message}", style="{",
        filename=log_file_path,
        encoding="utf-8",
        filemode="a"
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    if not torch.cuda.is_available():
        exception = RuntimeError("CUDA needs to be available.")
        logging.error(type(exception).__name__ + " - " + str(exception))
        raise exception

    # torch settings
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    # run experiments
    main()