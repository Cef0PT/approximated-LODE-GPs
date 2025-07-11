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
                for i in range(25):
                    if i == 5:
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
            return -1., str(e), None

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
        return None
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # covariance matrix will be dense tensor instead of linear operator -> more memory usage as in training
        logging.error(type(e).__name__ + " - " + str(e) + "\n")
        return str(e)


def get_run(run_id, fp):
    with open(fp, "r", encoding="utf-8") as f:
        # discard headers
        f.readline()

        # check every line separately to minimize memory usage
        e = []
        t = None
        while line := f.readline():
            # split at every comma, except those in double quotes
            data = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line.replace("\n", ""))
            if run_id == data[0]:
                t = float(data[1])
                e.append(data[2][1:-1])
        return t, list(filter(lambda s: s != "None", e))


def handle_os_kill(log_dir):
    """
        If we run an experiment or try to write a covariance matrix to file, we might use more memory than the OS
        allows. This can lead to the whole process being killed, by the OS. In those cases there are only two possible
        entries in the last line of the file.
        Either
        "INFO:root: Running experiment {run_id}" or
        "INFO:root: Trying to write covariance tensor to file..."
        In the second case the info of the run_id will be found in the 3rd line from the back.
            -> loop through lines and always save the last 3 lines; after loop, check if the last line corresponds to an
               OS kill and get the run_id depending on entry of last line (not necessarily the fastest solution, but
               fast enough and pretty simple; log file rather small)
    """
    # get last log file path (the actual last log file is this run -> get second to last log file)
    all_fps = sorted(log_dir.glob("*.log"))
    if len(all_fps) < 2:
        return
    fp = all_fps[-2]

    # get last three lines
    with open(fp, "r", encoding="utf-8") as f:
        # discard headers
        f.readline()
        last_three = [""] * 3
        while line := f.readline():
            last_three[:2] = last_three[1:]
            last_three[-1] = line

    # look for run_id in last and third to last lines
    # (we can assume this was an OS kill, if one of them contains the first case described in doc string)
    match = re.search(r"Running experiment (N[0-9]+_n[0-9]+_.+)\n", last_three[-1])
    if match is not None:
        error_message = "OS KILL"
        t = -1.
    else:
        match = re.search(r"Running experiment (N[0-9]+_n[0-9]+_.+)", last_three[0])
        error_message = "OS KILL file write"
        t, _ = get_run(match.group(1), csv_file_path)

    if match is not None:
        run_id = match.group(1)

        # add this run to csv file with exception "OS KILL"
        with open(csv_file_path, "a", encoding="utf-8") as f:
            f.write(f'\n{run_id},{t},"{error_message}"')

        logging.info("OS KILL detected, written data to csv file")


def main():
    handle_os_kill(log_path)

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
        failed_train = set()
        failed_file_write = set()
        nb_data = 1000
        while not set(all_gp_ids).issubset(failed_train):
            for gp_id in all_gp_ids:
                if gp_id not in failed_train:
                    run_id = f"N{nb_data}_{gp_id}"
                    params = gp_id[1:].split("_")
                    nb_eigenvalues = int(params[0])
                    loss_calc_method = params[1]
                    device = torch.device(params[2])

                    if not run_all and (csv_data := get_run(run_id, csv_file_path))[0] is not None:
                        if (len(csv_data[1]) >= 1 and
                                len(list(filter(lambda s: s != "OS KILL file write", csv_data[1]))) >= 1):
                            failed_train.add(gp_id)
                        if "OS KILL file write" in csv_data[1]:
                            # catch OS kills on file write to avoid crashes in future experiments
                            failed_file_write.add(gp_id)
                        logging.info(f"Found experiment {run_id} in csv file.\n")
                        continue

                    logging.info(f"Running experiment {run_id}")

                    # generate data
                    train_x, train_y = (data.to(device) for data in generate_data(nb_data))

                    # train model
                    run_time, e, model = train_gp(train_x, train_y, nb_eigenvalues, device, loss_calc_method)

                    # write results to csv file
                    with open(csv_file_path, "a", encoding="utf-8") as f:
                        f.write(f'\n{run_id},{run_time},"{e}"')

                    if model is None:
                        # run failed
                        failed_train.add(gp_id)
                        continue
                    elif gp_id in failed_file_write:
                        logging.info(f"Skipped writing of covariance function.\n")
                        continue

                    # write covar to file
                    logging.info("Trying to write covariance tensor to file...")
                    e_msg = covariance_to_file(train_x, model, covar_path / (run_id + ".pt"))
                    if e_msg is not None:
                        failed_file_write.add(gp_id)

            nb_data += 1000


if __name__ == "__main__":
    # config command-line options
    def len_gt_0(s):
        if len(s) > 0:
            return s
        raise argparse.ArgumentTypeError(f"Output file name must contain at least one character")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-a", "--run-all",
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
        "-o", "--output-file",
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
    try:
        main()
    except KeyboardInterrupt as exc:
        # log keyboard interrupts, so they are not confused with OS kills
        logging.error(type(exc).__name__)
        raise exc