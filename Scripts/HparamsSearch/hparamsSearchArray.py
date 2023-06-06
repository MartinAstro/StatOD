import os
import sys

from GravNN.Support.slurm_utils import get_available_cores

import StatOD
from Scripts.Scenarios.DMC_high_fidelity import DMC_high_fidelity
from StatOD.utils import *
from StatOD.utils import dict_values_to_list


def run_catch(args):
    args_list = dict_values_to_list(args)

    if args_list["pinn_file"][0] == "pm":
        pinn_file = "Data/Dataframes/eros_pm_053123.data"
        traj_file = "traj_eros_pm_053123"
    else:
        pinn_file = "Data/Dataframes/eros_poly_053123.data"
        traj_file = "traj_eros_poly_053123"

    config = DMC_high_fidelity(
        pinn_file,
        traj_file,
        args_list,
        show=False,
    )

    return config


def main():
    hparams = {
        "q_value": [1e-9, 1e-8, 1e-7],
        "epochs": [10, 100, 1000],
        "learning_rate": [1e-4, 1e-5, 1e-6],
        "batch_size": [256, 2048, 2**15],
        "train_fcn": ["pinn_a", "pinn_al"],
        "data_fraction": [0.3],
        "measurement_noise": ["noiseless", "noisy"],
        "pinn_file": ["pm", "poly"],
        "r_value": [1e-12, 1e-3],
        # "filter_type" : ["CKF", "EKF", "UKF"],
    }

    # append lock to the list of arg tupl
    get_available_cores()
    args = format_args(hparams)

    idx = int(sys.argv[1])
    args_i = args[idx][0]
    run_catch(args_i)


if __name__ == "__main__":
    main()
