import multiprocessing as mp
import os

from GravNN.Support.slurm_utils import get_available_cores

import StatOD
from Scripts.Scenarios.DMC_high_fidelity import DMC_high_fidelity
from StatOD.utils import *
from StatOD.utils import dict_values_to_list


def run_catch(args, halt=False):
    finished = False
    while not finished:
        try:
            os.path.dirname(StatOD.__file__)

            args_list = dict_values_to_list(args)

            if args_list["pinn_file"][0] == "pm":
                pinn_file = "Data/Dataframes/eros_pm_061023.data"
                traj_file = "traj_eros_pm_061023_32000.0_0.2"
            else:
                pinn_file = "Data/Dataframes/eros_poly_061023.data"
                traj_file = "traj_eros_poly_061023_32000.0_0.2"

            config = DMC_high_fidelity(
                pinn_file,
                traj_file,
                args_list,
                show=False,
            )
            finished = True
        except Exception as e:
            # Try it twice, if it fails, then it's a real error
            if halt is True:
                print(e)
            else:
                run_catch(args, halt=True)
    return config


def main(date):
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

    save_df = (
        os.path.dirname(StatOD.__file__)
        + f"/../Data/Dataframes/hparam_search_{date}.data"
    )
    # append lock to the list of arg tupl
    threads = get_available_cores()
    args = format_args(hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run_catch, args)
        configs = results.get()
    save_results(save_df, configs)


if __name__ == "__main__":
    main("061123")
