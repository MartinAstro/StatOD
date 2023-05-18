import multiprocessing as mp
import os

import StatOD
from Scripts.Experiments.EKF_PINN_DMC_pos_rotating import EKF_Rotating_Scenario
from Scripts.Scenarios.helper_functions import *
from StatOD.utils import dict_values_to_list


def run_catch(args):
    finished = False
    while not finished:
        try:
            statOD_dir = os.path.dirname(StatOD.__file__)
            pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"
            traj_file = "traj_rotating_gen_III_constant"
            args_list = dict_values_to_list(args)
            config = EKF_Rotating_Scenario(pinn_file, traj_file, args_list, show=False)
            finished = True
        except Exception as e:
            print(e)
    return config


def main():
    hparams = {
        "q_value": [1e-9, 1e-8, 1e-7],
        "epochs": [10, 50, 100],
        "learning_rate": [1e-4, 1e-5, 1e-6],
        "batch_size": [256, 1024, 2048],
        "train_fcn": ["pinn_a", "pinn_al"],
        "boundary_condition_data": [True, False],
        "data_fraction": [0.1, 0.5, 1.0],
    }

    hparams = {
        "q_value": [1e-9],
        "epochs": [10],
        "learning_rate": [1e-4, 1e-5],
        "batch_size": [2048],
        "train_fcn": ["pinn_a"],
        "data_fraction": [0.1],
    }

    save_df = os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/hparam_test.data"

    threads = 2
    args = format_args(hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run_catch, args)
        configs = results.get()
    save_results(save_df, configs)


if __name__ == "__main__":
    main()
