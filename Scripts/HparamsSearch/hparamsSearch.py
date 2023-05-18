import multiprocessing as mp

from Scripts.Scenarios.helper_functions import *
from Scripts.Experiments.EKF_PINN_DMC_pos_rotating import EKF_Rotating_Scenario
import os
import StatOD


def run_catch(args):
    finished = False
    while not finished:
        try:
            config = EKF_Rotating_Scenario(args)
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
    }

    save_df = (
        os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/hparams_rotating.data"
    )


    threads = 6
    args = format_args(hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run_catch, args)
        configs = results.get()
    save_results(save_df, configs)


if __name__ == "__main__":
    main()
