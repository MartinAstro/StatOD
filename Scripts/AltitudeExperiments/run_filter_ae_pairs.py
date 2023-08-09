import pickle
import sys

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros

import StatOD
from Scripts.Scenarios.DMC_high_fidelity import DMC_high_fidelity
from StatOD.utils import *


def get_model_params():
    hparams = {
        "r_value": [1e-12],
        "q_value": [1e-7],
        "epochs": [1000],
        "learning_rate": [1e-4],
        "batch_size": [1024],
        "meas_batch_size": [32864],
        "train_fcn": ["pinn_a"],
        "boundary_condition_data": [False],
        "measurement_noise": ["noiseless"],
        "eager": [False],
        "data_fraction": [0.33],
    }
    return hparams


def main(pinn_file, idx, show=False, df_file=None):
    radius = Eros().radius
    a_list = np.arange(2, 3, 0.1) * radius
    e_list = np.arange(0.0, 0.5, 0.05)

    pinn_model = os.path.basename(pinn_file).split(".")[0]
    # returns eros_pm_061323

    traj_files = []
    for a in a_list:
        for e in e_list:
            traj_file = f"traj_{pinn_model}_{a}_{e}"
            traj_files.append(traj_file)

    # use command line argument to select the index
    traj_file = traj_files[idx]

    # get orbital elements from trajectory
    e = float(traj_file.split("_")[-1])
    a = float(traj_file.split("_")[-2])

    # run the DMC experiment

    config = DMC_high_fidelity(
        pinn_file,
        traj_file,
        get_model_params(),
        show,
        df_file=df_file,
    )

    # get hparams from config
    metrics = {}
    metrics.update({"Planes": config["Planes"]})
    metrics.update({"Trajectory": config["Trajectory"]})
    metrics.update({"Extrapolation": config["Extrapolation"]})

    # save the data
    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    os.makedirs(f"{statOD_dir}/Data/TrajMetrics/", exist_ok=True)
    with open(f"{statOD_dir}/Data/TrajMetrics/{traj_file}_a-{a}_e-{e}.data", "wb") as f:
        pickle.dump(metrics, f)


def local_main():
    idx = 6
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    pinn_file = f"{statOD_dir}Data/Dataframes/eros_statOD_pm_071123.data"
    # pinn_file = f"{statOD_dir}Data/Dataframes/eros_poly_061023.data"
    # model = "eros_poly_061323"

    df_file = f"{statOD_dir}Data/Dataframes/ae_pair_results.data"
    main(pinn_file, idx, show=True, df_file=df_file)


def HPC_main():
    model = sys.argv[1]
    idx = int(sys.argv[2])

    main(model, idx)


if __name__ == "__main__":
    HPC_main()
    # local_main()
