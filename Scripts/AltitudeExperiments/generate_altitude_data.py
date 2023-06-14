import pickle
import sys

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros

from Scripts.Scenarios.DMC_high_fidelity import DMC_high_fidelity
from StatOD.utils import *


def get_model_params():
    hparams = {
        "r_value": [1e-12],
        "epochs": [1000],
        "learning_rate": [1e-4],
        "batch_size": [2**19],
        "train_fcn": ["pinn_a"],
        "boundary_condition_data": [False],
        "measurement_noise": ["noiseless"],
        "eager": [False],
        "data_fraction": [1.0],
    }
    return hparams


def main(pinn_model):
    radius = Eros().radius
    a_list = np.arange(2, 3, 0.1) * radius
    e_list = np.arange(0.0, 0.5, 0.05)

    pinn_file = f"Data/Dataframes/{pinn_model}.data"

    traj_files = []
    for a in a_list:
        for e in e_list:
            traj_file = f"traj_eros_pm_061023_{a}_{e}"
            traj_files.append(traj_file)

    # use command line argument to select the index
    idx = int(sys.argv[1])
    traj_file = traj_files[idx]
    
    # get orbital elements from trajectory
    e = float(traj_file.split("_")[-1])
    a = float(traj_file.split("_")[-2])

    # run the DMC experiment
    config = DMC_high_fidelity(
        pinn_file,
        traj_file,
        get_model_params(),
        show=False,
    )

    # get hparams from config
    metrics = {}
    metrics.update({"Planes" : config["Planes"]})
    metrics.update({"Trajectory" : config["Trajectory"]})
    metrics.update({"Extrapolation" : config["Extrapolation"]})

    # save the data
    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    os.makedirs(f"{statOD_dir}/Data/TrajMetrics/", exist_ok=True)
    with open(f"{statOD_dir}/Data/TrajMetrics/{traj_file}_a-{a}_e-{e}.data", "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    # pinn_model = "eros_poly_061323"
    # main(pinn_model)

    pinn_model = "eros_pm_061323"
    main(pinn_model)
