import os
import pickle

import numpy as np

import StatOD
from StatOD.measurements import h_pos


def generate_measurements(traj_file):
    with open(traj_file, "rb") as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]
    X_sc = traj_data["X"]  # SC traj w.r.t. asteroid
    X_asteroid = np.full((len(t), 6), np.array([[0, 0, 0, 0, 0, 0]]))

    Y = []
    for i in range(len(X_sc)):
        Y.append(h_pos(X_sc[i], X_asteroid[i]))
    Y = np.array(Y).squeeze()

    sigma_measurement = 1e-3
    noise = np.random.normal(0, sigma_measurement, size=np.shape(X_sc[:, 0:3]))

    true_measurements = {
        "time": t,
        "Y": Y,
        "h_args": X_asteroid,
    }
    noisy_measurements = {
        "time": t,
        "Y": Y + noise,
        "h_args": X_asteroid,
    }

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    directory = f"{statOD_dir}Data/Measurements/Position"
    os.makedirs(directory, exist_ok=True)
    meas_file = os.path.basename(traj_file).split(".")[0] + "_meas"
    with open(f"{directory}/{meas_file}_noiseless.data", "wb") as f:
        pickle.dump(true_measurements, f)

    with open(f"{directory}/{meas_file}_noisy.data", "wb") as f:
        pickle.dump(noisy_measurements, f)


if __name__ == "__main__":
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    generate_measurements(
        f"{statOD_dir}Data/Trajectories/traj_eros_poly_053123.data",
    )
