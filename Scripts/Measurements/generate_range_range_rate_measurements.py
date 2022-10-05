import numpy as np
import os
from StatOD.measurements import get_rho_rhod_el
import pickle

def generate_measurements_asteroid(traj_file):
    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]
    X_sc = traj_data["X"] # SC traj w.r.t. asteroid
    X_asteroid = np.full((len(t),6),np.array([[0,0,0,0,0,0]]))

    elevation_mask = np.deg2rad(10) # doesn't matter because observer at origin
    rho, rho_dot, el = get_rho_rhod_el(t, X_sc, X_asteroid, elevation_mask)

    sigma_rho = 1E-3 # 1 m
    sigma_rhod = 1E-6 # 1 mm/s

    noisy_rho = rho + np.random.normal(0, sigma_rho, size=np.shape(rho))
    noisy_rho_dot = rho_dot + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot))


    Y = np.hstack((rho, rho_dot))
    Y_noisy = np.hstack((noisy_rho, noisy_rho_dot))

    true_measurements = {
        'time' : t,
        'measurements' : Y,
        'h_args' : X_asteroid,
    }

    noisy_measurements = {
        'time' : t,
        'measurements' : Y_noisy,
        'h_args' : X_asteroid,
    }

    directory = 'Data/Measurements/Range'
    os.makedirs(directory, exist_ok=True)
    meas_file = os.path.basename(traj_file).split('.')[0] + "_meas"
    with open(f'{directory}/{meas_file}_noiseless.data', 'wb') as f:
        pickle.dump(true_measurements, f)

    with open(f'{directory}/{meas_file}_noisy.data', 'wb') as f:
        pickle.dump(noisy_measurements, f)

if __name__ == '__main__':
    generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_equitorial.data')
    generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_high_alt_30_timestep.data')
