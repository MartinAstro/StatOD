import numpy as np
import os
from StatOD.constants import ErosParams
from StatOD.data import get_measurements_pos
from StatOD.measurements import get_rho_rhod_el
import pickle

from StatOD.utils import ECEF_2_ECI, latlon2cart

def generate_measurements_asteroid(traj_file):
    """
    Generate range and range rate for a very simple asteroid scenario.
    Produce both truth and noisy measurements. 

    Assumptions:
    1: The observer is at a fixed point in inertial space 
       X_obs = [0,0,0, 0,0,0]
    2: The range and range-rate measurements are not eclipsed by the asteroid. 
       i.e. visibility remains even when the asteroid is between the observer 
       and the s/c. 

    """

    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]

    X_sc_ECI = traj_data["X"] # SC traj w.r.t. asteroid

    X_obs_ECI = np.full((len(t),6),np.array([[0,0,0,0,0,0]]))
    elevation_mask = np.deg2rad(10) # doesn't matter because observer at origin
    rho, rho_dot, el = get_rho_rhod_el(t, X_sc_ECI, X_obs_ECI, elevation_mask)

    sigma_rho = 1E-3 # 1 m
    sigma_rhod = 1E-6 # 1 mm/s

    noisy_rho = rho + np.random.normal(0, sigma_rho, size=np.shape(rho))
    noisy_rho_dot = rho_dot + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot))

    true_measurements = {
        'time' : t,
        'pos' : X_sc_ECI[:,0:3],
        'rho_1' : rho,
        'rho_dot_1' : rho_dot,
        'X_obs_1_ECI' : X_obs_ECI,
    }

    measurements = {
        'time' : t,
        'pos' : X_sc_ECI[:,0:3] + np.random.normal(0, sigma_rhod, size=np.shape(X_sc_ECI[:,0:3])),
        'rho_1' : noisy_rho,
        'rho_dot_1' : noisy_rho_dot,
        'X_obs_1_ECI' : X_obs_ECI,
    }

    meas_file = os.path.basename(traj_file).split('.')[0] + "_meas"
    with open(f'Data/Measurements/{meas_file}_noiseless.data', 'wb') as f:
        pickle.dump(true_measurements, f)

    with open(f'Data/Measurements/{meas_file}_noisy.data', 'wb') as f:
        pickle.dump(measurements, f)

if __name__ == '__main__':
    generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_equitorial.data')
    # generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined.data')
    # generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_short_timestep.data')
    # generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_high_alt_short_timestep.data')
    # generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_high_alt_shorter_timestep.data')
    # generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_high_alt_30_timestep.data')
    generate_measurements_asteroid('Data/Trajectories/trajectory_asteroid_inclined_high_alt_30_timestep.data')
