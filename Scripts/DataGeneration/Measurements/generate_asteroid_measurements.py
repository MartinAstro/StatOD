import numpy as np
from StatOD.constants import ErosParams
from StatOD.measurements import get_rho_rhod_el
import pickle

from StatOD.utils import ECEF_2_ECI, latlon2cart

def generate_measurements_asteroid():
    """
    Generate range and range rate for a very simple asteroid scenario.
    Produce both truth and noisy measurements. 

    Assumptions:
    1: The observer is at a fixed point in inertial space 
       X_obs = [0,0,0, 0,0,0]
    2: The range and range-rate measurements are not eclipsed by the asteroid. 
       i.e. visibility remains even when the asteroid is between the observer 
       and the s/c. 
    3: The asteroid Eros is at perihelion (1.334 AU) and the distance between the Earth 
        and the asteroid is 0.1334 AU along the X-axis
    """

    with open('Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]

    X_eros_ECI = ErosParams().X_BE_E # Ideally this will be more realistic and the position of Eros changes over time
    X_sc_ECI = traj_data["X"] + X_eros_ECI # SC traj in ECI frame

    X_obs_ECI = np.full((len(t),6),np.array([[0,0,0,0,0,0]]))
    elevation_mask = np.deg2rad(10) # doesn't matter because observer at origin
    rho, rho_dot, el = get_rho_rhod_el(t, X_sc_ECI, X_obs_ECI, elevation_mask)

    sigma_rho = 1E-3 # 1 m
    sigma_rhod = 1E-6 # 1 mm/s

    noisy_rho = rho + np.random.normal(0, sigma_rho, size=np.shape(rho))
    noisy_rho_dot = rho_dot + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot))

    true_measurements = {
        'time' : t,
        'rho_1' : rho,
        'rho_dot_1' : rho_dot,
        'X_obs_1_ECI' : X_obs_ECI,
    }

    measurements = {
        'time' : t,
        'rho_1' : noisy_rho,
        'rho_dot_1' : noisy_rho_dot,
        'X_obs_1_ECI' : X_obs_ECI,
    }

    with open('Data/Measurements/range_rangerate_asteroid_wo_noise.data', 'wb') as f:
        pickle.dump(true_measurements, f)

    with open('Data/Measurements/range_rangerate_asteroid_w_noise.data', 'wb') as f:
        pickle.dump(measurements, f)

if __name__ == '__main__':
    generate_measurements_asteroid()
