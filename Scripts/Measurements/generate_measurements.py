import numpy as np
from StatOD.constants import EarthParams
from StatOD.measurements import get_rho_rhod_el
import pickle

from StatOD.utils import ECEF_2_ECI, latlon2cart

def generate_measurements_J2():
    ep = EarthParams()

    elevation_mask = np.deg2rad(10)
    theta_0 = np.deg2rad(122)

    lat_1, lon_1 = np.pi - (np.deg2rad(-35.398333) + np.pi/2), np.deg2rad(148.981944)
    lat_2, lon_2 = np.pi - (np.deg2rad( 40.427222) + np.pi/2), np.deg2rad(355.749444)
    lat_3, lon_3 = np.pi - (np.deg2rad( 35.247164) + np.pi/2), np.deg2rad(243.205)

    with open('Data/Trajectories/trajectory_J2.data', 'rb') as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]
    X = traj_data["X"]

    X_obs_1_ECEF = latlon2cart(ep.R, lat_1, lon_1)  
    X_obs_2_ECEF = latlon2cart(ep.R, lat_2, lon_2)  
    X_obs_3_ECEF = latlon2cart(ep.R, lat_3, lon_3)  

    X_obs_1_ECI = ECEF_2_ECI(t, X_obs_1_ECEF, ep.omega, theta_0)
    X_obs_2_ECI = ECEF_2_ECI(t, X_obs_2_ECEF, ep.omega, theta_0)
    X_obs_3_ECI = ECEF_2_ECI(t, X_obs_3_ECEF, ep.omega, theta_0)

    rho_1, rho_dot_1, el_1 = get_rho_rhod_el(t, X, X_obs_1_ECI, elevation_mask)
    rho_2, rho_dot_2, el_2 = get_rho_rhod_el(t, X, X_obs_2_ECI, elevation_mask)
    rho_3, rho_dot_3, el_3 = get_rho_rhod_el(t, X, X_obs_3_ECI, elevation_mask)

    # Add noise
    sigma_rho = 1E-3 # 1 m
    sigma_rhod = 1E-6 # 1 mm/s

    noisy_rho_1 = rho_1 + np.random.normal(0, sigma_rho, size=np.shape(rho_1))
    noisy_rho_2 = rho_2 + np.random.normal(0, sigma_rho, size=np.shape(rho_2))
    noisy_rho_3 = rho_3 + np.random.normal(0, sigma_rho, size=np.shape(rho_3))

    noisy_rho_dot_1 = rho_dot_1 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_1))
    noisy_rho_dot_2 = rho_dot_2 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_2))
    noisy_rho_dot_3 = rho_dot_3 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_3))

    true_measurements = {
        'time' : t,
        
        'rho_1' : rho_1,
        'rho_2' : rho_2,
        'rho_3' : rho_3,

        'rho_dot_1' : rho_dot_1,
        'rho_dot_2' : rho_dot_2,
        'rho_dot_3' : rho_dot_3,

        'X_obs_1_ECI' : X_obs_1_ECI,
        'X_obs_2_ECI' : X_obs_2_ECI,
        'X_obs_3_ECI' : X_obs_3_ECI,
    }

    measurements = {
        'time' : t,
        
        'rho_1' : noisy_rho_1,
        'rho_2' : noisy_rho_2,
        'rho_3' : noisy_rho_3,

        'rho_dot_1' : noisy_rho_dot_1,
        'rho_dot_2' : noisy_rho_dot_2,
        'rho_dot_3' : noisy_rho_dot_3,

        'X_obs_1_ECI' : X_obs_1_ECI,
        'X_obs_2_ECI' : X_obs_2_ECI,
        'X_obs_3_ECI' : X_obs_3_ECI,
    }

    with open('Data/Measurements/range_rangerate_w_J2_wo_noise.data', 'wb') as f:
        pickle.dump(true_measurements, f)

    with open('Data/Measurements/range_rangerate_w_J2_w_noise.data', 'wb') as f:
        pickle.dump(measurements, f)



def generate_measurements_J3():
    ep = EarthParams()
    elevation_mask = np.deg2rad(10)
    theta_0 = np.deg2rad(122)

    lat_1, lon_1 = np.pi - (np.deg2rad(-35.398333) + np.pi/2), np.deg2rad(148.981944)
    lat_2, lon_2 = np.pi - (np.deg2rad( 40.427222) + np.pi/2), np.deg2rad(355.749444)
    lat_3, lon_3 = np.pi - (np.deg2rad( 35.247164) + np.pi/2), np.deg2rad(243.205)

    with open('Data/Trajectories/trajectory_J3.data', 'rb') as f:
        traj_data = pickle.load(f)

    t = traj_data["t"]
    X = traj_data["X"]

    X_obs_1_ECEF = latlon2cart(ep.R, lat_1, lon_1)  
    X_obs_2_ECEF = latlon2cart(ep.R, lat_2, lon_2)  
    X_obs_3_ECEF = latlon2cart(ep.R, lat_3, lon_3)  

    X_obs_1_ECI = ECEF_2_ECI(t, X_obs_1_ECEF, ep.omega, theta_0)
    X_obs_2_ECI = ECEF_2_ECI(t, X_obs_2_ECEF, ep.omega, theta_0)
    X_obs_3_ECI = ECEF_2_ECI(t, X_obs_3_ECEF, ep.omega, theta_0)

    rho_1, rho_dot_1, el_1 = get_rho_rhod_el(t, X, X_obs_1_ECI, elevation_mask)
    rho_2, rho_dot_2, el_2 = get_rho_rhod_el(t, X, X_obs_2_ECI, elevation_mask)
    rho_3, rho_dot_3, el_3 = get_rho_rhod_el(t, X, X_obs_3_ECI, elevation_mask)

    # Add noise
    sigma_rho = 1E-3 # 1 m
    sigma_rhod = 1E-6 # 1 mm/s

    noisy_rho_1 = rho_1 + np.random.normal(0, sigma_rho, size=np.shape(rho_1))
    noisy_rho_2 = rho_2 + np.random.normal(0, sigma_rho, size=np.shape(rho_2))
    noisy_rho_3 = rho_3 + np.random.normal(0, sigma_rho, size=np.shape(rho_3))

    noisy_rho_dot_1 = rho_dot_1 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_1))
    noisy_rho_dot_2 = rho_dot_2 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_2))
    noisy_rho_dot_3 = rho_dot_3 + np.random.normal(0, sigma_rhod, size=np.shape(rho_dot_3))

    true_measurements = {
        'time' : t,
        
        'rho_1' : rho_1,
        'rho_2' : rho_2,
        'rho_3' : rho_3,

        'rho_dot_1' : rho_dot_1,
        'rho_dot_2' : rho_dot_2,
        'rho_dot_3' : rho_dot_3,

        'X_obs_1_ECI' : X_obs_1_ECI,
        'X_obs_2_ECI' : X_obs_2_ECI,
        'X_obs_3_ECI' : X_obs_3_ECI,
    }

    measurements = {
        'time' : t,
        
        'rho_1' : noisy_rho_1,
        'rho_2' : noisy_rho_2,
        'rho_3' : noisy_rho_3,

        'rho_dot_1' : noisy_rho_dot_1,
        'rho_dot_2' : noisy_rho_dot_2,
        'rho_dot_3' : noisy_rho_dot_3,

        'X_obs_1_ECI' : X_obs_1_ECI,
        'X_obs_2_ECI' : X_obs_2_ECI,
        'X_obs_3_ECI' : X_obs_3_ECI,
    }

    with open('Data/Measurements/range_rangerate_w_J3_wo_noise.data', 'wb') as f:
        pickle.dump(true_measurements, f)

    with open('Data/Measurements/range_rangerate_w_J3_w_noise.data', 'wb') as f:
        pickle.dump(measurements, f)


if __name__ == '__main__':
    generate_measurements_J2()
    generate_measurements_J3()
