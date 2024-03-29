"""
Kalman Filter with Stochastic Noise Compensation and Spherical Harmonic Coefficients
=====================================================================================

"""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import StatOD
from StatOD.constants import *
from StatOD.data import get_measurements
from StatOD.dynamics import *
from StatOD.filters import FilterLogger, KalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.visualization.visualizations import *


def main():
    ep = ErosParams()
    cart_state = (
        np.array([2.40000000e04, 0.0, 0.0, 0.0, 4.70033081e00, 4.71606150e-01])
        + ep.X_BE_E
    )

    t, Y, X_stations_ECI = get_measurements(
        "Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60
    )

    # Decrease scenario length
    M_end = len(t) // 5
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    dx0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 = cart_state + dx0

    P_diag = np.array([1e3, 1e3, 1e3, 10, 10, 10]) ** 2
    R_diag = np.array([1e-3, 1e-6]) ** 2
    P0 = np.diag(P_diag)
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    Q0 = np.eye(3) * 1e-6**2
    Q_args = []
    Q_fcn = process_noise(x0, Q0, get_Q, Q_args, use_numba=False)

    # Initialize Dynamics and Measurements
    f_args = np.hstack((ep.mu, ep.X_BE_E[0:3]))
    f, dfdx = dynamics(x0, f_point_mass, f_args)
    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "f_args": f_args,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
    }

    h_args = X_stations_ECI[0]
    h, dhdx = measurements(x0, h_rho_rhod, h_args)
    h_dict = {"h": h, "dhdx": dhdx, "h_args": h_args}

    #########################
    # Generate f/h_args_vec #
    #########################

    f_args_vec = np.full((len(t), len(f_args)), f_args)
    h_args_vec = X_stations_ECI
    R_vec = np.repeat(np.array([R0]), len(t), axis=0)

    ##############
    # Run Filter #
    ##############

    start_time = time.time()
    logger = FilterLogger(len(x0), len(t))
    filter = KalmanFilter(t0, x0, dx0, P0, f_dict, h_dict, logger=logger)
    filter.run(t, Y[:, 1:], R_vec, f_args_vec, h_args_vec)
    print("Time Elapsed: " + str(time.time() - start_time))

    ##################################
    # Gather measurement predictions #
    ##################################

    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + "Data/Trajectories/trajectory_asteroid.data", "rb") as f:
        traj_data = pickle.load(f)

    x_truth = traj_data["X"][:M_end] + ep.X_BE_E
    y_hat_vec = np.zeros((len(t), 2))
    for i in range(len(t)):
        y_hat_vec[i] = filter.predict_measurement(
            logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i]
        )

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r"$\rho$", r"$\dot{\rho}$"])
    vis = VisualizationBase(logger, directory, False)
    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:, 1:], y_hat_vec, R_vec, y_labels)
    plt.show()


if __name__ == "__main__":
    main()
