"""
Kalman Filter with Smoother Example
=====================================

"""

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import StatOD
from StatOD.constants import EarthParams
from StatOD.data import get_measurements
from StatOD.dynamics import dynamics, f_J2
from StatOD.filters import FilterLogger, KalmanFilter, NonLinearBatchFilter, Smoother
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.visualization.visualizations import *


def main():
    ep = EarthParams()
    cart_state = np.array(
        [
            -3515.4903270335103,
            8390.716310243395,
            4127.627352553683,
            -4.357676322178153,
            -3.3565791387645487,
            3.111892927869902,
        ]
    )

    t, Y, X_stations_ECI = get_measurements(
        "Data/Measurements/range_rangerate_w_J2_w_noise.data"
    )

    # Decrease scenario length
    M_end = len(t) // 5
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    dx0 = np.array([0.1, 0.0, 0.0, 1e-4, 0.0, 0.0])
    x0 = cart_state + (dx0 / 10)

    P_diag = np.array([1, 1, 1, 1e-3, 1e-3, 1e-3]) ** 2
    R_diag = np.array([1e-3, 1e-6]) ** 2
    P0 = np.diag(P_diag)
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    Q0 = np.eye(3) * 1e-7**2
    Q_args = []
    # Q_fcn = process_noise(x0, Q0, get_Q, Q_args, use_numba=False)
    Q_fcn = None

    # Initialize Dynamics and Measurements
    f_args = np.array([ep.R, ep.mu, ep.J2])
    f, dfdx = dynamics(x0, f_J2, f_args)
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

    start_time = time.time()
    logger = FilterLogger(len(x0), len(t))
    filter = KalmanFilter(t0, x0, dx0, P0, f_dict, h_dict, logger=logger)
    filter.run(t, Y[:, 1:], R_vec, f_args_vec, h_args_vec)
    print("Time Elapsed: " + str(time.time() - start_time))

    ################
    # Run Smoother #
    ################

    # NOTE: The Smoother accounts for process noise

    # Run smoother on the CKF output
    smoother = Smoother(filter.logger)
    smoother.update()

    ############
    # Run NLBF #
    ############

    # NOTE: The NLBF cannot account for process noise

    logger = FilterLogger(len(x0), len(t))
    NLBFilter = NonLinearBatchFilter(
        t0, x0, dx0, P0, f_dict, h_dict, logger, iterations=1
    )
    start_time = time.time()
    NLBFilter.run(t, Y[:, 1:], R_vec, f_args_vec, h_args_vec, tol=1e-7)
    print("Time Elapsed: " + str(time.time() - start_time))

    ############
    # Plotting #
    ############
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + "Data/Trajectories/trajectory_J2.data", "rb") as f:
        traj_data = pickle.load(f)

    x_truth = traj_data["X"][:M_end]

    # CKF State Errors
    vis = VisualizationBase(filter.logger)
    vis.plot_state_errors(x_truth)

    # Smoother State Errors
    vis = VisualizationBase(smoother.logger)
    vis.plot_state_errors(x_truth)

    # NLB Filter State Errors
    vis = VisualizationBase(NLBFilter.logger)
    vis.plot_state_errors(x_truth)

    plt.show()


if __name__ == "__main__":
    main()
