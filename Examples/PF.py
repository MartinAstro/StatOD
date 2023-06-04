"""
Particle Filter Example
========================

"""

import time

import matplotlib.pyplot as plt
import numpy as np

from StatOD.constants import *
from StatOD.data import get_example8_measurements
from StatOD.dynamics import *
from StatOD.filters import *
from StatOD.measurements import *
from StatOD.visualization.visualizations import *

np.random.seed(1234)


def main():
    ######################
    ## Get Measurements ##
    ######################
    numba = True
    t, Y, x_truth, y_truth = get_example8_measurements(case=1)

    t0 = t[0]
    M_end = len(t) // 5

    t = t[:M_end]
    Y = Y[:M_end]

    ######################
    ## Set Parameters   ##
    ######################
    k = 1
    eta = 1000
    L = 5000  # particles

    ##############################
    ## Set State and Covariance ##
    ##############################

    x0 = np.array([0.0, 1.0])
    dx0 = np.array([0.0, 0.0])
    x0 += dx0

    P_diag = (
        np.array(
            [0.2, 0.2],
        )
        ** 2
    )
    R_diag = np.array([0.1]) ** 2

    P_0 = np.diag(P_diag)
    R0 = np.diag(R_diag)

    R_vec = np.full((len(t), R0.shape[0], R0.shape[1]), R0)

    ########################
    ## Configure Dynamics ##
    ########################

    c_args = np.array([k, eta])
    f, dfdx = dynamics(x0, f_spring_duffing, c_args, use_numba=numba)

    h_args = np.array([])
    h, dhdx = measurements(x0, spring_observation_1, h_args)

    Q_args = []
    Q0 = np.eye(2) * 1e-1**2
    Q_fcn = process_noise(x0, Q0, get_Q, Q_args, use_numba=False)

    ######################
    ## Configure Filter ##
    ######################

    # Initialize
    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "f_args": c_args,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
    }

    h_dict = {
        "h": h,
        "dhdx": dhdx,
        "h_args": h_args,
    }

    start_time = time.time()
    logger = FilterLogger(len(x0), len(t))
    filter = ExtendedKalmanFilter(t0, x0, dx0, P_0, f_dict, h_dict, logger=logger)
    filter.run(t, Y, R_vec, np.full(len(t), None), np.empty((len(t), 0)))

    print("Time Elapsed: " + str(time.time() - start_time))

    ###############################
    ## Configure Particle Filter ##
    ###############################
    x_0_k = np.random.uniform(-2, 2, size=(L, 2))

    start_time = time.time()
    logger = FilterLogger(len(x0), len(t))
    filter = ParticleFilter(t0, x_0_k, f_dict, h_dict, logger=logger)
    filter.run(t, Y, R_vec, np.full(len(t), None), np.empty((len(t), 0)))

    print("Time Elapsed: " + str(time.time() - start_time))

    ##############
    ## Plotting ##
    ##############

    plt.figure()
    plt.scatter(filter.x_i_m1[:, 0], filter.x_i_m1[:, 1], s=2)
    plt.ylabel(r"$\dot{x}$")
    plt.xlabel(r"$x$")
    plt.show()


if __name__ == "__main__":
    main()
