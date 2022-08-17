"""
Kalman Filter with Stochastic Noise Compensation Example
============================================================

"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import StatOD
from StatOD.data import get_measurements
from StatOD.dynamics import *
from StatOD.filters import FilterLogger, KalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.utils import pinnGravityModel
from StatOD.visualizations import *
from StatOD.constants import *
import GravNN

def main():
    ep = ErosParams()
    cart_state = np.array([2.40000000e+04, 0.0, 0.0, 
                           0.0, 4.70033081e+00, 4.71606150e-01]) + ep.X_BE_E
                           
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_BVP_PINN_III.data")
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60)

    # Decrease scenario length
    M_end = len(t) // 5
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    dx0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    x0 = cart_state + dx0

    P_diag = np.array([1E3, 1E3, 1E3, 10, 10, 10])**2
    R_diag = np.array([1E-3, 1E-6])**2
    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    Q0 = np.eye(3) * 1e-6 ** 2
    Q_args = []
    Q_fcn = process_noise(x0, Q0, get_Q, Q_args, use_numba=False)

    # Initialize Dynamics and Measurements
    f_args = np.hstack((model, ep.X_BE_E))
    f = f_PINN
    dfdx = dfdx_PINN

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
    h_dict = {'h': h, 'dhdx': dhdx, 'h_args': h_args}

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
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function
    filter.run(t, Y[:,1:], R_vec, f_args_vec, h_args_vec)
    print("Time Elapsed: " + str(time.time() - start_time))

    ##################################
    # Gather measurement predictions #
    ##################################
    
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + 'Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
        traj_data = pickle.load(f)

    x_truth = traj_data['X'][:M_end] + ep.X_BE_E
    y_hat_vec = np.zeros((len(t), 2))
    for i in range(len(t)):
        y_hat_vec[i] = filter.predict_measurement(logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i])

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$\rho$', r'$\dot{\rho}$'])
    vis = VisualizationBase(logger, directory, False)
    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:,1:], y_hat_vec, R_vec, y_labels)
    plt.show()

if __name__ == "__main__":
    main()
