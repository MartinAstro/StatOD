"""
Kalman Filter with Dynamic Model Compensation Example
============================================================

"""
import os
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
    X_SB_E = np.array([2.40000000e+04, 0.0, 0.0, 
                        0.0, 4.70033081e+00, 4.71606150e-01]) / 1E3 # SC to Asteroid in km
    cart_state = X_SB_E + ep.X_BE_E
                           
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v2.data")
    model.set_PINN_training_fcn("pinn_alc")
        # "/../Data/Dataframes/eros_BVP_PINN_III.data")
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60)

    # Decrease scenario length
    M_end = len(t) // 20
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = np.array([0,0,0])
    z0 = np.hstack((cart_state, w0))

    dz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    z0 = z0 + dz0

    P_diag = np.array([1E1, 1E1, 1E1, 1E-1, 1E-1, 1E-1, 1E-7, 1E-7, 1E-7])**2
    R_diag = np.array([1E-3, 1E-6])**2
    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    # tau = 140.0 # Larger values mean longer time correlation
    tau = 100.0 # Larger values mean longer time correlation
    Q0 = np.eye(3) * 5e-8 ** 2
    # tau = 100.0 # Larger values mean longer time correlation
    # Q0 = np.eye(3) * 1e-8 ** 2
    # Q0 = np.eye(3) * 1e-10 ** 2
    Q_args = [tau,]
    Q_fcn = process_noise(z0, Q0, get_Q_DMC, Q_args, use_numba=False)

    # Initialize Dynamics and Measurements
    f_args = np.hstack((model, ep.X_BE_E, tau))
    f = f_PINN_DMC
    dfdx = dfdx_PINN_DMC

    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "f_args": f_args,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
    }

    h_args = X_stations_ECI[0]
    h, dhdx = measurements(z0, h_rho_rhod, h_args)
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
    logger = FilterLogger(len(z0), len(t))
    filter = KalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function
    filter.atol = 1E-7
    filter.rtol = 1E-7

    batch_size = 32
    total_batches = len(Y) // batch_size
    for k in range(total_batches+1):

        # Gather measurements in batch
        start_idx = k*batch_size
        end_idx = None if (k+1)*batch_size >= len(Y) else (k+1)*batch_size
        t_batch = t[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx,1:]
        R_batch = R_vec[start_idx:end_idx]
        f_args_batch = f_args_vec[start_idx:end_idx]
        h_args_batch = h_args_vec[start_idx:end_idx]

        # usee latest trained model to f_args
        f_args_batch[:,0] = model

        # run the filter on the batch of measurements
        filter.run(t_batch, Y_batch, R_batch, f_args_batch, h_args_batch)

        # collect network training data
        X_train = filter.logger.x_hat_i_plus[start_idx:end_idx,0:3] - ep.X_BE_E[0:3]# position estimates
        Y_train = filter.logger.x_hat_i_plus[start_idx:end_idx,6:9] # high-order accel est
        model.train(X_train, Y_train, epochs=0, batch_size=batch_size)

    # save the updated network in a custom network directory
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    model.save("trained_networks_pm.data", data_dir)

    # (optional): save the log
    q = np.sqrt(np.diag(Q0)[0])
    filter.logger.save(f"DMC_{q}_{tau}")

    print("Time Elapsed: " + str(time.time() - start_time))

    ##################################
    # Gather measurement predictions #
    ##################################
    
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + 'Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
        traj_data = pickle.load(f)

    x_truth = traj_data['X'][:M_end] + ep.X_BE_E
    w_truth =  np.full((len(x_truth),3), 0) # DMC should be zero ideally 
    x_truth = np.hstack((x_truth, w_truth)) 
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
