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
from StatOD.data import get_measurements, get_measurements_pos
from StatOD.dynamics import *
from StatOD.filters import FilterLogger, KalmanFilter, ExtendedKalmanFilter, Smoother, UnscentedKalmanFilter
from StatOD.measurements import h_rho_rhod, measurements, h_pos
from StatOD.utils import pinnGravityModel
from StatOD.visualizations import *
from StatOD.constants import *
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.CelestialBodies.Asteroids import Eros

from helper_functions import * 



def main():

    dim_constants = {
        "t_star" : 1E4,
        "m_star" : 1E0,
        "l_star" : 1E1
    }

    # f_fcn, dfdx_fcn, q_fcn, tau = get_DMC_first_order(tau=120)
    f_fcn, dfdx_fcn, q_fcn, tau = get_DMC_zero_order()

    h_fcn = h_pos
    R_diag = np.array([1E-3, 1E-3, 1E-3])**2 

    data_file = "trajectory_asteroid_inclined_high_alt_30_timestep"
    # data_file = "trajectory_asteroid_equitorial"
    traj_data = get_trajectory_data(data_file)

    t, Y, X_stations_ECI = get_measurements_pos(f"Data/Measurements/{data_file}_meas_noiseless.data", t_gap=30)
    Y[0,1:] = np.nan

    q = 5e-7
    batch_size = 256 * 4
    epochs = 00
    lr = 5E-6
                           
    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn['l_star'] *= 1E3

    model = pinnGravityModel(os.path.dirname(StatOD.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v4.data",
        learning_rate=lr,
        dim_constants=dim_constants_pinn)
    model.set_PINN_training_fcn("pinn_a")
    
    cart_state = traj_data['X'][0].copy() # SC to Asteroid in km 
    eros_pos = np.zeros((6,))

    # Decrease scenario length
    M_end = len(t) // 10
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = np.zeros((3,)) #traj_data['W_pinn'][0,:]
    z0 = np.hstack((cart_state, w0))

    P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-7, 1E-7, 1E-7])**2


    # non-dimensionalize states
    t, z0, Y, P_diag, R_diag, tau, q, dim_constants = non_dimensionalize(t, z0, Y, P_diag, R_diag, tau, q, dim_constants)

    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    Q0 = np.eye(3) * q ** 2
    Q_args = [tau,]
    Q_fcn = process_noise(z0, Q0, q_fcn, Q_args, use_numba=False)

    # Initialize Dynamics and Measurements
    f_args = np.hstack((model, eros_pos, tau))
    f = f_fcn
    dfdx = dfdx_fcn

    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "f_args": f_args,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
        "Q_dt" : 30
    }

    h_args = X_stations_ECI[0]
    h, dhdx = measurements(z0, h_fcn, h_args)
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
    dz0 = None
    filter = ExtendedKalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function

    filter.atol = 1E-9
    filter.rtol = 1E-9

    train_idx_list = []
    total_batches = len(Y) // batch_size
    model.train_idx = 0
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
        X_train = filter.logger.x_hat_i_plus[start_idx:end_idx,0:3] - eros_pos[0:3]# position estimates
        Y_train = filter.logger.x_hat_i_plus[start_idx:end_idx,6:9] # high-order accel est

        # Don't train on the last batch of data if it's too small
        if k != total_batches:
            model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size)
            model.train_idx += 1
            train_idx_list.append(end_idx)

    if len(train_idx_list) > 0 : train_idx_list.pop()

    # save the updated network in a custom network directory
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    # model.save("trained_networks_pm.data", data_dir)

    # (optional): save the log
    q = np.sqrt(np.diag(Q0)[0])
    # filter.logger.save(f"DMC_{q}_{tau}")

    print("Time Elapsed: " + str(time.time() - start_time))


    ##################################
    # Gather measurement predictions #
    ##################################

    x_truth = traj_data['X'][:M_end] # in km and km/s
    w_truth =  np.full((len(x_truth),3), 0) # DMC should be zero ideally 
    w_truth = traj_data['W'][:M_end] # accelerations above pm 

    x_truth += eros_pos
    x_truth = np.hstack((x_truth, w_truth)) 
    y_hat_vec = np.zeros((len(t), 3))
    for i in range(len(t)):
        y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i], np.zeros_like(logger.x_hat_i_plus[i]), h_args_vec[i])

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$x$', r'$y$', r"$z$"])


    ########################
    # Plotting and Metrics #
    ########################

    logger, t, Y, y_hat_vec, R_vec, dim_constants = dimensionalize(logger, t, Y, y_hat_vec, R_vec, dim_constants)

    vis = VisualizationBase(logger, directory, False)
    plt.rc('text', usetex=False)

    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:,1:], y_hat_vec, R_vec, y_labels)
    vis.plot_vlines(train_idx_list)

    # Plot the DMC values 
    plot_DMC(logger, w_truth)

    planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    print(f"Error Percent Average: {np.nanmean(planes_exp.percent_error_acc)}")

    plot_error_planes(planes_exp, max_error=20, logger=logger)
   
    plt.show()

if __name__ == "__main__":
    finished = False
    main()

    # finished = False
    # while not finished:
    #     try:
    #         main()
    #         finished=True
    #     except:
    #         finished = False
    #         print("Didn't finish")
