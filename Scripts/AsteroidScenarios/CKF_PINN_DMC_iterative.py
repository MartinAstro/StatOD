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
from StatOD.filters import FilterLogger, KalmanFilter, ExtendedKalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.utils import pinnGravityModel
from StatOD.visualizations import *
from StatOD.constants import *
import GravNN
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data
def main():
    batch_size = 32
    # batch_size = 256
    q = 1e-11 
    # q = 1e-11 
    # q = 5e-11 
    tau = 1E4 # Larger values mean smaller time correlation
    epochs = 100
    lr = 1E-6
    pinn_constraint_fcn = "pinn_a"
    # Among best params 
    # tau = 140.0 # Larger values mean smaller time correlation
    # Q0 = np.eye(3) * 5e-8 ** 2


    ep = ErosParams()
    X_SB_E = np.array([2.40000000e+04, 0.0, 0.0, 
                        0.0, 4.70033081e+00, 4.71606150e-01]) / 1E3 # SC to Asteroid in km
                           
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v2.data", learning_rate=lr)
    # model.set_PINN_training_fcn("pinn_alc")
    model.set_PINN_training_fcn(pinn_constraint_fcn)
        # "/../Data/Dataframes/eros_BVP_PINN_III.data")
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_simple_wo_noise.data", t_gap=60)
    cart_state = X_SB_E 
    eros_pos = np.zeros_like(ep.X_BE_E)

    # t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60)
    # cart_state = X_SB_E + ep.X_BE_E
    # eros_pos = ep.X_BE_E
    

    # Decrease scenario length
    M_end = len(t) // 20
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = np.array([0,0,0])
    z0 = np.hstack((cart_state, w0))

    dz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    z0 = z0 + dz0

    P_diag = np.array([1E2, 1E2, 1E2, 1E-1, 1E-1, 1E-1, 1E-9, 1E-9, 1E-9])**2
    R_diag = np.array([1E-3, 1E-6])**2

    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise

    Q0 = np.eye(3) * q ** 2
    Q_args = [tau,]
    Q_fcn = process_noise(z0, Q0, get_Q_DMC, Q_args, use_numba=False)

    # Initialize Dynamics and Measurements
    f_args = np.hstack((model, eros_pos, tau))
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
    filter = ExtendedKalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    # filter = KalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function
    filter.atol = 1E-7
    filter.rtol = 1E-7

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
        model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size)
        model.train_idx += 1
        train_idx_list.append(end_idx)

    train_idx_list.pop()
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
    
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + 'Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
        traj_data = pickle.load(f)

    x_truth = traj_data['X'][:M_end] + eros_pos
    w_truth =  np.full((len(x_truth),3), 0) # DMC should be zero ideally 
    x_truth = np.hstack((x_truth, w_truth)) 
    y_hat_vec = np.zeros((len(t), 2))
    for i in range(len(t)):
        # y_hat_vec[i] = filter.predict_measurement(logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i])
        y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i], np.zeros_like(logger.x_hat_i_plus[i]), h_args_vec[i])

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$\rho$', r'$\dot{\rho}$'])
    vis = VisualizationBase(logger, directory, False)
    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:,1:], y_hat_vec, R_vec, y_labels)

    for i in plt.get_fignums():
        plt.figure(i)
        for idx in train_idx_list:
            ylim = plt.gca().get_ylim()
            plt.vlines(logger.t_i[idx], ylim[0], ylim[1])

    planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    print(f"Error Percent Average: {np.nanmean(planes_exp.percent_error_acc)}")

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
