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
from StatOD.filters import FilterLogger, KalmanFilter, ExtendedKalmanFilter, Smoother, UnscentedKalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.utils import pinnGravityModel
from StatOD.visualizations import *
from StatOD.constants import *
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data
def main():

    dim_constants = {
        "t_star" : 1E4,
        "m_star" : 1E0,
        "l_star" : 1E1
    }

    q_fcn = get_Q_DMC_zero_order_model
    f_fcn = f_PINN_DMC_zero_order
    dfdx_fcn = dfdx_PINN_DMC_zero_order

    data_file = "trajectory_asteroid_inclined_high_alt_30_timestep"

    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + f'Data/Trajectories/{data_file}.data', 'rb') as f:
        traj_data = pickle.load(f)

    t, Y, X_stations_ECI = get_measurements(f"Data/Measurements/{data_file}_meas_noiseless.data", t_gap=30)
    R_diag = np.array([0.0,0.0])**2 

    Y[0,1:] = np.nan
    q = 1e-14
    scale = 1

    def Q_forced(z0, Q0, q_fcn, Q_args, use_numba=False):
        Q = np.zeros((9,9))
        Q[0:3,0:3] = np.eye(3)*1E-6**2  / scale
        Q[3:6,3:6] = np.eye(3)*1E-8**2  / scale
        Q[6:9,6:9] = np.eye(3)*q**2 /scale
        return Q

    batch_size = 256 * 4
    tau = 90
    Q_dt = 30 # seconds

    epochs = 00
    lr = 1E-6
    pinn_constraint_fcn = "pinn_a"

    X_SB_E = traj_data['X'][0].copy() # SC to Asteroid in km
                           
    dim_constants_pinn = {
        "t_star" : dim_constants['t_star'],
        "m_star" : dim_constants['m_star'],
        "l_star" : dim_constants['l_star']*1E3
    }

    model = pinnGravityModel(os.path.dirname(StatOD.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v4.data", # Better dimensionalization
        learning_rate=lr,
        dim_constants=dim_constants_pinn)
    model.set_PINN_training_fcn(pinn_constraint_fcn)
    

    cart_state = X_SB_E 
    eros_pos = np.zeros((6,))

    # Decrease scenario length
    M_end = len(t) // 6
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = traj_data['W_pinn'][0,:]
    z0 = np.hstack((cart_state, w0))

    dz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    z0 = z0 + dz0

    P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-7, 1E-7, 1E-7])**2
    P_diag = np.array([1E-5, 1E-5, 1E-5, 1E-6, 1E-6, 1E-6, 1E-9, 1E-9, 1E-9])**2

    t_star = dim_constants['t_star']
    l_star = dim_constants['l_star']
    ms = dim_constants['l_star'] / dim_constants['t_star']
    ms2 = dim_constants['l_star'] / dim_constants['t_star']**2

    # non-dimensionalize states
    cart_state /= l_star
    z0[0:3] /= l_star
    z0[3:6] /= ms
    z0[6:9] /= ms2
    t /= t_star
    Y[:,1] /= l_star
    Y[:,2] /= ms
    P_diag[0:3] /= l_star**2
    P_diag[3:6] /= ms**2
    P_diag[6:9] /= ms2**2
    R_diag[0] /= l_star**2
    R_diag[1] /= ms**2
    tau /= t_star
    q /= ms2

    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
    Q0 = np.eye(3) * q ** 2
    Q_args = [tau, model]
    Q_fcn = Q_forced

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
        "Q_dt" : Q_dt
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

    alpha = 1E-1 # determines the spread of the sigma points
    beta = 2.0 # used to incorporate knowledge of the distribution 
    kappa = 0 # 1E-3 # secondary scaling parameter (normally zero)

    start_time = time.time()
    logger = FilterLogger(len(z0), len(t))
    filter = UnscentedKalmanFilter(t0, z0, dz0, P0, alpha, kappa, beta, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_unscented_no_jit # can't pass the model into the numba JIT function
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
    y_hat_vec = np.zeros((len(t), 2))
    for i in range(len(t)):
        y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i],  h_args_vec[i])

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$\rho$', r'$\dot{\rho}$'])


    cart_state *= l_star
    logger.x_hat_i_plus[:,0:3] *= l_star
    logger.x_hat_i_plus[:,3:6] *= l_star / t_star
    logger.x_hat_i_plus[:,6:9] *= ms2
    t *= t_star
    Y[:,1] *= l_star
    Y[:,2] *= ms
    y_hat_vec[:,0] *= l_star
    y_hat_vec[:,1] *= ms
    
    logger.t_i *= t_star
    logger.P_i_plus[:,0:3,0:3] *= l_star**2
    logger.P_i_plus[:,3:6,3:6] *= ms**2
    logger.P_i_plus[:,6:9,6:9] *= ms2**2
    R_vec[:,0,0] *= l_star**2
    R_vec[:,1,1] *= ms**2



    vis = VisualizationBase(logger, directory, False)
    plt.rc('text', usetex=False)

    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:,1:], y_hat_vec, R_vec, y_labels)
    vis.plot_vlines(train_idx_list)

    # Plot the DMC values 
    def plot_DMC(x, y1, y2):
        plt.plot(x, y1)
        plt.plot(x, y2)
        criteria1 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 > 0)))).T, axis=1)
        criteria2 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 < 0)))).T, axis=1)
        criteria3 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 < 0)))).T, axis=1)
        criteria4 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 > 0)))).T, axis=1)
        percent_productive = np.round((np.count_nonzero(criteria1) + np.count_nonzero(criteria2)) / len(x) * 100,2)
        plt.gca().annotate(f"Percent Useful: {percent_productive}",xy=(0.75, 0.75), xycoords='axes fraction', size=8)
        plt.gca().fill_between(x, y1, y2, where=criteria1, color='green', alpha=0.3,
                    interpolate=True)
        plt.gca().fill_between(x, y1, y2, where=criteria2, color='green', alpha=0.3,
                    interpolate=True)
        plt.gca().fill_between(x, y1, y2, where=criteria3, color='red', alpha=0.3,
                    interpolate=True)
        plt.gca().fill_between(x, y1, y2, where=criteria4, color='red', alpha=0.3,
                    interpolate=True)

    plt.figure()
    plt.subplot(311)
    plot_DMC(logger.t_i, logger.x_hat_i_plus[:,6], w_truth[:,0])
    plt.subplot(312)
    plot_DMC(logger.t_i, logger.x_hat_i_plus[:,7], w_truth[:,1])
    plt.subplot(313)
    plot_DMC(logger.t_i, logger.x_hat_i_plus[:,8], w_truth[:,2])


    DMC_mag = np.linalg.norm(logger.x_hat_i_plus[:,6:9], axis=1)
    plt.figure()
    plt.plot(DMC_mag)
    print(f"Average DMC Mag {np.mean(DMC_mag)}")


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
