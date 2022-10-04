"""
Kalman Filter with Dynamic Model Compensation Example
============================================================

"""
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Scripts.Plots.plot_field_error_planes import generate_plot
import StatOD
from StatOD.data import get_measurements
from StatOD.dynamics import *
from StatOD.filters import FilterLogger, KalmanFilter, ExtendedKalmanFilter, Smoother, SquareRootInformationFilter, UnscentedKalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.utils import pinnGravityModel
from StatOD.visualizations import *
from StatOD.constants import *
import GravNN
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.GravityModels.PointMass import get_pm_data
def main():

    dim_constants = {
        "t_star" : 1E0,
        "m_star" : 1E0,
        "l_star" : 1E0
    }
    dim_constants = {
        "t_star" : 1E4,
        "m_star" : 1E0,
        "l_star" : 1E1
    }

    # q = 5E-8
    q = 1E-8
    # q = 5E-9
    Q_dt = 30 # seconds

    tau = 0    
    q_fcn = get_Q_DMC_zero_order
    f_fcn = f_PINN_DMC_zero_order
    dfdx = dfdx_PINN_DMC_zero_order

    # tau = 200
    # q_fcn = get_Q_DMC
    # f_fcn = f_PINN_DMC
    # dfdx = dfdx_PINN_DMC

    def Q_forced(z0, Q0, q_fcn, Q_args, use_numba=False):
        Q = np.zeros((9,9))
        Q[0:3,0:3] = np.eye(3)*1E-3**2 / 1E6
        Q[3:6,3:6] = np.eye(3)*1E-5**2/ 1E6
        Q[6:9,6:9] = np.eye(3)*q**2/ 1E6
        return Q

    # Don't use until DMC is tracked properly 
    batch_size = 256 * 4 
    epochs = 00
    lr = 1E-5
    pinn_constraint_fcn = "pinn_a"

    # data_file = "trajectory_asteroid_inclined_short_timestep"
    data_file = "trajectory_asteroid_equitorial"
    # data_file = "trajectory_asteroid_inclined_high_alt_30_timestep"
    # data_file = "trajectory_asteroid_inclined_high_alt_short_timestep"
    # data_file = "trajectory_asteroid_inclined_high_alt_shorter_timestep"
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + f'Data/Trajectories/{data_file}.data', 'rb') as f:
        traj_data = pickle.load(f)

    X_SB_E = traj_data['X'][0].copy() # in km
                           
    dim_constants_pinn = {
        "t_star" : dim_constants['t_star'],
        "m_star" : dim_constants['m_star'],
        "l_star" : dim_constants['l_star']*1E3
    }
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v2.data", 
        # "/../Data/Dataframes/eros_point_mass_v3.data", 
        # "/../Data/Dataframes/eros_BVP_PINN_III.data", 
        learning_rate=lr,
        dim_constants=dim_constants_pinn)
    model.set_PINN_training_fcn(pinn_constraint_fcn)

    # t, Y, X_stations_ECI = get_measurements(f"Data/Measurements/{data_file}_meas_noisy.data", t_gap=30)
    R_diag = np.array([1E-3, 1E-6])**2 / 1E10
    
    t, Y, X_stations_ECI = get_measurements(f"Data/Measurements/{data_file}_meas_noiseless.data", t_gap=30)
    # R_diag = np.array([0.0,0.0])**2 

    Y[0,1:] = np.nan
    # Y[:,1:] = np.nan # no measurements
    cart_state = X_SB_E 
    eros_pos = np.zeros((6,))

    # Decrease scenario length
    M_end = len(t) // 10
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = np.array([0,0,0])  
    w0 = traj_data['W'][0,:]
    # w0 = traj_data['W_pinn'][0,:]
    z0 = np.hstack((cart_state, w0))

    dz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    z0 = z0 + dz0

    # P_diag = np.array([1E0, 1E0, 1E0, 1E-3, 1E-3, 1E-3, 1E-6, 1E-6, 1E-6])**2
    # P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-5, 1E-5, 1E-5, 1E-6, 1E-6, 1E-6])**2
    # P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6])**2
    # P_diag = np.array([1E-2, 1E-2, 1E-2, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7])**2
    # P_diag = np.array([1E-14, 1E-14, 1E-14, 1E-14, 1E-14, 1E-14, 1E-14, 1E-14, 1E-14])**2
    P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-7, 1E-7, 1E-7])**2
    # P_diag = np.array([1E-4, 1E-4, 1E-4, 1E-5, 1E-5, 1E-5, 1E-7, 1E-7, 1E-7])**2
    # P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-5, 1E-5, 1E-5])**2
    # P_diag = np.array([1E1, 1E1, 1E1, 1E-2, 1E-2, 1E-2, 1E-6, 1E-6, 1E-6])**2


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
    Q_dt /= t_star

    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise

    Q0 = np.eye(3) * q ** 2
    # Q_args = [tau,]
    Q_args = [tau, model]
    # Q_fcn = get_Q_DMC_zero_order_model

    # Q_fcn = process_noise(z0, Q0, q_fcn, Q_args, use_numba=False)
    Q_fcn = Q_forced
    # Q_fcn = None
    # Q_fcn = process_noise(z0, Q0, get_Gamma_SRIF_DMC, Q_args, use_numba=False)


    # Initialize Dynamics and Measurements
    f_args = np.hstack((model, eros_pos, tau))
    f_dict = {
        "f": f_fcn,
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

    start_time = time.time()
    logger = FilterLogger(len(z0), len(t))
    # filter = ExtendedKalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    # filter = UnscentedKalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    # filter = SquareRootInformationFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function
    filter.atol = 1E-11
    filter.rtol = 1E-11

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
        # filter.Q_args = [tau, ]
        filter.Q_args = [tau, model]

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

    # smooth = Smoother(logger)
    # smooth.update()
    # logger = smooth.logger

    ##################################
    # Gather measurement predictions #
    ##################################
    
    x_truth = traj_data['X'][:M_end] # in km and km/s
    w_truth =  np.full((len(x_truth),3), 0) # DMC should be zero ideally 
    w_truth = traj_data['W'][:M_end] # accelerations above pm 
    # w_truth = traj_data['W_pinn'][:M_end] # accelerations above pm 

    x_truth += eros_pos
    x_truth = np.hstack((x_truth, w_truth)) 
    y_hat_vec = np.zeros((len(t), 2))
    for i in range(len(t)):
        y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i], np.zeros_like(logger.x_hat_i_plus[i]), h_args_vec[i])

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
    vis.plot_state_errors(x_truth)
    vis.plot_residuals(Y[:,1:], y_hat_vec, R_vec, y_labels)
    vis.plot_vlines(train_idx_list)

    # Plot the DMC values 
    plt.figure()
    plt.subplot(311)
    plt.plot(logger.t_i, logger.x_hat_i_plus[:,6], marker='o', markersize=2)
    plt.plot(logger.t_i, w_truth[:,0], marker='o', markersize=2)
    plt.subplot(312)
    plt.plot(logger.t_i, logger.x_hat_i_plus[:,7], marker='o', markersize=2)
    plt.plot(logger.t_i, w_truth[:,1], marker='o', markersize=2)
    plt.subplot(313)
    plt.plot(logger.t_i, logger.x_hat_i_plus[:,8], marker='o', markersize=2)
    plt.plot(logger.t_i, w_truth[:,2], marker='o', markersize=2)

    DMC_mag = np.linalg.norm(logger.x_hat_i_plus[:,6:9], axis=1)
    plt.figure()
    plt.plot(logger.t_i, DMC_mag)
    print(f"Average DMC Mag {np.mean(DMC_mag)}")

    # if epochs > 0:
        
    planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
    # planes_exp.config['gravity_data_fcn'] = [get_pm_data]
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    print(f"Error Percent Average: {np.nanmean(planes_exp.percent_error_acc)}")
    
    visPlanes = PlanesVisualizer(planes_exp)
    X_traj = logger.x_hat_i_plus[:,0:3]*1E3 / Eros().radius
    x = visPlanes.experiment.x_test
    y = visPlanes.experiment.percent_error_acc
    plt.figure()
    visPlanes.max = 20
    visPlanes.plot_plane(x,y, plane='xy')
    plt.plot(X_traj[:,0], X_traj[:,1], color='black', linewidth=0.5)
    fig2 = plt.figure()
    visPlanes.plot_plane(x,y, plane='xz')
    plt.plot(X_traj[:,0], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    fig3 = plt.figure()
    visPlanes.plot_plane(x,y, plane='yz')
    plt.plot(X_traj[:,1], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    from Scripts.Plots.plot_asteroid_trajectory import plot_cartesian_state_3d
    plot_cartesian_state_3d(traj_data['X'][:M_end]*1E3, Eros().obj_8k)
    plot_cartesian_state_3d(logger.x_hat_i_plus[:,0:3]*1E3, Eros().obj_8k, new_fig = False, cmap=plt.cm.autumn)

    plt.show()

if __name__ == "__main__":
    finished = False
    main()
    # while not finished:
    #     try:
    #         main()
    #         finished=True
    #     except:
    #         finished = False
    #         print("Didn't finish")
