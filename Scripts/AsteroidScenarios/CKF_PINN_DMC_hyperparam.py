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
import itertools
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data

def main():

    # # full run
    # hparams = {
    #     'q_value' : [1E-11, 1E-10, 1E-9],
    #     "tau" : [50, 500, 5000],
    #     'epochs' : [10, 100],
    #     'learning_rate' : [1E-6, 1E-7, 1E-8],
    #     'batch_size' : [128, 512],
    #     "train_fcn" : ['pinn_a','pinn_alc']
    # }
    # full run 2
    hparams = {
        'q_value' : [1E-11, 1E-10],
        "tau" : [5000, 7500],
        'epochs' : [100, 200],
        'learning_rate' : [1E-6, 1E-7],
        'batch_size' : [128],
        "train_fcn" : ['pinn_a','pinn_alc']
    }

    # # full run 2
    # hparams = {
    #     'q_value' : [1E-11],
    #     "tau" : [6250, 7500],
    #     'epochs' : [100, 200],
    #     'learning_rate' : [1E-6],
    #     'batch_size' : [128],
    #     "train_fcn" : ['pinn_a','pinn_alc']
    # }

    # short run 
    # hparams = {
    #     'q_value' : [1E-12, 1E-10],
    #     "tau" : [150, 500],
    #     'epochs' : [10, 100],
    #     'learning_rate' : [1E-6, 1E-8],
    #     'batch_size' : [128, 512],
    #     "train_fcn" : ['pinn_a']
    # }

    # Trial
    # hparams = {
    #     'q_value' : [1E-12],
    #     "tau" : [150, 500],
    #     'epochs' : [10, 5],
    #     'learning_rate' : [ 1E-8],
    #     'batch_size' : [512],
    #     "train_fcn" : ['pinn_a']
    # }
    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    session_num = 0
    for hparam_inst in permutations_dicts:
        print("--- Starting trial: %d" % session_num)
        print({key: value for key, value in hparam_inst.items()})
        session_num += 1
        finished = False
        while not finished:
            try:
                run(hparam_inst)
                finished = True
            except Exception as e:
                print(e)

def run(hparams):
    q = hparams['q_value']
    tau = hparams['tau']
    epochs = hparams['epochs'] 
    learning_rate = hparams['learning_rate'] 
    batch_size = hparams['batch_size'] 
    train_fcn = hparams['train_fcn']
    Q0 = np.eye(3) * q ** 2
    save_df = "trained_networks_pm_hparams_EKF_4.data"


    ep = ErosParams()
    X_SB_E = np.array([2.40000000e+04, 0.0, 0.0, 
                        0.0, 4.70033081e+00, 4.71606150e-01]) / 1E3 # SC to Asteroid in km
    cart_state = X_SB_E + ep.X_BE_E
                           
    model = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v2.data", learning_rate=learning_rate)
    model.set_PINN_training_fcn(train_fcn)
        # "/../Data/Dataframes/eros_BVP_PINN_III.data")
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60)

    # Decrease scenario length
    M_end = len(t) // 5
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    w0 = np.array([0,0,0])
    z0 = np.hstack((cart_state, w0))

    dz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    z0 = z0 + dz0

    P_diag = np.array([1E1, 1E1, 1E1, 1E-1, 1E-1, 1E-1, 1E-9, 1E-9, 1E-9])**2
    R_diag = np.array([1E-3, 1E-6])**2
    P0 = np.diag(P_diag) 
    R0 = np.diag(R_diag)
    t0 = 0.0

    # Initialize Process Noise
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
    filter = ExtendedKalmanFilter(t0, z0, dz0, P0, f_dict, h_dict, logger=logger)
    filter.f_integrate = dynamics_ivp_no_jit # can't pass the model into the numba JIT function
    filter.atol = 1E-7
    filter.rtol = 1E-7

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
        model.train(X_train, Y_train, epochs=epochs, batch_size=batch_size)



    planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    hparams.update({'results' : planes_exp.percent_error_acc})

    # save the updated network in a custom network directory
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    model.config.update({'hparams' : [hparams]})
    model.save(save_df, data_dir)
    filter.logger.save(f"DMC_{q}_{tau}")

    print("Time Elapsed: " + str(time.time() - start_time))

    # add planes experiment and record metrics into dataframe for parallel coordinate plot (plotly)


if __name__ == "__main__":
    main()
