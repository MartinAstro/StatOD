"""
Kalman Filter with Dynamic Model Compensation Example
============================================================

"""
import itertools
import multiprocessing as mp
import os 
import StatOD
from StatOD.data import get_measurements_pos
from StatOD.measurements import h_pos

def format_args(hparams):
    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args = []
    session_num = 0
    for hparam_inst in permutations_dicts:
        print("--- Starting trial: %d" % session_num)
        print({key: value for key, value in hparam_inst.items()})
        session_num += 1
        args.append((hparam_inst,))
    return args

def save_results(df_file, configs):
    import pandas as pd
    for config in configs:
        config = dict(sorted(config.items(), key = lambda kv: kv[0]))
        config['PINN_constraint_fcn'] = [config['PINN_constraint_fcn'][0]]# Can't have multiple args in each list
        df = pd.DataFrame().from_dict(config).set_index('timetag')

        try: 
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except: 
            df.to_pickle(df_file)
    
def main():

    hparams = {
        'q_value' : [1E-9, 1E-8, 1E-7],
        'epochs' : [10, 50, 100],
        'learning_rate' : [1E-6, 1E-5],
        'batch_size' : [512, 1024, 2048],
        "train_fcn" : ['pinn_a'],
        "tau" : [0]
    }
    hparams = {
        'q_value' : [1E-9, 1E-8, 1E-7],
        'epochs' : [10, 50, 100],
        'learning_rate' : [1E-6],
        'batch_size' : [1024, 2048],
        "train_fcn" : ['pinn_alc'],
        "tau" : [0]
    }

    # save_df = os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/df_DMC_Forced_Q.data"
    save_df = os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/df_fixed_IAC.data"

    threads = 3
    args = format_args(hparams)
    # run_catch(*args[0])
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run_catch, args)
        configs = results.get()
    save_results(save_df, configs)


def run_catch(args):
    finished = False
    # config = run(args)

    while not finished:
        try:
            config = run(args)
            finished = True
        except Exception as e:
            print(e)
    return config 



def run(hparams):


    import os
    import time
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import StatOD
    from StatOD.data import get_measurements
    from StatOD.dynamics import f_PINN_DMC, dfdx_PINN_DMC, dynamics_ivp_no_jit, get_Q_DMC, process_noise
    from StatOD.dynamics import get_Q_DMC_zero_order, f_PINN_DMC_zero_order, dfdx_PINN_DMC_zero_order, get_DMC_zero_order
    from StatOD.filters import FilterLogger, KalmanFilter, ExtendedKalmanFilter
    from StatOD.measurements import h_rho_rhod, measurements
    from StatOD.utils import pinnGravityModel
    from StatOD.visualizations import VisualizationBase
    from StatOD.constants import ErosParams
    import GravNN
    from GravNN.Analysis.PlanesExperiment import PlanesExperiment
    from GravNN.GravityModels.Polyhedral import get_poly_data
    from helper_functions import non_dimensionalize, dimensionalize, get_trajectory_data


    q = hparams['q_value']
    tau = hparams['tau']
    epochs = hparams['epochs'] 
    lr = hparams['learning_rate'] 
    batch_size = hparams['batch_size'] 
    pinn_constraint_fcn = hparams['train_fcn']
    Q0 = np.eye(3) * q ** 2

    
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
    epochs = 100
    # lr = 5E-6
    lr = 1E-5
                           
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
    M_end = len(t) // 1
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

    planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    hparams.update({'results' : planes_exp.percent_error_acc})


    # save the updated network in a custom network directory
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    model.config.update({'hparams' : [hparams]})
    model.save(None, data_dir) # save the network, but not into the directory right now
    # filter.logger.save(f"DMC_{q}_{tau}")

    print(np.nanmean(planes_exp.percent_error_acc))
    print("Time Elapsed: " + str(time.time() - start_time))
    print(f"Percent Error {np.nanmean(planes_exp.percent_error_acc)}")

    # add planes experiment and record metrics into dataframe for parallel coordinate plot (plotly)
    return model.config


if __name__ == "__main__":
    main()
