
import numpy as np
import StatOD
import multiprocessing as mp
from helper_functions import * 
def main():
    hparams = {
        'q_value' : [1E-9, 1E-8, 1E-7],
        'epochs' : [10, 50, 100],
        'learning_rate' : [1E-4, 1E-5, 1E-6],
        'batch_size' : [256, 1024, 2048],
        "train_fcn" : ['pinn_a','pinn_alc'],
        'boundary_condition_data' : [True, False]
    }
    # hparams = {
    #     'q_value' : [1E-9],
    #     'epochs' : [10],
    #     'learning_rate' : [1E-5],
    #     'batch_size' : [2048],
    #     "train_fcn" : ['pinn_a'],
    #     'boundary_condition_data' : [True, False]
    # }

    save_df = os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/hparams_rotating_setup_test.data"

    threads = 2
    args = format_args(hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run_catch, args)
        configs = results.get()
    save_results(save_df, configs)


def run_catch(args):
    finished = False
    while not finished:
        try:
            config = run(args)
            finished = True
        except Exception as e:
            print(e)
    return config 


def rotating_fcn(tVec, omega, X_train, Y_train):
    BN = compute_BN(tVec, omega)
    X_train_B = np.einsum('ijk,ik->ij', BN, X_train)  # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428
    Y_train_B = np.einsum('ijk,ik->ij', BN, Y_train)  
    return X_train_B, Y_train_B



def run(hparams):


    from Scripts.AsteroidScenarios.AnalysisBaseClass import AnalysisBaseClass
    from Scripts.AsteroidScenarios.Scenarios import ScenarioPositions
    from Scripts.AsteroidScenarios.helper_functions import get_trajectory_data
    from StatOD.data import get_measurements_general
    from StatOD.dynamics import get_rot_DMC_zero_order
    from StatOD.filters import ExtendedKalmanFilter
    from StatOD.measurements import h_pos
    import numpy as np
    import os
    import StatOD
    import matplotlib.pyplot as plt

    from StatOD.utils import pinnGravityModel

    q = hparams['q_value']
    epochs = hparams['epochs'] 
    lr = hparams['learning_rate'] 
    batch_size = hparams['batch_size'] 
    pinn_constraint_fcn = hparams['train_fcn']
    bc_data = hparams['boundary_condition_data']
    Q0 = np.eye(3) * q ** 2


    dim_constants = {
        "t_star" : 1E4,
        "m_star" : 1E0,
        "l_star" : 1E1
    }

    # load trajectory data and initialize state, covariance
    traj_file = "traj_rotating"
    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack((traj_data['X'][0], traj_data['W_pinn'][0]))
    P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-7, 1E-7, 1E-7])**2
    

    # Measurement information
    measurement_file = f"Data/Measurements/Position/{traj_file}_meas_noiseless.data"
    t_vec, Y_vec, h_args_vec = get_measurements_general(measurement_file, t_gap=60, data_fraction=1)
    R_diag = np.array([1E-3, 1E-3, 1E-3])**2


    # Initialize the PINN-GM
    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn['l_star'] *= 1E3
    model = pinnGravityModel(os.path.dirname(StatOD.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v4.data",
        learning_rate=lr,
        dim_constants=dim_constants_pinn)
    model.set_PINN_training_fcn(pinn_constraint_fcn)


    # Dynamics and noise information 
    eros_pos = np.zeros((6,))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_rot_DMC_zero_order()
    f_args = np.hstack((model, eros_pos, 0.0, ErosParams().omega))
    f_args = np.full((len(t_vec), len(f_args)), f_args)
    f_args[:,-2] = t_vec
    f_args[:,-1] = ErosParams().omega

    Q0 = np.eye(3)*(q)**2

    scenario = ScenarioPositions({
        'dim_constants' : [dim_constants],
        'N_states' : [len(x0)],
        'model' : [model]
    })    

    scenario.initializeMeasurements(
        t_vec=t_vec,
        Y_vec=Y_vec, 
        R=R_diag, 
        h_fcn=h_pos,
        h_args_vec=h_args_vec
        )
    
    scenario.initializeDynamics(
        f_fcn=f_fcn,
        dfdx_fcn=dfdx_fcn,
        f_args=f_args
    )

    scenario.initializeNoise(
        q_fcn=q_fcn,
        q_args=q_args,
        Q0=Q0
    )
    
    scenario.initializeIC(
        t0=t_vec[0],
        x0=x0,
        P0=P_diag
    )

    scenario.non_dimensionalize()
    scenario.initializeFilter(ExtendedKalmanFilter)

    network_train_config = {
        'batch_size' : batch_size,
        'epochs' : epochs,
        'BC_data' : bc_data,
        'rotating' : True,
        'rotating_fcn' : rotating_fcn
    }
    scenario.run(network_train_config)
    scenario.dimensionalize()


    analysis = AnalysisBaseClass(scenario)
    
    # convert from N to B frame
    logger = analysis.scenario.filter.logger
    BN = compute_BN(logger.t_i, ErosParams().omega)
    X_B = np.einsum('ijk,ik->ij', BN, logger.x_hat_i_plus[:,0:3])
    analysis.scenario.filter.logger.x_hat_i_plus[:,0:3] = X_B
    
    # plot in B-Frame
    percent_error_vec = analysis.run_planes_experiment()
    
    # save the updated network in a custom network directory
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    hparams.update({'results' : percent_error_vec})
    model.config.update({'hparams' : [hparams]})
    model.save(None, data_dir) # save the network, but not into the directory right now


    # add planes experiment and record metrics into dataframe for parallel coordinate plot (plotly)
    return model.config



if __name__ == "__main__":
    main()