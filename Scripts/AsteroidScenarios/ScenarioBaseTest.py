from Scripts.AsteroidScenarios.AnalysisBaseClass import AnalysisBaseClass
from Scripts.AsteroidScenarios.ScenarioBaseClass import ScenarioBaseClass
from Scripts.AsteroidScenarios.helper_functions import get_trajectory_data
from StatOD.data import get_measurements_general
from StatOD.dynamics_DMC import get_DMC_zero_order
from StatOD.filters import ExtendedKalmanFilter
from StatOD.measurements import h_pos
import numpy as np
import os
import StatOD
import matplotlib.pyplot as plt

from StatOD.utils import pinnGravityModel


def dimensionalize(scenario):
    t_star = scenario.t_star
    l_star = scenario.l_star
    ms = scenario.ms
    ms2 = scenario.ms2

    scenario.filter.logger.x_hat_i_plus[:,0:3] *= l_star
    scenario.filter.logger.x_hat_i_plus[:,3:6] *= ms
    scenario.filter.logger.x_hat_i_plus[:,6:9] *= ms2

    scenario.t *= t_star
    scenario.Y *= l_star
    
    scenario.filter.logger.t_i *= t_star
    scenario.filter.logger.P_i_plus[:,0:3,0:3] *= l_star**2
    scenario.filter.logger.P_i_plus[:,3:6,3:6] *= ms**2
    scenario.filter.logger.P_i_plus[:,6:9,6:9] *= ms2**2
    
    scenario.R *= l_star**2
        
def non_dimensionalize(scenario):

    scenario.x0[0:3] /= scenario.l_star
    scenario.x0[3:6] /= scenario.ms
    scenario.x0[6:9] /= scenario.ms2
    
    scenario.P0[0:3,0:3] /= scenario.l_star**2
    scenario.P0[3:6,3:6] /= scenario.ms**2
    scenario.P0[6:9,6:9] /= scenario.ms2**2

    scenario.Y /= scenario.l_star

    scenario.R /= scenario.l_star**2

    scenario.t /= scenario.t_star
    # tau /= scenario.t_star
    scenario.Q0 /= scenario.ms2**2




def main():
    dim_constants = {
        "t_star" : 1E4,
        "m_star" : 1E0,
        "l_star" : 1E1
    }

    # load trajectory data and initialize state, covariance
    traj_file = "trajectory_asteroid_inclined_high_alt_30_timestep"
    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack((traj_data['X'][0].copy(), [0.0, 0.0, 0.0]))
    P_diag = np.array([1E-3, 1E-3, 1E-3, 1E-4, 1E-4, 1E-4, 1E-7, 1E-7, 1E-7])**2
    

    # Measurement information
    measurement_file = f"Data/Measurements/Position/{traj_file}_meas_noiseless.data"
    t_vec, Y_vec, h_args_vec = get_measurements_general(measurement_file, t_gap=30, data_fraction=0.25)
    R_diag = np.array([1E-3, 1E-3, 1E-3])**2


    # Initialize the PINN-GM
    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn['l_star'] *= 1E3
    model = pinnGravityModel(os.path.dirname(StatOD.__file__) + \
        "/../Data/Dataframes/eros_point_mass_v4.data",
        learning_rate=5E-6,
        dim_constants=dim_constants_pinn)
    model.set_PINN_training_fcn("pinn_a")


    # Dynamics and noise information 
    eros_pos = np.zeros((6,))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_DMC_zero_order()
    f_args = np.hstack((model, eros_pos))
    f_args = np.full((len(t_vec), len(f_args)), f_args)
    Q0 = np.eye(3)*(1E-9)**2

    scenario = ScenarioBaseClass({
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

    non_dimensionalize(scenario)
    scenario.initializeFilter(ExtendedKalmanFilter)

    network_train_config = {
        'batch_size' : 32,
        'epochs' : 0,
        'BC_data' : False
    }
    scenario.run(network_train_config)
    dimensionalize(scenario)


    analysis = AnalysisBaseClass(scenario)
    analysis.generate_y_hat()
    analysis.generate_filter_plots(
        x_truth=np.hstack((traj_data['X'], traj_data['W_pinn'])),
        w_truth=traj_data['W_pinn'],
        y_labels=['x', 'y', 'z']
        )
    analysis.generate_gravity_plots()
    plt.show()


if __name__ == "__main__":
    main()