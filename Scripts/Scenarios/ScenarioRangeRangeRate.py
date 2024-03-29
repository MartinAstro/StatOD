import os

import matplotlib.pyplot as plt
import numpy as np

import StatOD
from Scripts.AsteroidScenarios.AnalysisBaseClass import AnalysisBaseClass
from Scripts.AsteroidScenarios.helper_functions import get_trajectory_data
from Scripts.AsteroidScenarios.Scenarios import *
from StatOD.data import get_measurements_general
from StatOD.dynamics_DMC import get_DMC_zero_order
from StatOD.filters import ExtendedKalmanFilter
from StatOD.measurements import h_rho_rhod
from StatOD.models import pinnGravityModel


def main():
    dim_constants = {
        "t_star": 1e4,
        "m_star": 1e0,
        "l_star": 1e1,
    }

    # load trajectory data and initialize state, covariance
    traj_file = "trajectory_asteroid_inclined_high_alt_30_timestep"
    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack((traj_data["X"][0].copy(), [0.0, 0.0, 0.0]))
    P_diag = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-7, 1e-7, 1e-7]) ** 2

    # Measurement information
    measurement_file = f"Data/Measurements/Range/{traj_file}_meas_noiseless.data"
    t_vec, Y_vec, h_args_vec = get_measurements_general(
        measurement_file,
        t_gap=30,
        data_fraction=0.15,
    )
    R_diag = np.array([1e-3, 1e-6]) ** 2

    # Initialize the PINN-GM
    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn["l_star"] *= 1e3
    model = pinnGravityModel(
        os.path.dirname(StatOD.__file__)
        + "/../Data/Dataframes/eros_point_mass_v4.data",
        learning_rate=5e-6,
        dim_constants=dim_constants_pinn,
    )
    model.set_PINN_training_fcn("pinn_a")

    # Dynamics and noise information
    eros_pos = np.zeros((6,))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_DMC_zero_order()
    f_args = np.hstack((model, eros_pos))
    f_args = np.full((len(t_vec), len(f_args)), f_args)
    Q0 = np.eye(3) * (1e-9) ** 2

    scenario = ScenarioRangeRangeRate(
        {
            "dim_constants": [dim_constants],
            "N_states": [len(x0)],
            "model": [model],
        },
    )

    scenario.initializeMeasurements(
        t_vec=t_vec,
        Y_vec=Y_vec,
        R=R_diag,
        h_fcn=h_rho_rhod,
        h_args_vec=h_args_vec,
    )

    scenario.initializeDynamics(
        f_fcn=f_fcn,
        dfdx_fcn=dfdx_fcn,
        f_args=f_args,
    )

    scenario.initializeNoise(
        q_fcn=q_fcn,
        q_args=q_args,
        Q0=Q0,
    )

    scenario.initializeIC(
        t0=t_vec[0],
        x0=x0,
        P0=P_diag,
    )

    scenario.non_dimensionalize()
    scenario.initializeFilter(ExtendedKalmanFilter)

    network_train_config = {
        "batch_size": 32,
        "epochs": 0,
        "BC_data": False,
    }
    scenario.run(network_train_config)
    scenario.dimensionalize()

    analysis = AnalysisBaseClass(scenario)
    analysis.generate_y_hat()
    analysis.generate_filter_plots(
        x_truth=np.hstack((traj_data["X"], traj_data["W_pinn"])),
        w_truth=traj_data["W_pinn"],
        y_labels=["rho", "rho dot"],
    )
    analysis.generate_gravity_plots()
    plt.show()


if __name__ == "__main__":
    main()
