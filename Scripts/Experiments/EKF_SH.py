"""
Kalman Filter with Dynamic Model Compensation Example
============================================================

"""
import os

import matplotlib.pyplot as plt
import numpy as np
from helper_functions import *

import StatOD
from Scripts.AsteroidScenarios.AnalysisBaseClass import AnalysisBaseClass
from Scripts.AsteroidScenarios.helper_functions import get_trajectory_data
from Scripts.AsteroidScenarios.Scenarios import ScenarioPositions
from StatOD.constants import *
from StatOD.data import get_measurements_general
from StatOD.dynamics import *
from StatOD.filters import ExtendedKalmanFilter
from StatOD.measurements import h_pos
from StatOD.models import pinnGravityModel
from StatOD.visualizations import *


def rotating_fcn(tVec, omega, X_train, Y_train):
    BN = compute_BN(tVec, omega)
    X_train_B = np.einsum(
        "ijk,ik->ij", BN, X_train
    )  # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428
    Y_train_B = np.einsum("ijk,ik->ij", BN, Y_train)
    return X_train_B, Y_train_B


def main():
    dim_constants = {
        "t_star": 1e4,
        "m_star": 1e0,
        "l_star": 1e1,
    }

    # load trajectory data and initialize state, covariance
    traj_file = "traj_rotating"
    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack((traj_data["X"][0], traj_data["W_pinn"][0]))
    P_diag = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-7, 1e-7, 1e-7]) ** 2

    # Measurement information
    measurement_file = f"Data/Measurements/Position/{traj_file}_meas_noiseless.data"
    t_vec, Y_vec, h_args_vec = get_measurements_general(
        measurement_file, t_gap=60, data_fraction=1
    )
    R_diag = np.array([1e-3, 1e-3, 1e-3]) ** 2
    R_diag = np.array([1e-12, 1e-12, 1e-12]) ** 2

    # Initialize the PINN-GM
    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn["l_star"] *= 1e3
    model = pinnGravityModel(
        os.path.dirname(StatOD.__file__)
        + "/../Data/Dataframes/eros_point_mass_v4.data",
        learning_rate=1e-5,
        dim_constants=dim_constants_pinn,
    )
    model.set_PINN_training_fcn("pinn_a")

    # Dynamics and noise information
    eros_pos = np.zeros((6,))
    f_fcn, dfdx_fcn = dynamics(x0, f_C22)
    f_fcn, dfdx_fcn, q_fcn, q_args = get_rot_DMC_zero_order()
    f_args = np.hstack((model, eros_pos, 0.0, ErosParams().omega))
    f_args = np.full((len(t_vec), len(f_args)), f_args)
    f_args[:, -2] = t_vec
    f_args[:, -1] = ErosParams().omega

    Q0 = np.eye(3) * (1e-9) ** 2

    scenario = ScenarioPositions(
        {
            "dim_constants": [dim_constants],
            "N_states": [len(x0)],
            "model": [model],
        }
    )

    scenario.initializeMeasurements(
        t_vec=t_vec,
        Y_vec=Y_vec,
        R=R_diag,
        h_fcn=h_pos,
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
        "batch_size": 1024,
        "epochs": 100,
        "BC_data": False,
        "rotating": True,
        "rotating_fcn": rotating_fcn,
    }
    scenario.run(network_train_config)
    scenario.dimensionalize()

    analysis = AnalysisBaseClass(scenario)
    analysis.generate_y_hat()
    analysis.generate_filter_plots(
        x_truth=np.hstack((traj_data["X"], traj_data["W_pinn"])),
        w_truth=traj_data["W_pinn"],
        y_labels=["x", "y", "z"],
    )

    # convert from N to B frame
    logger = analysis.scenario.filter.logger
    BN = compute_BN(logger.t_i, ErosParams().omega)
    X_B = np.einsum("ijk,ik->ij", BN, logger.x_hat_i_plus[:, 0:3])
    analysis.scenario.filter.logger.x_hat_i_plus[:, 0:3] = X_B

    # plot in B-Frame
    analysis.generate_gravity_plots()
    plt.show()


if __name__ == "__main__":
    main()
