import matplotlib.pyplot as plt
import numpy as np
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from Scripts.Scenarios.helper_functions import *

from Scripts.Visualization.FilterVisualizer import FilterVisualizer
from Scripts.Scenarios.helper_functions import get_trajectory_data
from Scripts.Scenarios.ScenarioPositions import ScenarioPositions
from StatOD.constants import ErosParams
from StatOD.data import get_measurements_general
from StatOD.dynamics import *
from StatOD.filters import ExtendedKalmanFilter
from StatOD.measurements import h_pos
from StatOD.models import pinnGravityModel
from StatOD.visualizations import *
from Scripts.Analysis.AnalysisBaseClass import AnalysisBaseClass
from Scripts.Visualization.GravityPlanesVisualizer import GravityPlanesVisualizer

plt.switch_backend("WebAgg")


def rotating_fcn(tVec, omega, X_train, Y_train):
    BN = compute_BN(tVec, omega)
    X_train_B = np.einsum(
        "ijk,ik->ij",
        BN,
        X_train,
    )  # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428
    Y_train_B = np.einsum("ijk,ik->ij", BN, Y_train)
    return X_train_B, Y_train_B


def EKF_Rotating_Scenario(pinn_file, traj_file, hparams, show=False):

    q = hparams.get("q_value", [1e-9])[0]
    r = hparams.get("r_value", [1e-12])[0]
    epochs = hparams.get("epochs", [100])[0]
    lr = hparams.get("learning_rate", [1e-4])[0]
    batch_size = hparams.get("batch_size", [1024])[0]
    pinn_constraint_fcn = hparams.get("train_fcn", ["pinn_a"])[0]
    bc_data = hparams.get("boundary_condition_data", [False])[0]
    measurement_noise = hparams.get("measurement_noise", ["noiseless"])[0]

    dim_constants = {"t_star": 1e4, "m_star": 1e0, "l_star": 1e1}

    # load trajectory data and initialize state, covariance

    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack(
        (
            traj_data["X"][0],  # + np.random.uniform(-1e-6, 1e-6, size=(6,)),
            traj_data["W_pinn"][0],  # + np.random.uniform(-1e-9, 1e-9, size=(3,)),
        ),
    )
    P_diag = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-7, 1e-7, 1e-7]) ** 2

    ###########################
    # Measurement information #
    ###########################

    measurement_file = (
        f"Data/Measurements/Position/{traj_file}_meas_{measurement_noise}.data"
    )
    t_vec, Y_vec, h_args_vec = get_measurements_general(
        measurement_file,
        t_gap=60,
        data_fraction=0.1,
    )

    R_diag = np.eye(3) * r**2

    ###########################
    # Initialize the PINN-GM  #
    ###########################

    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn["l_star"] *= 1e3
    model = pinnGravityModel(
        pinn_file,
        learning_rate=lr,
        dim_constants=dim_constants_pinn,
    )
    model.set_PINN_training_fcn(pinn_constraint_fcn)

    ##################################
    # Dynamics and noise information #
    ##################################

    eros_pos = np.zeros((6,))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_rot_DMC_zero_order()
    f_args = np.hstack((model, eros_pos, 0.0, ErosParams().omega))
    f_args = np.full((len(t_vec), len(f_args)), f_args)
    f_args[:, -2] = t_vec
    f_args[:, -1] = ErosParams().omega

    Q0 = np.eye(3) * q**2

    scenario = ScenarioPositions(
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
        h_fcn=h_pos,
        h_args_vec=h_args_vec,
    )

    scenario.initializeDynamics(f_fcn=f_fcn, dfdx_fcn=dfdx_fcn, f_args=f_args)
    scenario.initializeNoise(q_fcn=q_fcn, q_args=q_args, Q0=Q0)
    scenario.initializeIC(t0=t_vec[0], x0=x0, P0=P_diag)

    scenario.non_dimensionalize()
    scenario.initializeFilter(ExtendedKalmanFilter)
    scenario.filter.atol = 1e-10
    scenario.filter.rtol = 1e-10

    network_train_config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "BC_data": bc_data,
        "rotating": True,
        "rotating_fcn": rotating_fcn,
        "synthetic_data": False,
        "num_samples": 1000,
    }
    scenario.run(network_train_config)
    scenario.dimensionalize()

    # Plot filter results
    vis = FilterVisualizer(scenario)
    vis.generate_y_hat()
    vis.generate_filter_plots(
        x_truth=np.hstack((traj_data["X"], traj_data["W_pinn"])),
        w_truth=traj_data["W_pinn"],
        y_labels=["x", "y", "z"],
    )

    # convert from N to B frame
    logger = vis.scenario.filter.logger
    BN = compute_BN(logger.t_i, ErosParams().omega)
    X_B = np.einsum("ijk,ik->ij", BN, logger.x_hat_i_plus[:, 0:3])
    vis.scenario.filter.logger.x_hat_i_plus[:, 0:3] = X_B

    # Run all the experiments on the trained model
    analysis = AnalysisBaseClass(scenario)
    analysis.true_gravity_fcn = get_hetero_poly_data
    metrics = analysis.run()

    if show:
        # Plot experiment results
        planes_vis = GravityPlanesVisualizer()
        planes_vis.run(
            analysis.planes_exp,
            max_error=10,
            logger=vis.scenario.filter.logger,
        )
        plt.show()

    data_dir = os.path.dirname(StatOD.__file__) + "/../Data"
    hparams.update(metrics)
    model.config.update({"hparams": [hparams]})
    model.save(None, data_dir)  # save the network, but not into the directory right now

    # add planes experiment and record metrics into dataframe for parallel coordinate plot (plotly)
    return model.config


if __name__ == "__main__":

    statOD_dir = os.path.dirname(StatOD.__file__)

    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly_no_fuse.data"
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly_dropout.data"
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"

    traj_file = "traj_rotating_gen_III_constant_no_fuse"
    traj_file = "traj_rotating_gen_III_constant_dropout"  # 10 orbits
    traj_file = "traj_rotating_gen_III_constant"

    hparams = {
        "q_value": [1e-9],
        "r_value": [1e-12],
        "epochs": [100],
        "learning_rate": [1e-4],
        "batch_size": [1024],
        "train_fcn": ["pinn_a"],
        "boundary_condition_data": [False],
        "measurement_noise": ["noiseless"],
    }

    EKF_Rotating_Scenario(pinn_file, traj_file, hparams, show=False)
