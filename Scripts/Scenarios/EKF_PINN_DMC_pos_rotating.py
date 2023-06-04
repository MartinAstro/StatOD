import matplotlib.pyplot as plt
import numpy as np

from Scripts.Factories.CallbackFactory import CallbackFactory
from Scripts.Factories.DynArgsFactory import DynArgsFactory
from Scripts.Scenarios.ScenarioPositions import ScenarioPositions
from StatOD.callbacks import *
from StatOD.data import get_measurements_general
from StatOD.dynamics import *
from StatOD.filters import ExtendedKalmanFilter, KalmanFilter
from StatOD.measurements import h_pos
from StatOD.models import pinnGravityModel
from StatOD.utils import *
from StatOD.utils import dict_values_to_list
from StatOD.visualization.visualizations import *

plt.switch_backend("WebAgg")


def rotating_fcn(tVec, omega, X_train, Y_train):
    BN = compute_BN(tVec, omega)

    # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428
    X_train_B = np.einsum(
        "ijk,ik->ij",
        BN,
        X_train,
    )
    Y_train_B = np.einsum("ijk,ik->ij", BN, Y_train)
    return X_train_B, Y_train_B


def EKF_Rotating_Scenario(pinn_file, traj_file, hparams, output_file, show=False):
    q = hparams.get("q_value", [1e-9])[0]
    r = hparams.get("r_value", [1e-12])[0]
    epochs = hparams.get("epochs", [100])[0]
    lr = hparams.get("learning_rate", [1e-4])[0]
    batch_size = hparams.get("batch_size", [1024])[0]
    pinn_constraint_fcn = hparams.get("train_fcn", ["pinn_a"])[0]
    bc_data = hparams.get("boundary_condition_data", [False])[0]
    measurement_noise = hparams.get("measurement_noise", ["noiseless"])[0]
    eager = hparams.get("eager", [False])[0]
    data_frac = hparams.get("data_fraction", [1.0])[0]
    dim_constants = {"t_star": 1e4, "m_star": 1e0, "l_star": 1e1}

    # load trajectory data and initialize state, covariance

    traj_data = get_trajectory_data(traj_file)
    x0 = np.hstack(
        (
            traj_data["X"][0],  # + np.random.uniform(-1e-6, 1e-6, size=(6,)),
            traj_data["W_pinn"][0],  # + np.random.uniform(-1e-9, 1e-9, size=(3,)),
        ),
    )
    dx0 = np.zeros_like(x0)
    P_diag = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-9, 1e-9, 1e-9]) ** 2

    ###########################
    # Measurement information #
    ###########################

    measurement_file = (
        f"Data/Measurements/Position/{traj_file}_meas_{measurement_noise}.data"
    )
    t_vec, Y_vec, h_args_vec = get_measurements_general(
        measurement_file,
        t_gap=60,
        data_fraction=data_frac,  # 0.001,
    )

    R_diag = np.eye(3) * r**2

    ###########################
    # Initialize the PINN-GM  #
    ###########################

    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn["l_star"] *= 1e3
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"

    model = pinnGravityModel(
        pinn_file,
        learning_rate=lr,
        dim_constants=dim_constants_pinn,
        custom_data_dir=statOD_dir,
        eager=eager,
    )
    model.set_PINN_training_fcn(pinn_constraint_fcn)

    ##################################
    # Dynamics and noise information #
    ##################################

    f_fcn, dfdx_fcn, q_fcn, q_args = get_DMC_zero_order()
    f_args = DynArgsFactory().get_gravity_only_args(model, t_vec)

    Q0 = np.eye(3) * q**2
    Q_dt = 60.0
    # Q_dt = 0.1

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
    scenario.initializeNoise(q_fcn=q_fcn, q_args=q_args, Q0=Q0, Q_dt=Q_dt)
    scenario.initializeIC(t0=t_vec[0], x0=x0, P0=P_diag, dx0=dx0)

    scenario.non_dimensionalize()
    scenario.initializeFilter(KalmanFilter)
    scenario.initializeFilter(ExtendedKalmanFilter)
    scenario.filter.atol = 1e-10
    scenario.filter.rtol = 1e-10

    # Initialize Callbacks
    callbacks_dict = CallbackFactory.generate_callbacks()

    network_train_config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "BC_data": bc_data,
        "rotating": True,
        "rotating_fcn": rotating_fcn,
        "synthetic_data": False,
        "num_samples": 1000,
        "callbacks": callbacks_dict,
        # "X_COM": np.array([[new_COM, 0.0, 0.0]]),
        # # "X_COM": np.array([[10.0, 0.0, 0.0]]),
        # "COM_samples": 1,
        # "COM_radius": 1e-12,
    }
    callbacks = scenario.run(network_train_config)
    scenario.dimensionalize()

    # compute metrics and update the config file
    metrics = {}
    for name, callback in callbacks.items():
        metrics[name] = callback.data[-1]
    metrics = dict_values_to_list(metrics)  # ensures compatability
    model.gravity_model.config.update(metrics)

    # save the model + config
    model.save(output_file)

    # vis = FilterVisualizer(scenario)
    # vis.generate_y_hat()
    # vis.generate_filter_plots(
    #     x_truth=np.hstack((traj_data["X"], traj_data["W_pinn"])),
    #     w_truth=traj_data["W_pinn"],
    #     y_labels=["x", "y", "z"],
    # )

    if show:
        plt.show()

    return model.config


if __name__ == "__main__":
    import time

    start_time = time.time()
    pinn_file = "Data/Dataframes/eros_poly_053123.data"
    traj_file = "traj_eros_poly_053123"
    output_file = "output_filter_060123.data"
    hparams = {
        # "q_value": [5e-8],
        "q_value": [1e-7],
        # "q_value": [1e-7],
        "r_value": [1e-12],
        "epochs": [500],
        "learning_rate": [1e-4],
        "batch_size": [20000],
        "train_fcn": ["pinn_a"],
        "boundary_condition_data": [False],
        # "measurement_noise": ["noisy"],
        "measurement_noise": ["noiseless"],
        "eager": [False],
        "data_fraction": [0.1],  # 1.0],
    }

    EKF_Rotating_Scenario(pinn_file, traj_file, hparams, output_file, show=True)
    print("Total Time: " + str(time.time() - start_time))
