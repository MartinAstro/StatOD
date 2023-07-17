import matplotlib.pyplot as plt
import numpy as np
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_symmetric_data
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

from Scripts.Factories.CallbackFactory import CallbackFactory

# from Scripts.Factories.CallbackFactory import CallbackFactory
from Scripts.Factories.DynArgsFactory import DynArgsFactory
from StatOD.callbacks import *
from StatOD.data import get_measurements_general
from StatOD.dynamics import *
from StatOD.filters import ExtendedKalmanFilter
from StatOD.measurements import h_pos
from StatOD.models import pinnGravityModel
from StatOD.scenarios import ScenarioHF
from StatOD.utils import *
from StatOD.utils import dict_values_to_list
from StatOD.visualization.FilterVisualizer import FilterVisualizer
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


def plot_planes(gravity_model, config):
    planet = config["planet"][0]
    points = 100
    radius_bounds = [-2 * planet.radius, 2 * planet.radius]
    planes_exp = PlanesExperiment(
        gravity_model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()

    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=10, annotate_stats=True)


def print_metrics(metrics):
    pprint(metrics["Extrapolation"])
    pprint(metrics["Planes"][0])
    dX_sum = 0
    try:
        for key, value in metrics["Trajectory"][0].items():
            if "dX_sum" in key:
                dX_sum += value
        print(dX_sum / 4)
    except:
        pass
    return


def generate_plots(scenario, traj_data, model):
    # Plot filter results
    vis = FilterVisualizer(scenario)
    vis.generate_y_hat()
    vis.generate_filter_plots(
        x_truth=np.hstack((traj_data["X"], traj_data["W_pinn"])),
        w_truth=traj_data["W_pinn"],
        y_labels=["x", "y", "z"],
    )

    # convert from N to B frame
    # logger = vis.scenario.filter.logger
    # BN = compute_BN(logger.t_i, ErosParams().omega)
    # X_B = np.einsum("ijk,ik->ij", BN, logger.x_hat_i_plus[:, 0:3])
    # vis.scenario.filter.logger.x_hat_i_plus[:, 0:3] = X_B
    # vis.generate_filter_plots(traj_data["X"], traj_data["W_pinn"])

    # # run the visualization suite on the model
    # vis = ExperimentPanelVisualizer(model)
    # vis.plot(X_B=X_B)


def DMC_high_fidelity(pinn_file, traj_file, hparams, show=False, df_file=None):
    q = hparams.get("q_value", [1e-9])[0]
    r = hparams.get("r_value", [1e-12])[0]
    epochs = hparams.get("epochs", [100])[0]
    lr = hparams.get("learning_rate", [1e-4])[0]
    batch_size = hparams.get("batch_size", [1024])[0]
    meas_batch_size = hparams.get("meas_batch_size", [1024])[0]
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
            traj_data["X"][0],
            traj_data["W_pinn"][0],  # + np.random.uniform(-1e-9, 1e-9, size=(3,)),
        ),
    )
    P_diag = np.array([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-7, 1e-7, 1e-7]) ** 2

    pos_sigma = 1e-2
    vel_sigma = 1e-6 * 1e2
    acc_sigma = 1e-9
    P_diag = (
        np.array(
            [
                pos_sigma,
                pos_sigma,
                pos_sigma,
                vel_sigma,
                vel_sigma,
                vel_sigma,
                acc_sigma,
                acc_sigma,
                acc_sigma,
            ],
        )
        ** 2
    )

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
    # t_vec[1] = 0

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
        eager=eager,
    )
    model.set_PINN_training_fcn(pinn_constraint_fcn)
    model.gravity_model.config.update(
        {
            "gravity_data_fcn": [get_hetero_poly_symmetric_data],
        },
    )

    # plot_planes(model.gravity_model, model.gravity_model.config)

    ##################################
    # Dynamics and noise information #
    ##################################

    np.zeros((6,))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_DMC_HF_zero_order()
    f_args = DynArgsFactory().get_HF_args(model)
    q_args = [dfdx_fcn, f_args]

    f_args = np.full((len(t_vec), len(f_args)), f_args)
    f_args[:, -2] = t_vec

    Q0 = np.eye(3) * q**2
    Q_dt = 60.0

    scenario = ScenarioHF(
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
    scenario.initializeIC(t0=t_vec[0], x0=x0, P0=P_diag)

    scenario.non_dimensionalize()
    scenario.initializeFilter(ExtendedKalmanFilter)
    scenario.filter.atol = 1e-10
    scenario.filter.rtol = 1e-10

    # Initialize Callbacks
    callbacks_dict = CallbackFactory().generate_callbacks(
        radius_multiplier=2,
        skip_trajectory=False,
    )
    # callbacks_dict = {}

    network_train_config = {
        "batch_size": batch_size,
        "meas_batch_size": meas_batch_size,
        "epochs": epochs,
        "BC_data": bc_data,
        "rotating": True,
        "rotating_fcn": rotating_fcn,
        "synthetic_data": False,
        "num_samples": 1000,
        "callbacks": callbacks_dict,
        "print_interval": [100],
        # "COM_samples": 1,
        # "X_COM": np.array([[0.0, 0.0, 0.0]]),
        "COM_radius": None,
    }
    callbacks = scenario.run(network_train_config)
    scenario.dimensionalize()

    # compute metrics and update the config file
    metrics = {}
    for name, callback in callbacks.items():
        metrics[name] = callback.data[-1]
    metrics = dict_values_to_list(metrics)  # ensures compatability
    model.gravity_model.config.update(metrics)
    model.config.update(model.gravity_model.config)
    model.config.update(
        {
            "hparams": [hparams],
            "time_elapsed": [scenario.time_elapsed],
            "gravity_data_fcn": [get_hetero_poly_symmetric_data],
        },
    )

    # generate_plots(scenario, traj_data, model.gravity_model)
    # plot_planes(model.gravity_model, model.gravity_model.config)
    print_metrics(metrics)

    # save the model + config
    model.save(df_file)

    if show:
        plt.show()

    return model.config


if __name__ == "__main__":
    import time

    start_time = time.time()

    # model = "eros_poly_071123"
    model = "eros_poly_071123"
    pinn_file = f"Data/Dataframes/{model}.data"
    traj_file = f"traj_eros_poly_061023_32000.0_0.35000000000000003"

    statOD_dir = os.path.dirname(StatOD.__file__)
    df_file = statOD_dir + f"/../Data/Dataframes/best_case_model_071123.data"

    hparams = {
        "q_value": [1e-9],
        "r_value": [1e-12],
        "epochs": [10000],
        "learning_rate": [1e-4],
        "meas_batch_size": [32768],
        "train_fcn": ["pinn_a"],
        "boundary_condition_data": [False],
        "measurement_noise": ["noiseless"],
        "eager": [False],
        "data_fraction": [1.0],
        "batch_size" : [1024*2]
    }

    DMC_high_fidelity(pinn_file, traj_file, hparams, show=True, df_file=df_file)
    print("Total Time: " + str(time.time() - start_time))
