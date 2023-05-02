import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


def plot_DMC_subplot(x, y1, y2):
    plt.plot(x, y1)
    plt.plot(x, y2)
    criteria1 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 > 0)))).T, axis=1)
    criteria2 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 < 0)))).T, axis=1)
    criteria3 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 < 0)))).T, axis=1)
    criteria4 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 > 0)))).T, axis=1)
    percent_productive = np.round(
        (np.count_nonzero(criteria1) + np.count_nonzero(criteria2)) / len(x) * 100,
        2,
    )
    plt.gca().annotate(
        f"Percent Useful: {percent_productive}",
        xy=(0.75, 0.75),
        xycoords="axes fraction",
        size=8,
    )
    plt.gca().fill_between(
        x,
        y1,
        y2,
        where=criteria1,
        color="green",
        alpha=0.3,
        interpolate=True,
    )
    plt.gca().fill_between(
        x,
        y1,
        y2,
        where=criteria2,
        color="green",
        alpha=0.3,
        interpolate=True,
    )
    plt.gca().fill_between(
        x,
        y1,
        y2,
        where=criteria3,
        color="red",
        alpha=0.3,
        interpolate=True,
    )
    plt.gca().fill_between(
        x,
        y1,
        y2,
        where=criteria4,
        color="red",
        alpha=0.3,
        interpolate=True,
    )


def plot_DMC(logger, w_truth):
    idx_max = len(logger.t_i) if len(logger.t_i) < len(w_truth) else len(w_truth)

    plt.figure()
    plt.subplot(311)
    plot_DMC_subplot(
        logger.t_i[:idx_max],
        logger.x_hat_i_plus[:idx_max, 6],
        w_truth[:idx_max, 0],
    )
    plt.subplot(312)
    plot_DMC_subplot(
        logger.t_i[:idx_max],
        logger.x_hat_i_plus[:idx_max, 7],
        w_truth[:idx_max, 1],
    )
    plt.subplot(313)
    plot_DMC_subplot(
        logger.t_i[:idx_max],
        logger.x_hat_i_plus[:idx_max, 8],
        w_truth[:idx_max, 2],
    )

    # Plot magnitude
    DMC_mag = np.linalg.norm(logger.x_hat_i_plus[:, 6:9], axis=1)
    plt.figure()
    plt.plot(DMC_mag)
    print(f"Average DMC Mag {np.mean(DMC_mag)}")


def plot_error_planes(planes_exp, max_error, logger):
    visPlanes = PlanesVisualizer(planes_exp)
    plt.rc("text", usetex=False)
    X_traj = logger.x_hat_i_plus[:, 0:3] * 1e3 / Eros().radius
    x = visPlanes.experiment.x_test
    y = visPlanes.experiment.percent_error_acc
    plt.figure()
    visPlanes.max = max_error
    visPlanes.plot_plane(x, y, plane="xy")
    plt.plot(X_traj[:, 0], X_traj[:, 1], color="black", linewidth=0.5)
    plt.figure()
    visPlanes.plot_plane(x, y, plane="xz")
    plt.plot(X_traj[:, 0], X_traj[:, 2], color="black", linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.figure()
    visPlanes.plot_plane(x, y, plane="yz")
    plt.plot(X_traj[:, 1], X_traj[:, 2], color="black", linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])


def get_trajectory_data(data_file):
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + f"Data/Trajectories/{data_file}.data", "rb") as f:
        data = pickle.load(f)
    return data


def boundary_condition_data(N, dim_constants):
    s = np.random.uniform(-1, 1, size=(N,))
    t = np.random.uniform(-1, 1, size=(N,))
    u = np.random.uniform(-1, 1, size=(N,))

    # Forces data to be on sphere of radius r*500
    coordinates = np.vstack([s, t, u]).T
    r_mag = np.linalg.norm(coordinates, axis=1)
    s /= r_mag
    t /= r_mag
    u /= r_mag

    # r = Eros().radius*4
    r = np.random.uniform(Eros().radius * 4, Eros().radius * 6, size=s.shape)
    x = r * s
    y = r * t
    z = r * u
    X_train = np.column_stack((x, y, z))

    pm_gravity = PointMass(Eros())
    Y_train = pm_gravity.compute_acceleration(X_train)

    X_train /= 1000  # convert to km
    Y_train /= 1000  # convert to km

    X_train /= dim_constants["l_star"]
    Y_train /= dim_constants["l_star"] / dim_constants["t_star"] ** 2

    return X_train, Y_train


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
        config = dict(sorted(config.items(), key=lambda kv: kv[0]))
        config["PINN_constraint_fcn"] = [
            config["PINN_constraint_fcn"][0],
        ]  # Can't have multiple args in each list
        df = pd.DataFrame().from_dict(config).set_index("timetag")

        try:
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except:
            df.to_pickle(df_file)


def compute_BN(tVec, omega):
    theta = tVec * omega
    C00 = np.cos(theta)
    C01 = -np.sin(theta)
    C10 = np.sin(theta)
    C11 = np.cos(theta)
    Cij = np.zeros_like(C00)
    C22 = np.zeros_like(C00) + 1

    C = np.block(
        [
            [[C00], [C01], [Cij]],
            [[C10], [C11], [Cij]],
            [[Cij], [Cij], [C22]],
        ],
    )
    C = np.transpose(C, axes=[2, 0, 1])

    return C
