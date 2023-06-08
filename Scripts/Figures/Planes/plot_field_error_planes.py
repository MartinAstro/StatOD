import os
import pickle

import GravNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


def run_planes_exp(model, config, radius, density):
    planes_exp = PlanesExperiment(model, config, [-radius, radius], density)
    planes_exp.config["gravity_data_fcn"] = [get_poly_data]
    planes_exp.run()
    mask = planes_exp.get_planet_mask()
    planes_exp.percent_error_acc[mask] = np.nan
    print(f"Average Percent Error {np.nanmean(planes_exp.percent_error_acc)}")
    print(f"Median Percent Error {np.nanmedian(planes_exp.percent_error_acc)}")
    print(
        f"Pixels that exceed 10% error: {np.count_nonzero(planes_exp.percent_error_acc > 10)} / {len(mask)}",
    )
    return planes_exp


def generate_plot(planes_exp, max_percent, X_traj=None):
    if X_traj is None:
        X_traj = np.full((1, 3), np.nan)

    vis = PlanesVisualizer(planes_exp)

    plt.rc("font", size=6.0)
    plt.rc("figure", figsize=(6.5 / 3, 6.5 / 3))
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("axes", labelsize=8)

    vis.max = max_percent
    x = vis.experiment.x_test
    y = vis.experiment.percent_error_acc

    cbar_label = "Acceleration Percent Error"
    fig1 = plt.figure()
    vis.plot_plane(x, y, plane="xy", colorbar_label=cbar_label)
    plt.plot(X_traj[:, 0], X_traj[:, 1], color="black", linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    fig2 = plt.figure()
    vis.plot_plane(x, y, plane="xz", colorbar_label=cbar_label)
    plt.plot(X_traj[:, 0], X_traj[:, 2], color="black", linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    fig3 = plt.figure()
    vis.plot_plane(x, y, plane="yz", colorbar_label=cbar_label)
    plt.plot(X_traj[:, 1], X_traj[:, 2], color="black", linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    return [fig1, fig2, fig3]


def main():
    planet = Eros()
    radius = planet.radius * 4
    density = 50
    max_percent = 20

    dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(dir + "Data/Trajectories/traj_rotating.data", "rb") as f:
        data = pickle.load(f)

    X_m = data["X"][:] * 1e3 / planet.radius
    N = len(X_m) // 5
    X_m = X_m[0:N]

    # # # PINN trained on Point Mass
    # df = pd.read_pickle(dir + "Data/Dataframes/eros_point_mass_v4.data")
    # model_id = df["id"].values[model_idx] # ALC without extra
    # config, model = load_config_and_model(model_id, df)
    # planes_exp = run_planes_exp(model, config, radius, density)
    # fig_list = generate_plot(planes_exp, max_percent)
    # for idx, fig in enumerate(fig_list):
    #     plt.figure(fig.number)
    #     # plt.savefig(os.path.dirname(StatOD.__file__) + f"/../Plots/PM_Error_{idx}.pdf")

    # PINN trained using DMC
    # df = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm.data")
    # model_id = df["id"].values[32] # best
    dir = os.path.dirname(StatOD.__file__) + "/../"
    df = pd.read_pickle(dir + "Data/Dataframes/eros_point_mass_v5_2.data")
    model_id = df["id"].values[-1]  # best
    # df = pd.read_pickle(dir + "Data/Dataframes/hparams_rotating.data")
    # model_id = df["id"].values[261] # best
    config, model = load_config_and_model(
        model_id,
        df,
        custom_data_dir=os.path.dirname(GravNN.__file__) + "/../Data",
    )
    planes_exp = run_planes_exp(model, config, radius, density)
    fig_list = generate_plot(planes_exp, max_percent)
    for idx, fig in enumerate(fig_list):
        plt.figure(fig.number)
        # plt.savefig(os.path.dirname(StatOD.__file__) + f"/../Plots/PINN_DMC_Error_{idx}.pdf")

    # Spherical Harmonic Solution
    # dir = os.path.dirname(StatOD.__file__) + "/../"
    # original_sh_model = SphericalHarmonics(dir + 'Data/Products/SH_Eros_model.csv', degree=2)
    # model = sphericalHarmonicModel(original_sh_model) # wrapper to make compatable with planes experiment. TODO: Make general gravity model base class
    # planes_exp = run_planes_exp(model, config, radius, density)
    # fig_list = generate_plot(planes_exp, max_percent)
    # for idx, fig in enumerate(fig_list):
    #     plt.figure(fig.number)
    # plt.savefig(os.path.dirname(StatOD.__file__) + f"/../Plots

    plt.show()


if __name__ == "__main__":
    main()
