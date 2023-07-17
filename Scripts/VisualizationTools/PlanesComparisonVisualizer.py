import os
import pickle

import GravNN
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.TrajectoryExperiment import TrajectoryExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_sym_model,
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase

import StatOD
from StatOD.utils import (
    compute_semimajor,
)


class PlanesComparisonVisualizer(VisualizationBase):
    def __init__(self, original_model, new_model, **kwargs):
        super().__init__(**kwargs)
        self.new_model = new_model
        self.original_model = original_model

        self.new_exp = {}
        self.original_exp = {}

        load_exp = kwargs.get("load_exp", False)
        self.run_original_experiments(load_exp)
        self.run_new_experiments(load_exp)

    def run_new_experiments(self, load_exp=False):
        if load_exp:
            with open("best_model_exp.data", "rb") as f:
                self.planes_exp = pickle.load(f)
                self.extrap_exp = pickle.load(f)
                self.traj_exps = pickle.load(f)
            return

        #####################
        # Planes Experiment
        #####################
        planes_exp = PlanesExperiment(
            self.new_model,
            self.new_model.config,
            [
                -self.new_model.config["planet"][0].radius * 2,
                self.new_model.config["planet"][0].radius * 2,
            ],
            50,
            omit_train_data=True,
        )
        planes_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        planes_exp.run()
        self.new_exp["planes"] = planes_exp

        #########################
        # Extrapolation Experiment
        #########################
        extrap_exp = ExtrapolationExperiment(
            self.new_model,
            self.new_model.config,
            points=1000,
            omit_train_data=True,
        )
        extrap_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        extrap_exp.run()
        self.new_exp["extrap"] = extrap_exp

    def run_original_experiments(self, load_exp=False):
        if load_exp:
            with open("best_model_exp.data", "rb") as f:
                self.planes_exp = pickle.load(f)
                self.extrap_exp = pickle.load(f)
                self.traj_exps = pickle.load(f)
            return

        #####################
        # Planes Experiment
        #####################
        planes_exp = PlanesExperiment(
            self.original_model,
            self.new_model.config,
            [
                -self.new_model.config["planet"][0].radius * 2,
                self.new_model.config["planet"][0].radius * 2,
            ],
            50,
            omit_train_data=True,
        )
        planes_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        planes_exp.load_model_data(self.original_model)
        planes_exp.run()
        self.original_exp["planes"] = planes_exp

        #########################
        # Extrapolation Experiment
        #########################
        extrap_exp = ExtrapolationExperiment(
            self.original_model,
            self.new_model.config,
            points=1000,
            omit_train_data=True,
        )
        extrap_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        extrap_exp.run()
        self.original_exp["extrap"] = extrap_exp

    def plot(self, callback_idx=-1, X_B=None, max=10):
        vis = VisualizationBase()

        vis.fig_size = (vis.w_full, vis.h_full)
        fig = plt.figure(figsize=self.fig_size)
        gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[0.05, 1, 1, 1])

        # Original Model
        ax0 = fig.add_subplot(gs[0, 1 + 0])
        ax1 = fig.add_subplot(gs[0, 1 + 1])
        ax2 = fig.add_subplot(gs[0, 1 + 2])

        # Refined Model
        ax3 = fig.add_subplot(gs[1, 1 + 0])
        ax4 = fig.add_subplot(gs[1, 1 + 1])
        ax5 = fig.add_subplot(gs[1, 1 + 2])

        # Extrapolation
        ax6 = fig.add_subplot(gs[2, :])
        plt.subplots_adjust(wspace=0.00, hspace=0.00)
        plt.rc("text", usetex=True)

        def plot_single_plane(vis, ax, x, y, plane="xy", cbar=False, **kwargs):
            plt.sca(ax)
            vis.plot_plane(
                x,
                y,
                plane=plane,
                cbar=cbar,
                labels=False,
                ticks=False,
                annotate_stats=True,
                **kwargs,
            )

        original_vis = PlanesVisualizer(
            self.original_exp["planes"], halt_formatting=True
        )
        original_vis.max = max
        x = original_vis.experiment.x_test
        y = original_vis.experiment.percent_error_acc
        plot_single_plane(original_vis, ax0, x, y, plane="xy", cbar=False)
        plot_single_plane(original_vis, ax1, x, y, plane="xz", cbar=False)
        plot_single_plane(original_vis, ax2, x, y, plane="yz", cbar=False)

        new_vis = PlanesVisualizer(self.new_exp["planes"], halt_formatting=True)
        new_vis.max = max
        x = new_vis.experiment.x_test
        y = new_vis.experiment.percent_error_acc
        if X_B is not None:
            X_B *= 1e3 / vis.radius

        plot_single_plane(new_vis, ax3, x, y, plane="xy", cbar=False)
        plot_single_plane(new_vis, ax4, x, y, plane="xz", cbar=False)
        plot_single_plane(
            new_vis,
            ax5,
            x,
            y,
            plane="yz",
            cbar=True,
            cbar_gs=gs[0:2, 0],
            colorbar_label="Percent Error (\%)",
        )

        plt.sca(ax6)
        extrap_vis = ExtrapolationVisualizer(
            self.new_exp["extrap"],
            plot_fcn=plt.semilogy,
            halt_formatting=False,
            annotate=False,
        )
        extrap_vis.plot_interpolation_percent_error(
            new_fig=False, plot_max=False, plot_std=False, label="PINN", color="blue"
        )

        extrap_vis = ExtrapolationVisualizer(
            self.original_exp["extrap"],
            plot_fcn=plt.semilogy,
            halt_formatting=False,
            annotate=False,
        )
        extrap_vis.plot_interpolation_percent_error(
            new_fig=False, plot_max=False, plot_std=False, label="Poly", color="green"
        )
        plt.legend()


def main():
    # Deploy
    gravNN_dir = os.path.abspath(os.path.dirname(StatOD.__file__)) + "/../"
    df_file = gravNN_dir + "Data/Dataframes/best_case_model_071123.data"

    # load the dataframe
    df = pd.read_pickle(df_file)

    # load model and config
    config, pinn_model = load_config_and_model(df.id.values[-1], df, only_weights=True)

    # load the constant density polyhedral model
    poly_model = Polyhedral(Eros(), Eros().obj_200k)

    # run the visualization suite on the model
    vis = PlanesComparisonVisualizer(poly_model, pinn_model, load_exp=False)
    vis.fig_size = (vis.w_full, vis.w_full)
    vis.plot()

    StatOD_dir = os.path.abspath(os.path.dirname(StatOD.__file__)) + "/../"
    vis.save(plt.gcf(), f"{StatOD_dir}/Plots/panel_plot_new.pdf")
    plt.savefig(
        f"{StatOD_dir}/Plots/panel_plot_new.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    main()
