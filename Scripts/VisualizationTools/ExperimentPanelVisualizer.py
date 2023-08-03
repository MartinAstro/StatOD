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
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.Visualization.TrajectoryVisualizer import TrajectoryVisualizer
from GravNN.Visualization.VisualizationBase import VisualizationBase

import StatOD
from StatOD.utils import (
    compute_semimajor,
)


class ExperimentPanelVisualizer(VisualizationBase):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        load_exp = kwargs.get("load_exp", False)
        self.run_experiments(load_exp)

    def get_trajectories(self):
        X_1 = np.array(
            [
                -19243.595703125,
                21967.5078125,
                17404.74609375,
                -2.939612865447998,
                -1.1707247495651245,
                -1.7654979228973389,
            ],
        )
        X_2 = np.array(
            [
                -17720.09765625,
                29013.974609375,
                0.0,
                -3.0941531658172607,
                -1.8855023384094238,
                -0.0,
            ],
        )
        X_3 = np.array(
            [
                -22921.6484375,
                4955.83154296875,
                24614.02734375,
                -2.5665197372436523,
                0.5549010038375854,
                -2.496790885925293,
            ],
        )
        X_4 = np.array(
            [
                -45843.296875,
                9911.6630859375,
                49228.0546875,
                -1.8148034811019897,
                0.3923742473125458,
                -1.7654976844787598,
            ],
        )

        def compute_T(X, mu):
            a = compute_semimajor(X, mu)
            if np.isnan(a):
                a = 100
            n = np.sqrt(mu / a**3)
            T = 2 * np.pi / n
            return T

        X_list = np.array([X_1, X_2, X_3, X_4])
        mu = self.model.config["planet"][0].mu
        T_list = np.array([compute_T(X, mu) for X in X_list])
        return X_list, T_list

    def run_experiments(self, load_exp=False):
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
            self.model,
            self.model.config,
            [
                -self.model.config["planet"][0].radius * 2,
                self.model.config["planet"][0].radius * 2,
            ],
            100,
            omit_train_data=True,
        )
        planes_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        planes_exp.run()
        self.planes_exp = planes_exp

        #########################
        # Extrapolation Experiment
        #########################
        extrap_exp = ExtrapolationExperiment(
            self.model,
            self.model.config,
            points=1000,
            omit_train_data=True,
        )
        extrap_exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
        extrap_exp.run()

        self.extrap_exp = extrap_exp

        #########################
        # Trajectories Experiment
        #########################
        truth_model = generate_heterogeneous_sym_model(Eros(), Eros().obj_8k)
        X0_list, T_list = self.get_trajectories()
        traj_exps = []
        for i in range(len(X0_list)):
            X0 = X0_list[i]
            T = T_list[i]
            exp = TrajectoryExperiment(truth_model, X0, T)
            exp.add_test_model(self.model, "PINN", "red", "--")
            exp.run()
            traj_exps.append(exp)
        self.traj_exps = traj_exps

        # save the experiments to a pickle file
        with open("best_model_exp.data", "wb") as f:
            del planes_exp.model
            del extrap_exp.model
            for traj in traj_exps:
                for test_models in traj.test_models:
                    try:
                        test_models.pop("model")
                    except:
                        pass

            pickle.dump(planes_exp, f)
            pickle.dump(extrap_exp, f)
            pickle.dump(traj_exps, f)

    def plot(self, callback_idx=-1, X_B=None, max=10):
        vis = PlanesVisualizer(self.planes_exp, halt_formatting=True)
        vis.max = max

        vis.fig_size = (vis.w_full, vis.w_full / 2.0)
        fig = plt.figure(figsize=self.fig_size)
        gs = gridspec.GridSpec(2, 5, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        ax4 = fig.add_subplot(gs[1, 0:3])
        ax5 = fig.add_subplot(gs[:, 3:5], projection="3d")

        plt.subplots_adjust(wspace=0.00, hspace=0.00)

        plt.rc("text", usetex=True)

        x = vis.experiment.x_test
        y = vis.experiment.percent_error_acc
        if X_B is not None:
            X_B *= 1e3 / vis.radius
        plt.sca(ax1)
        vis.plot_plane(x, y, plane="xy", annotate_stats=False, cbar=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        if X_B is not None:
            ax1.plot(X_B[:, 0], X_B[:, 1], color="black", linewidth=1)

        plt.sca(ax2)
        vis.plot_plane(x, y, plane="xz", annotate_stats=False, cbar=False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        if X_B is not None:
            ax2.plot(X_B[:, 0], X_B[:, 2], color="black", linewidth=1)

        plt.sca(ax3)
        vis.plot_plane(x, y, plane="yz", annotate_stats=False, cbar=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        if X_B is not None:
            ax3.plot(X_B[:, 1], X_B[:, 2], color="black", linewidth=1)

        plt.sca(ax4)
        extrap_vis = ExtrapolationVisualizer(
            self.extrap_exp,
            plot_fcn=plt.semilogy,
            halt_formatting=False,
            annotate=False,
        )
        extrap_vis.plot_interpolation_percent_error(new_fig=False)

        plt.sca(ax5)
        for i, traj_exp in enumerate(self.traj_exps):
            traj_vis = TrajectoryVisualizer(traj_exp, halt_formatting=False)
            # new_fig = True if i == 0 else False
            traj_vis.plot_3d_trajectory(new_fig=False)

        # remove legend
        ax5.get_legend().remove()


def main():
    # load the output dataframe

    gravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__)) + "/../"
    df_file = gravNN_dir + "Data/Dataframes/ae_pair_results.data"

    # Testing
    gravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__)) + "/../"
    df_file = gravNN_dir + "Data/Dataframes/eros_poly_071123.data"

    # Deploy
    gravNN_dir = os.path.abspath(os.path.dirname(StatOD.__file__)) + "/../"
    df_file = gravNN_dir + "Data/Dataframes/best_case_model_071123.data"

    # load the dataframe
    df = pd.read_pickle(df_file)

    # load model and config
    config, model = load_config_and_model(df, df.id.values[-1], only_weights=True)

    # run the visualization suite on the model
    vis = ExperimentPanelVisualizer(model, load_exp=False)
    vis.fig_size = (vis.w_full, vis.w_full / 5 * 2)
    vis.plot()

    StatOD_dir = os.path.abspath(os.path.dirname(StatOD.__file__)) + "/../"
    vis.save(plt.gcf(), f"{StatOD_dir}/Plots/panel_plot.pdf")
    plt.savefig(f"{StatOD_dir}/Plots/panel_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
