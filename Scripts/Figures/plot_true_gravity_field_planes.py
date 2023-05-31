import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


class TruePlanesVisualizer(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)

    def plot(self, percent_error=False, max=None, **kwargs):
        if percent_error:
            # x/a_test is the truth and x/a_pred is the prediction
            x = self.experiment.x_test

            a_pred = self.experiment.a_pred
            a_test = self.experiment.a_test
            da = a_test - a_pred

            a_test_norm = np.linalg.norm(a_test, axis=1, keepdims=True)
            da_norm = np.linalg.norm(da, axis=1, keepdims=True)
            y = da_norm / a_test_norm * 100

            cmap = cm.jet  # cm.RdYlGn.reversed()

        else:
            # x/a_test is the truth and x/a_pred is the prediction
            x = self.experiment.x_test
            y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)
            cmap = cm.viridis

        if max is None:
            self.max = np.nanmean(y) + 1 * np.nanstd(y)
        else:
            self.max = max

        plt.figure()
        plt.subplot(1, 3, 1)
        self.plot_plane(x, y, plane="xy", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(1, 3, 2)
        self.plot_plane(x, y, plane="xz", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(1, 3, 3)
        self.plot_plane(x, y, plane="yz", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")


def main():
    """Plots the true gravity field in a plane"""

    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"

    df = pd.read_pickle(pinn_file)

    config, model = load_config_and_model(
        df.id.values[-1],
        df,
        custom_dtype="float32",
        only_weights=True,
    )

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"

    ###########################################
    # True Acceleration of Heterogeneous Poly #
    ###########################################

    planet = Eros()
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot()
    vis.save(plt.gcf(), f"{statOD_dir}true_gravity_field_planes.pdf")

    #####################################
    # Percent Error of Homogeneous Poly #
    #####################################

    model = Polyhedral(planet, planet.obj_8k)
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot(percent_error=True, max=30)
    vis.save(plt.gcf(), f"{statOD_dir}poly_gravity_field_planes.pdf")

    ###############################
    # Percent Error of Point Mass #
    ###############################

    model = PointMass(planet)
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot(percent_error=True, max=30)
    vis.save(plt.gcf(), f"{statOD_dir}pm_gravity_field_planes.pdf")

    plt.show()


if __name__ == "__main__":
    main()
