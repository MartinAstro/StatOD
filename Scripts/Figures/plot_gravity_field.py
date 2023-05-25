import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


class TruePlanesVisualizer(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)

    def plot(self, **kwargs):
        plt.figure()
        plt.subplot(2, 2, 1)
        self.plot_density_map(self.experiment.x_train)

        x = self.experiment.x_test
        y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)
        self.max = np.nanmean(y) + 1 * np.nanstd(y)

        cbar_label = "Gravity Field"
        plt.subplot(2, 2, 2)
        self.plot_plane(x, y, plane="xy", colorbar_label=cbar_label, **kwargs)
        plt.subplot(2, 2, 3)
        self.plot_plane(x, y, plane="xz", colorbar_label=cbar_label, **kwargs)
        plt.subplot(2, 2, 4)
        self.plot_plane(x, y, plane="yz", colorbar_label=cbar_label, **kwargs)
        plt.tight_layout()


def main():
    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"

    df = pd.read_pickle(pinn_file)

    config, model = load_config_and_model(
        df.id.values[-1],
        df,
        custom_dtype="float32",
        only_weights=True,
    )

    planet = Eros()
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_data]
    exp.run()
    vis = TruePlanesVisualizer(exp)
    vis.plot()

    plt.show()


if __name__ == "__main__":
    main()
