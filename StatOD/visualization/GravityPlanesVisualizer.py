import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

from StatOD.utils import *


class GravityPlanesVisualizer(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        pass

    def run(self, max_error, logger=None):
        plt.rc("text", usetex=False)
        if logger is not None:
            X_traj = logger.x_hat_i_plus[:, 0:3] * 1e3 / Eros().radius
        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        plt.figure()
        self.max = max_error
        self.plot_plane(x, y, plane="xy", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        if logger is not None:
            plt.plot(X_traj[:, 0], X_traj[:, 1], color="black", linewidth=0.5)
        plt.figure()
        self.plot_plane(x, y, plane="xz", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        if logger is not None:
            plt.plot(X_traj[:, 0], X_traj[:, 2], color="black", linewidth=0.5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.figure()
        self.plot_plane(x, y, plane="yz", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        if logger is not None:
            plt.plot(X_traj[:, 1], X_traj[:, 2], color="black", linewidth=0.5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    def save_all(self, network_directory):
        figure_nums = plt.get_fignums()
        for i in range(-3, 0):
            fig = plt.figure(figure_nums[i])
            super().save(fig, network_directory + f"/plane_{i}.png")