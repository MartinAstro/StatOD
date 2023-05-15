import matplotlib.pyplot as plt
import numpy as np
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data

from Scripts.Scenarios.helper_functions import *
from StatOD.visualizations import VisualizationBase


class GravityPlanesVisualizer:
    def __init__(self):
        pass

    def run(self, planes_exp, max_error, logger):
        visPlanes = PlanesVisualizer(planes_exp)
        plt.rc("text", usetex=False)
        X_traj = logger.x_hat_i_plus[:, 0:3] * 1e3 / Eros().radius
        x = visPlanes.experiment.x_test
        y = visPlanes.experiment.percent_error_acc
        plt.figure()
        visPlanes.max = max_error
        visPlanes.plot_plane(x, y, plane="xy", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        plt.plot(X_traj[:, 0], X_traj[:, 1], color="black", linewidth=0.5)
        plt.figure()
        visPlanes.plot_plane(x, y, plane="xz", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        plt.plot(X_traj[:, 0], X_traj[:, 2], color="black", linewidth=0.5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.figure()
        visPlanes.plot_plane(x, y, plane="yz", annotate_stats=True)
        plt.sca(plt.gcf().axes[0])
        plt.plot(X_traj[:, 1], X_traj[:, 2], color="black", linewidth=0.5)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
