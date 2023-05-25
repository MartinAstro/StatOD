import matplotlib.pyplot as plt
import numpy as np
from GravNN.GravityModels.Polyhedral import get_poly_data

from Scripts.Scenarios.helper_functions import *
from StatOD.visualization.visualizations import VisualizationBase


class FilterVisualizer:
    def __init__(self, scenario):
        self.scenario = scenario
        self.true_gravity_fcn = get_poly_data

    def generate_y_hat(self):
        t = self.scenario.t
        logger = self.scenario.filter.logger
        h_args_vec = self.scenario.h_args
        filter = self.scenario.filter

        y_hat_vec = np.zeros((len(t), len(self.scenario.Y[0])))
        for i in range(len(t)):
            if self.scenario.filter_type == "KalmanFilter":
                y_hat_vec[i] = filter.predict_measurement(
                    logger.x_i[i],
                    logger.dx_i_plus[i],
                    h_args_vec[i],
                )
            elif self.scenario.filter_type == "UnscentedKalmanFilter":
                # doesn't include dx argument
                y_hat_vec[i] = filter.predict_measurement(
                    logger.x_hat_i_plus[i],
                    h_args_vec[i],
                )
            else:
                y_hat_vec[i] = filter.predict_measurement(
                    logger.x_hat_i_plus[i],
                    np.zeros_like(logger.x_hat_i_plus[i]),
                    h_args_vec[i],
                )

        self.y_hat = y_hat_vec

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r"$x$", r"$y$", r"$z$"])

    def generate_filter_plots(self, x_truth, w_truth, y_labels=None):
        vis = VisualizationBase(self.scenario.filter.logger, None, False)
        plt.rc("text", usetex=False)
        vis.plot_state_errors(x_truth)
        vis.plot_residuals(self.scenario.Y, self.y_hat, self.scenario.R, y_labels)
        vis.plot_vlines(self.scenario.train_idx_list)

        # Plot the DMC values
        self.plot_DMC(self.scenario.filter.logger, w_truth)

    def plot_DMC(self, logger, w_truth):
        idx_max = len(logger.t_i) if len(logger.t_i) < len(w_truth) else len(w_truth)

        plt.figure()
        plt.subplot(311)
        self.plot_DMC_subplot(
            logger.t_i[:idx_max],
            logger.x_hat_i_plus[:idx_max, 6],
            w_truth[:idx_max, 0],
        )
        plt.subplot(312)
        self.plot_DMC_subplot(
            logger.t_i[:idx_max],
            logger.x_hat_i_plus[:idx_max, 7],
            w_truth[:idx_max, 1],
        )
        plt.subplot(313)
        self.plot_DMC_subplot(
            logger.t_i[:idx_max],
            logger.x_hat_i_plus[:idx_max, 8],
            w_truth[:idx_max, 2],
        )

        # Plot magnitude
        DMC_mag = np.linalg.norm(logger.x_hat_i_plus[:, 6:9], axis=1)
        plt.figure()
        plt.plot(DMC_mag)
        print(f"Average DMC Mag {np.mean(DMC_mag)}")

    def plot_DMC_subplot(self, x, y1, y2):
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
