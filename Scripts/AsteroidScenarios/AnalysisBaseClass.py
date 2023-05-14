import matplotlib.pyplot as plt
import numpy as np
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data

from Scripts.AsteroidScenarios.helper_functions import *
from StatOD.visualizations import VisualizationBase


class AnalysisBaseClass:
    def __init__(self, scenario):
        self.scenario = scenario
        self.planes_exp = None
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
        plot_DMC(self.scenario.filter.logger, w_truth)

    def generate_gravity_plots(self):
        if self.planes_exp is not None:
            plot_error_planes(
                self.planes_exp,
                max_error=10,
                logger=self.scenario.filter.logger,
            )
        else:
            self.run_planes_experiment()
            plot_error_planes(
                self.planes_exp,
                max_error=10,
                logger=self.scenario.filter.logger,
            )

    def run_planes_experiment(self):
        model = self.scenario.model
        planes_exp = PlanesExperiment(
            model.gravity_model,
            model.config,
            [
                -model.config["planet"][0].radius * 4,
                model.config["planet"][0].radius * 4,
            ],
            50,
        )
        planes_exp.config["gravity_data_fcn"] = [self.true_gravity_fcn]
        planes_exp.run()
        mask = planes_exp.get_planet_mask()
        planes_exp.percent_error_acc[mask] = np.nan
        print(f"Error Percent Average: {np.nanmean(planes_exp.percent_error_acc)}")
        print(
            f"Fraction of Pixels with > 10\% Error: {np.count_nonzero(planes_exp.percent_error_acc > 10)/len(mask)}",
        )
        self.planes_exp = planes_exp

        return planes_exp.percent_error_acc
