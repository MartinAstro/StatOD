import numpy as np
import matplotlib.pyplot as plt
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from StatOD.visualizations import VisualizationBase
from helper_functions import * 
from GravNN.GravityModels.Polyhedral import get_poly_data

class AnalysisBaseClass:
    def __init__(self, scenario):
        self.scenario = scenario
    
    def generate_y_hat(self):
        t = self.scenario.t
        logger = self.scenario.filter.logger
        h_args_vec = self.scenario.h_args
        filter = self.scenario.filter

        y_hat_vec = np.zeros((len(t), 3))
        for i in range(len(t)):

            if self.scenario.filter_type == "KalmanFilter":
                y_hat_vec[i] = filter.predict_measurement(logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i])
            elif self.scenario.filter_type == "UnscentedKalmanFilter":
                # doesn't include dx argument
                y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i], h_args_vec[i])
            else:
                y_hat_vec[i] = filter.predict_measurement(logger.x_hat_i_plus[i], np.zeros_like(logger.x_hat_i_plus[i]), h_args_vec[i])
        
        self.y_hat = y_hat_vec


    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$x$', r'$y$', r"$z$"])


    def generate_filter_plots(self, x_truth, w_truth, y_labels=None):
        vis = VisualizationBase(self.scenario.filter.logger, None, False)
        plt.rc('text', usetex=False)
        vis.plot_state_errors(x_truth)
        vis.plot_residuals(self.scenario.Y,
                        self.y_hat, 
                        self.scenario.R, 
                        y_labels)
        vis.plot_vlines(self.scenario.train_idx_list)

        # Plot the DMC values 
        plot_DMC(self.scenario.filter.logger, w_truth)

    def generate_gravity_plots(self):
        model = self.scenario.model
        planes_exp = PlanesExperiment(model.gravity_model, model.config, [-model.config['planet'][0].radius*4, model.config['planet'][0].radius*4], 50)
        planes_exp.config['gravity_data_fcn'] = [get_poly_data]
        planes_exp.run()
        mask = planes_exp.get_planet_mask()
        planes_exp.percent_error_acc[mask] = np.nan
        print(f"Error Percent Average: {np.nanmean(planes_exp.percent_error_acc)}")

        plot_error_planes(planes_exp, max_error=20, logger=self.scenario.filter.logger)
    