import matplotlib.pyplot as plt
import numpy as np
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data

from Scripts.Scenarios.helper_functions import *
from StatOD.visualizations import VisualizationBase
import pandas as pd


class AnalysisBaseClass:
    def __init__(self, scenario):
        self.scenario = scenario
        self.true_gravity_fcn = get_poly_data

    def run(self):
        self.run_planes_experiment()
        # self.run_extrapolation_experiment()

        metrics = self.gather_metrics()
        return metrics

    def gather_metrics(self):
        metrics = {}
        metrics.update(self.get_planes_metrics())

        return metrics

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
        self.planes_exp = planes_exp

    # def run_extrapolation_experiment(self):
    #    from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment

    #     model = self.scenario.model.gravity_model
    #     config = self.scenario.model.gravity_model.config
    #     points = 1000

    #     exp = ExtrapolationExperiment(model, config, points)
    #     exp.run()

    #     self.extrapolation_exp = exp

    def get_planes_metrics(self):
        mask = self.planes_exp.get_planet_mask()
        self.planes_exp.percent_error_acc[mask] = np.nan
        percent_error_avg = np.nanmean(self.planes_exp.percent_error_acc)
        percent_error_std = np.nanstd(self.planes_exp.percent_error_acc)
        percent_error_max = np.nanmax(self.planes_exp.percent_error_acc)

        high_error_pixel = (
            np.count_nonzero(self.planes_exp.percent_error_acc > 10) / len(mask) * 100
        )

        print(f"Error Percent Average: {np.nanmean(self.planes_exp.percent_error_acc)}")
        print(f"Fraction of Pixels with > 10\% Error: {high_error_pixel}")

        return {
            "percent_error_avg": percent_error_avg,
            "percent_error_std": percent_error_std,
            "percent_error_max": percent_error_max,
            "high_error_pixel": high_error_pixel,
        }

    # def get_extrapolation_metrics(self):
    #     altitudes = self.x_test
    #     percent_error = self.extrapolation_exp.losses["percent"][self.idx_test]
    #     percent_error_avg = np.nanmean(percent_error)
    #     percent_error_std = np.nanmean(percent_error)
    #     return {
    #         "extrap_percent_error_avg": percent_error_avg,
    #     }
