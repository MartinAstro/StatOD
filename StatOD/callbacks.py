from abc import abstractclassmethod

import numpy as np
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.TrajectoryExperiment import TrajectoryExperiment
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer

from StatOD.utils import (
    compute_semimajor,
)


class CallbackBase:
    def __init__(self, **kwargs):
        self.data = []
        self.t_list = []
        self.t_i = 0

    def __call__(self, model, t_i):
        callback_data = self.run(model)
        self.data.append(callback_data)
        self.t_list.append(self.t_i)
        self.t_i = t_i
        return

    @abstractclassmethod
    def run(self):
        """Perform some assessment and return with a dictionary of metrics"""
        pass

    # def save(self, directory):
    #     for data in self.data:
    #         data.save(directory)


class PlanesCallback(CallbackBase):
    def __init__(self, radius_multiplier=5, **kwargs):
        super().__init__(**kwargs)
        self.radius_multiplier = radius_multiplier

    def run(self, model):
        planet = model.config["planet"][0]
        multiplier = self.radius_multiplier
        bounds = [-multiplier * planet.radius, multiplier * planet.radius]
        exp = PlanesExperiment(model, model.config, bounds, samples_1d=100)
        exp.run()

        # Compute metrics
        metrics = {}
        metrics["percent_error_avg"] = np.nanmean(exp.percent_error_acc)
        metrics["percent_error_std"] = np.nanstd(exp.percent_error_acc)
        metrics["percent_error_max"] = np.nanmax(exp.percent_error_acc)
        metrics["high_error_pixel"] = (
            np.count_nonzero(exp.percent_error_acc > 10)
            / len(exp.percent_error_acc)
            * 100
        )

        return metrics


class ExtrapolationCallback(CallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model):
        exp = ExtrapolationExperiment(model, model.config, points=1000)
        exp.run()

        vis = ExtrapolationVisualizer(exp)
        metrics = {}

        metrics["inter_avg"] = np.nanmean(
            vis.experiment.losses["percent"][vis.idx_test][: vis.max_idx],
        )
        metrics["exter_avg"] = np.nanmean(
            vis.experiment.losses["percent"][vis.idx_test],
        )

        return metrics


class TrajectoryCallback(CallbackBase):
    def __init__(self, truth_model, **kwargs):
        super().__init__(**kwargs)
        self.truth_model = truth_model
        self.X0_list = []
        self.T_list = []
        self.mu = truth_model.planet.mu

    def run(self, model):
        metrics = {}

        for i in range(len(self.X0_list)):
            X0 = self.X0_list[i]
            T = self.T_list[i]
            exp = TrajectoryExperiment(self.truth_model, X0, T)
            exp.add_test_model(model, "PINN", "red")
            exp.run()

            metrics["dX_sum_" + str(i)] = np.linalg.norm(exp.test_models[0]["pos_diff"])
            metrics["t_" + str(i)] = exp.test_models[0]["elapsed_time"]

        return metrics

    def compute_T(self, X):
        a = compute_semimajor(X, self.mu)
        if np.isnan(a):
            a = 100
        n = np.sqrt(self.mu / a**3)
        T = 2 * np.pi / n
        return T

    def add_trajectory(self, X0):
        T = self.compute_T(X0)
        self.X0_list.append(X0)
        self.T_list.append(T)
