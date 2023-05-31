from abc import abstractclassmethod

from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.TrajectoryExperiment import TrajectoryExperiment


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

        return exp


class ExtrapolationCallback(CallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model):
        exp = ExtrapolationExperiment(model, model.config, points=1000)
        exp.run()
        return exp


class TrajectoryCallback(CallbackBase):
    def __init__(self, truth_model, X0, T, **kwargs):
        super().__init__(**kwargs)
        self.truth_model = truth_model
        self.X0 = X0
        self.T = T

    def run(self, model):
        exp = TrajectoryExperiment(self.truth_model, self.X0, self.T)
        exp.add_test_model(model, "PINN", "red")
        exp.run()
        return exp
