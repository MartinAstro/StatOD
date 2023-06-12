import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_sym_model

from StatOD.callbacks import *


class CallbackFactory:
    def __init__(self):
        pass

    def generate_callbacks(self, radius_multiplier=2, pbar=False):
        gravity_model_true = generate_heterogeneous_sym_model(Eros(), Eros().obj_8k)

        planes_callback = PlanesCallback(radius_multiplier=radius_multiplier)
        extrapolation_callback = ExtrapolationCallback()
        traj_callback = TrajectoryCallback(gravity_model_true, pbar=pbar)

        X_1 = np.array(
            [
                -19243.595703125,
                21967.5078125,
                17404.74609375,
                -2.939612865447998,
                -1.1707247495651245,
                -1.7654979228973389,
            ],
        )
        X_2 = np.array(
            [
                -17720.09765625,
                29013.974609375,
                0.0,
                -3.0941531658172607,
                -1.8855023384094238,
                -0.0,
            ],
        )
        X_3 = np.array(
            [
                -22921.6484375,
                4955.83154296875,
                24614.02734375,
                -2.5665197372436523,
                0.5549010038375854,
                -2.496790885925293,
            ],
        )
        X_4 = np.array(
            [
                -45843.296875,
                9911.6630859375,
                49228.0546875,
                -1.8148034811019897,
                0.3923742473125458,
                -1.7654976844787598,
            ],
        )

        traj_callback.add_trajectory(X_1)
        traj_callback.add_trajectory(X_2)
        traj_callback.add_trajectory(X_3)
        traj_callback.add_trajectory(X_4)

        callbacks_dict = {
            "Planes": planes_callback,
            "Extrapolation": extrapolation_callback,
            "Trajectory": traj_callback,
        }
        return callbacks_dict
