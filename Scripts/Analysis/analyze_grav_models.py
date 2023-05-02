import os

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.PlanesDist import PlanesDist

import StatOD
from StatOD.models import pinnGravityModel


def main():

    planet = Eros()
    trajectory = PlanesDist(planet, [-planet.radius, planet.radius], 10)

    # load true gravity model
    true_gravity_model = Polyhedral(planet, planet.obj_8k, trajectory=trajectory)
    true_gravity_model.load()

    # load the learned gravity model
    data_dir = os.path.dirname(StatOD.__file__) + "/../Data/"
    gravity_model = pinnGravityModel("trained_networks.data", custom_data_dir=data_dir)

    acc_true = true_gravity_model.accelerations
    acc_pred = gravity_model.compute_acceleration(trajectory.positions)

    np.linalg.norm(acc_pred - acc_true, axis=1) / np.linalg.norm(acc_true) * 100


if __name__ == "__main__":
    main()
