from GravNN.Trajectories import DHGridDist, RandomDist
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
import time,pickle, os
import numpy as np
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, save


def main():
    regress_deg = 2
    remove_deg = -1

    with open(f'Data/Trajectories/traj_rotating.data', 'rb') as f:
        data = pickle.load(f)

    # convert to meters
    x = data['X_B'] #* 1E3
    y = data['A_B'] #* 1E3

    planet = Eros()
    regressor = BLLS(regress_deg, planet, remove_deg)
    results = regressor.update(x, y)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)

    save('Data/Products/SH_Eros_model.csv', planet, C_lm, S_lm)

if __name__ == "__main__":
    main()
