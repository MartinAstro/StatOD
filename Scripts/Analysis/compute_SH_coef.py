import os
import pickle

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import format_coefficients, save

import StatOD


def main(regress_deg):
    remove_deg = -1

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"

    with open(f"{statOD_dir}Data/Trajectories/traj_eros_poly_053123.data", "rb") as f:
        data = pickle.load(f)

    # convert to meters
    x = data["X_B"] * 1E3
    y = data["A_B"] * 1E3

    planet = Eros()
    regressor = BLLS(regress_deg, planet, remove_deg)
    results = regressor.update(x, y)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)

    file_name = f"{statOD_dir}Data/Products/SH_Eros_model_{regress_deg}.csv"
    save(file_name, planet, C_lm, S_lm)


if __name__ == "__main__":
    main(2)
    main(4)
    main(8)
    main(16)
