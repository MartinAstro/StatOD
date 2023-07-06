import os

import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Model import load_config_and_model

import StatOD
from Scripts.VisualizationTools.TruePlanesVisualizer import TruePlanesVisualizer


def main(regress_deg):
    """Plots the true gravity field in a plane"""

    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"

    df = pd.read_pickle(pinn_file)

    config, model = load_config_and_model(
        df.id.values[-1],
        df,
        custom_dtype="float32",
        only_weights=True,
    )

    file_name = f"{statOD_dir}/../Data/Products/SH_Eros_model_{regress_deg}.csv"
    model = SphericalHarmonics(file_name, regress_deg)

    planet = Eros()
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot(percent_error=True, max=10, log=False)
    vis.save(
        plt.gcf(),
        f"{statOD_dir}/../Plots/SH{regress_deg}_gravity_field_planes.pdf",
    )


if __name__ == "__main__":
    main(2)
    main(4)
    main(8)
    main(16)
    plt.show()
