import os

import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model

import StatOD
from Scripts.VisualizationTools.TruePlanesVisualizer import TruePlanesVisualizer


def main():
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

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"

    ###########################################
    # True Acceleration of Heterogeneous Poly #
    ###########################################

    planet = Eros()
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot()
    vis.save(plt.gcf(), f"{statOD_dir}true_gravity_field_planes.pdf")

    #####################################
    # Percent Error of Homogeneous Poly #
    #####################################

    model = Polyhedral(planet, planet.obj_8k)
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot(percent_error=True, max=10, log=False)
    vis.save(plt.gcf(), f"{statOD_dir}poly_gravity_field_planes.pdf")

    ###############################
    # Percent Error of Point Mass #
    ###############################

    model = PointMass(planet)
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    vis = TruePlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.h_tri)  # 3 columns of 4
    vis.plot(percent_error=True, max=10, log=False)
    vis.save(plt.gcf(), f"{statOD_dir}pm_gravity_field_planes.pdf")

    plt.show()


if __name__ == "__main__":
    main()
