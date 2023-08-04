import os

import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    generate_heterogeneous_sym_model,
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


def plot_planes(model, config):
    statOD_dir = os.path.dirname(StatOD.__file__)
    plots_dir = f"{statOD_dir}/../Plots/"
    model_name = model.__class__.__name__

    planet = Eros()
    exp = PlanesExperiment(model, config, [-2 * planet.radius, 2 * planet.radius], 50)
    exp.config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]
    exp.run()

    # plot the error of the model
    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.w_quad)  # 3 columns of 4
    vis.plot(z_max=10)
    vis.save(
        plt.gcf(),
        f"{plots_dir}{model_name}_gravity_field_planes_error.pdf",
    )

    # Plot the true gravity field
    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_quad * 3, vis.w_quad)  # 3 columns of 4
    vis.plot_gravity_field(z_max=None)
    vis.save(
        plt.gcf(),
        f"{plots_dir}{model_name}_gravity_field_planes_truth.pdf",
    )


def main():
    """Plots the true gravity field in a plane"""

    planet = Eros()

    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_file = f"{statOD_dir}/../Data/Dataframes/eros_constant_poly.data"

    df = pd.read_pickle(pinn_file)

    config, model = load_config_and_model(
        df,
        df.id.values[-1],
        custom_dtype="float32",
        only_weights=True,
    )

    ############################################
    # Plot the True Model
    ############################################
    model = generate_heterogeneous_sym_model(planet, planet.obj_8k)
    plot_planes(model, config)

    ############################################
    # Plot the Point Mass Model
    ############################################

    model = PointMass(planet)
    plot_planes(model, config)

    ############################################
    # Plot the Spherical Harmonics Model
    ############################################

    regress_deg = 16
    file_name = f"{statOD_dir}/../Data/Products/SH_Eros_model_{regress_deg}.csv"
    model = SphericalHarmonics(file_name, regress_deg)
    plot_planes(model, config)

    ############################################
    # Plot the Polyhedral Model
    ############################################

    model = Polyhedral(planet, planet.obj_8k)
    plot_planes(model, config)


if __name__ == "__main__":
    main()
    plt.show()
