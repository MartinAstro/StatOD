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


def run_exp(planet, model, config):
    exp = PlanesExperiment(
        model,
        config,
        [-2 * planet.radius, 2 * planet.radius],
        100,
        omit_train_data=True,
    )
    exp.load_model_data(model)
    exp.run()
    model_name = model.__class__.__name__
    return exp, model_name


def main():
    """Plots the true gravity field in a plane"""

    statOD_dir = os.path.dirname(StatOD.__file__)
    pinn_file = f"{statOD_dir}/../Data/Dataframes/best_case_model_071123.data"

    df = pd.read_pickle(pinn_file)

    config, model = load_config_and_model(
        df,
        df.id.values[-1],
        custom_dtype="float32",
        only_weights=True,
    )
    config["grav_file"] = [Eros().obj_8k]
    config["gravity_data_fcn"] = [get_hetero_poly_symmetric_data]

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"
    planet = Eros()

    ###########################################
    # True Acceleration of Heterogeneous Poly #
    ###########################################
    model = generate_heterogeneous_sym_model(planet, planet.obj_8k)
    exp, model_name = run_exp(planet, model, config)

    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_full, vis.w_full / 3 * 1.2)  # 3 columns of 4
    vis.plot_gravity_field(log=True)
    vis.save(plt.gcf(), f"{statOD_dir}{model_name}_gravity_field_planes_truth.pdf")

    ###############################
    # Percent Error of Point Mass #
    ###############################

    model = PointMass(planet)
    exp, model_name = run_exp(planet, model, config)

    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_full, vis.w_full / 3 * 1.0)  # 3 columns of 4
    vis.plot(z_min=0, z_max=10, log=False, cbar=False)
    vis.save(plt.gcf(), f"{statOD_dir}{model_name}_gravity_field_planes_error.pdf")

    ##########################
    # Percent Error of SH 16 #
    ##########################

    regress_deg = 16
    file_name = f"{statOD_dir}/../Data/Products/SH_Eros_model_{regress_deg}.csv"
    model = SphericalHarmonics(file_name, regress_deg)
    exp, model_name = run_exp(planet, model, config)

    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_full, vis.w_full / 3 * 1.0)  # 3 columns of 4
    vis.plot(z_min=0, z_max=10, log=False, cbar=False)
    vis.save(plt.gcf(), f"{statOD_dir}{model_name}_gravity_field_planes_error.pdf")

    # ####################################
    # # Percent Error of Homogeneous Poly #
    # ####################################

    model = Polyhedral(planet, planet.obj_8k)
    exp, model_name = run_exp(planet, model, config)

    vis = PlanesVisualizer(exp)
    vis.fig_size = (vis.w_full, vis.w_full / 3 * 1.2)  # 3 columns of 4
    vis.plot(z_min=0, z_max=10, log=False)
    vis.save(plt.gcf(), f"{statOD_dir}{model_name}_gravity_field_planes_error.pdf")

    plt.show()


if __name__ == "__main__":
    main()
