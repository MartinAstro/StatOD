import os

import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    # generate_heterogeneous_sym_model,
    get_hetero_poly_symmetric_data,
)
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Model import load_config_and_model

import StatOD
from Scripts.Factories.CallbackFactory import CallbackFactory


def compute_metrics(model):
    callbacks = CallbackFactory().generate_callbacks(pbar=True, radius_multiplier=2)

    # run the callbacks
    t_i_dummy = 0.0  # dummy
    for callback in callbacks.values():
        callback(model, t_i_dummy)

    # extract the data from the callbacks
    metrics = {}
    for name, callback in callbacks.items():
        metrics[name] = callback.data[-1]

    # compute the metrics for the table
    all_metrics = {}

    dX_sum = 0
    t_sum = 0
    traj_count = 0
    for key, value in metrics["Trajectory"].items():
        if "dX_sum" in key:
            dX_sum += value
            traj_count += 1
        elif "t_" in key:
            t_sum += value

    all_metrics.update(metrics["Extrapolation"])
    all_metrics.update(metrics["Planes"])
    all_metrics.update({"dX_sum": dX_sum, "t_sum": t_sum})
    all_metrics.update({"Model": model.__class__.__name__})

    # return the metrics
    return all_metrics


def main():
    metrics_list = []
    planet = Eros()

    statOD_dir = os.path.dirname(StatOD.__file__)

    ############################################
    # Plot the Best Case PINN Model
    ############################################

    pinn_file = f"{statOD_dir}/../Data/Dataframes/best_case_model_071123.data"
    df = pd.read_pickle(pinn_file)
    config, model = load_config_and_model(
        df,
        df.id.values[-1],
        custom_dtype="float32",
        only_weights=True,
    )
    config.update(
        {
            "gravity_data_fcn": [get_hetero_poly_symmetric_data],
        },
    )

    model.config = config
    metrics = compute_metrics(model)
    metrics_list.append(metrics)

    ############################################
    # Plot the Worst Case PINN Model
    ############################################

    pinn_file = f"{statOD_dir}/../Data/Dataframes/worst_case_model_071123.data"
    df = pd.read_pickle(pinn_file)
    config, model = load_config_and_model(
        df,
        df.id.values[-1],
        custom_dtype="float32",
        only_weights=True,
    )
    config.update(
        {
            "gravity_data_fcn": [get_hetero_poly_symmetric_data],
        },
    )

    model.config = config
    metrics = compute_metrics(model)
    metrics_list.append(metrics)

    ############################################
    # Plot the True Model
    ############################################
    # model = generate_heterogeneous_sym_model(planet, planet.obj_8k)
    # model.config = config
    # metrics = compute_metrics(model)
    # metrics_list.append(metrics)

    ############################################
    # Plot the Point Mass Model
    ############################################

    model = PointMass(planet)
    model.config = config
    metrics = compute_metrics(model)
    metrics_list.append(metrics)

    ############################################
    # Plot the Spherical Harmonics Model
    ############################################

    regress_deg = 16
    file_name = f"{statOD_dir}/../Data/Products/SH_Eros_model_{regress_deg}.csv"
    model = SphericalHarmonics(file_name, regress_deg)
    model.config = config
    metrics = compute_metrics(model)
    metrics_list.append(metrics)

    ############################################
    # Plot the Polyhedral Model
    ############################################

    model = Polyhedral(planet, planet.obj_8k)
    model.config = config
    metrics = compute_metrics(model)
    metrics_list.append(metrics)

    # convert the metrics into a dataframe and save to file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_pickle(f"{statOD_dir}/../Data/Dataframes/standard_model_metrics.data")


if __name__ == "__main__":
    main()
