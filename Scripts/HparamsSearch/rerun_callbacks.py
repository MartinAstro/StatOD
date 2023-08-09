import os

import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    get_hetero_poly_symmetric_data,
)

from GravNN.Networks.Model import load_config_and_model

import StatOD
from Scripts.Factories.CallbackFactory import CallbackFactory


def compute_metrics(model):
    callbacks = CallbackFactory().generate_callbacks(pbar=True, radius_multiplier=2)
    callbacks.pop("Trajectory")
    callbacks.pop("Planes")

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

    all_metrics.update(metrics["Extrapolation"])

    return all_metrics


def main():
    # Load the dataframe
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    df = pd.read_pickle(f"{statOD_dir}Data/Dataframes/eros_statOD_poly_080823.data")

    for i, row in enumerate(df):
        # load a model associated with each row
        config, model = load_config_and_model(df, idx=i)

        # Make sure the correct truth is getting used
        model.config.update(
            {
                "gravity_data_fcn": [get_hetero_poly_symmetric_data],
                "grav_file": [Eros().obj_8k],
            },
        )

        # rerun the extrapolation experiment
        metrics = compute_metrics(model)

        # Make a new metrics dataframe that will be used to update the original
        metrics_df = pd.DataFrame(metrics, index=[i])

        # udpate the original dataframe with the metrics dataframe
        for col in metrics_df.columns:
            df.loc[col].iloc[i].update(metrics_df[col])

        # assert that the original dataframe now has the new value for the column
        assert df.loc[col].iloc[i] == metrics_df[col].iloc[0]

    df.to_pickle(f"{statOD_dir}Data/Dataframes/eros_statOD_poly_080823_corrected.data")

    df


if __name__ == "__main__":
    main()
