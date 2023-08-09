import os
import multiprocessing as mp
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import (
    get_hetero_poly_symmetric_data,
)

from GravNN.Networks.Model import load_config_and_model
import GravNN
import numpy as np
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


def get_metrics(i):
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    try:
        # load a model associated with each row
        config, model = load_config_and_model(
            f"{statOD_dir}Data/Dataframes/hparam_080823.data",
            idx=i,
            only_weights=True,
            custom_dtype="float32",
        )

        # Make sure the correct truth is getting used
        model.config.update(
            {
                "gravity_data_fcn": [get_hetero_poly_symmetric_data],
                "grav_file": [Eros().obj_8k],
            },
        )

        # rerun the extrapolation experiment
        metrics = compute_metrics(model)
        print("Finished: " + str(i))
        return (i, metrics)
    except:
        return (i, None)


def main():
    # Load the dataframe
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    df = pd.read_pickle(f"{statOD_dir}Data/Dataframes/hparam_080823.data")

    # multiprocess
    with mp.Pool(32) as pool:
        results = pool.map(get_metrics, range(0, 4))

    # update the dataframe
    for result in results:
        i, metrics = result

        # get the current row and index
        row = df.iloc[i]
        idx = df.index[i]

        # Skip errors
        if metrics is None:
            print("metrics failed")
            continue
        if pd.isna(row["Extrapolation"]):
            print("No extrapolation")
            continue

        # get past extrapolation data
        row_metrics = row["Extrapolation"]
        for key, value in metrics.items():
            print(f"Old Data: {row_metrics}")
            row_metrics[key] = value

        print(f"New Data: {row_metrics}")

        # Update the old row
        df.at[idx, "Extrapolation"] = row_metrics
        assert df.iloc[i]["Extrapolation"][key] == value

    df.to_pickle(f"{statOD_dir}Data/Dataframes/hparam_080823_corrected.data")


if __name__ == "__main__":
    main()
