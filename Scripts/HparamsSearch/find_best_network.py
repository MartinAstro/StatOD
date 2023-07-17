import os

import pandas as pd

import StatOD
from Scripts.utils.metrics_formatting import * 

def main():

    # load in the dataframe
    directory = os.path.dirname(StatOD.__file__)
    df = pd.read_pickle(
        directory + "/../Data/Dataframes/hparam_071123.data",
    )
    df.dropna(subset="hparams", inplace=True)
    df = hparams_to_columns(df)
    df = make_trajectory_metric(df)
    df = metrics_to_columns(df)

    # only filter the point mass cases
    query = "Planes_percent_error_avg < 100 and hparams_pinn_file == 'pm'"
    df = df.query(query)



    # sort the dataframe by hparams_percent_mean_avg
    df = df.sort_values(by=['Planes_percent_error_avg'])
    # df = df.sort_values(by=['Trajectory_avg_dX'])

    # for the best model, print the columns that have the "hparams" in the key
    for key in df.columns:
        if "hparams" in key:
            print(key, df.iloc[0][key])

    # make a dictionary of the hparams and print it
    hparams = {}
    for key in df.columns:
        if "hparams" in key:
            hparams[key] = df.iloc[0][key]

    print(hparams)


if __name__ == "__main__":
    main()