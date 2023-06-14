import os

import pandas as pd

import StatOD


def main():

    # load in the dataframe
    directory = os.path.dirname(StatOD.__file__)
    df = pd.read_pickle(
        directory + "/../Data/Dataframes/hparam_061423.data",
    )

    # sort the dataframe by hparams_percent_mean_avg
    df = df.sort_values(by=['hparams_percent_error_avg'])

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