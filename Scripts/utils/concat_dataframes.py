import glob
import os

import GravNN
import pandas as pd

import StatOD


def gather_config_paths(min_idx, max_idx):
    directory = os.path.dirname(GravNN.__file__)
    files = glob.glob(directory + "/../Data/Networks/*")
    files.sort(key=os.path.getmtime, reverse=True)
    config_paths_paths = files[min_idx:max_idx]
    return config_paths_paths


def make_df_from_configs(min_idx, max_idx):
    config_paths = gather_config_paths(min_idx, max_idx)
    df = pd.DataFrame()
    for file_name in config_paths:
        df_i = pd.read_pickle(file_name + "/config.data")
        df = df.append(df_i)
    return df


def concat_dfs(df1, df2):
    df_new = pd.concat([df1, df2])
    return df_new


def main():
    df_name = "hparam_060523"
    min_idx = 0
    max_idx = 1295

    df = make_df_from_configs(min_idx, max_idx)
    df.to_pickle(
        os.path.dirname(StatOD.__file__) + f"/../Data/Dataframes/{df_name}.data",
    )

    # df10 = pd.read_pickle("Data/Dataframes/epochs_N_search_10_metrics.data")
    # df20 = pd.read_pickle("Data/Dataframes/epochs_N_search_20_metrics.data")
    # df40 = pd.read_pickle("Data/Dataframes/epochs_N_search_40_metrics.data")
    # df80 = pd.read_pickle("Data/Dataframes/epochs_N_search_80_metrics.data")
    # all_df = pd.concat([df10, df20, df40, df80])
    # all_df.to_pickle(os.path.dirname(GravNN.__file__)+f'/../Data/Dataframes/epochs_N_search_all_metrics.data')


if __name__ == "__main__":
    main()
