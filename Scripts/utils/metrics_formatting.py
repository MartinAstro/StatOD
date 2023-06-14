
def make_trajectory_metric(df):
    # Take the trajectory column, and only select
    # the keys with dX_sum. Then average across those
    # scalar values to compute the final metric for the trajectory
    df = df.copy()
    for i in range(len(df)):
        row = df.iloc[i]
        dX_sum = 0
        traj_count = 0
        for key, value in row["Trajectory"].items():
            if "dX_sum" in key:
                dX_sum += value
                traj_count += 1
        df.at[row.name, "Trajectory"] = {"avg_dX": dX_sum / traj_count}
    return df


def metrics_to_columns(df):
    # Take the dictionaries in Planes, Extrapolation, and Trajectory columns
    # and make them into their own columns
    df = df.copy()
    for i in range(len(df)):
        row = df.iloc[i]
        for column in ["Planes", "Extrapolation", "Trajectory"]:
            for key, value in row[column].items():
                df.loc[row.name, f"{column}_{key}"] = value
    df.drop(columns=["Planes", "Extrapolation", "Trajectory"], inplace=True)
    return df


def hparams_to_columns(df):
    # take hparams dictionary and append hparam_ to each key
    # and save as a column in the dataframe
    df = df.copy()
    for i in range(len(df)):
        row = df.iloc[i]
        for key, value in row["hparams"].items():
            df.loc[row.name, f"hparams_{key}"] = value[0]
    df.drop(columns=["hparams"], inplace=True)
    return df
