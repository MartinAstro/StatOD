import os

import pandas as pd

import StatOD


def main():
    # load in metrics dataframe
    statOD_dir = os.path.dirname(StatOD.__file__)
    metrics_file = f"{statOD_dir}/../Data/Dataframes/standard_model_metrics.data"
    metrics_df = pd.read_pickle(metrics_file)

    # replace values in percent_error_avg, percent_error_std, and percent_error_max with
    # nans if they are greater than 1000
    # metrics_df.loc[metrics_df["percent_error_avg"] > 1000, "percent_error_avg"] = np.nan
    # metrics_df.loc[metrics_df["percent_error_std"] > 1000, "percent_error_std"] = np.nan
    # metrics_df.loc[metrics_df["percent_error_max"] > 1000, "percent_error_max"] = np.nan

    # Format columns to scientific notation
    metrics_df["dX_sum"] = metrics_df["dX_sum"].map("{:.1f}".format)
    metrics_df["percent_error_avg"] = metrics_df["percent_error_avg"].map(
        "{:.1f}".format,
    )
    metrics_df["percent_error_std"] = metrics_df["percent_error_std"].map(
        "{:.1f}".format,
    )
    metrics_df["percent_error_max"] = metrics_df["percent_error_max"].map(
        "{:.1f}".format,
    )

    # Make a single column for the planes
    metrics_df["new_col"] = metrics_df.apply(
        lambda row: f"{row['percent_error_avg']}",  # +/- {row['percent_error_std']} ({row['percent_error_max']})",
        axis=1,
    )

    # Separate the camel case word in the Model column
    metrics_df["Model"] = metrics_df["Model"].str.replace(r"(?<!^)(?=[A-Z])", " ")

    # Update the column names
    metrics_df = metrics_df.rename(
        columns={
            "inter_avg": "Interpolation %",
            "extra_avg": "Extrapolation %",
            "dX_sum": "d$X$ Sum",
            "t_sum": "Time [s]",
            "new_col": "Planes %",
        },
    )

    # drop the columns we don't want
    metrics_df = metrics_df.drop(
        columns=[
            "percent_error_std",
            "percent_error_max",
            "high_error_pixel",
        ],
    )

    # make Model the index
    metrics_df = metrics_df.set_index("Model")

    # Reorder the columns
    metrics_df = metrics_df.reindex(
        columns=[
            "Planes %",
            "Interpolation %",
            "Extrapolation %",
            "d$X$ Sum",
            "Time [s]",
        ],
    )

    # Make a latex table
    latex_table = metrics_df.to_latex(
        index=True,
        float_format="%.2f",
        column_format="l" + "c" * (len(metrics_df.columns) - 1),
        escape=False,
    )

    print(latex_table)


if __name__ == "__main__":
    main()
