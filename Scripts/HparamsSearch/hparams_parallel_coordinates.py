import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sigfig
from plotly.io import write_html, write_image

import StatOD


class ParallelCoordinatePlot:
    def __init__(self, df, metric="Percent mean", metric_max=None, metric_min=None):
        self.df = df
        self.metric = metric
        self.metric_data = self.df[self.metric]
        self.linear_columns = [
            "Interpolation",
            "Extrapolation",
            "Trajectory",
            "Percent mean",
        ]
        self.metric_max = metric_max
        self.metric_min = metric_min
        if metric_max is None:
            self.metric_max = self.metric_data.mean() + self.metric_data.std() * 2
        if metric_min is None:
            self.metric_min = self.metric_data.min()

    def run(self):
        metric_ticks = []
        for value in np.linspace(self.metric_min, self.metric_max, 8):
            value_rounded = sigfig.round(value, sigfigs=3)
            metric_ticks.append(value_rounded)

        labels_dict = {
            self.metric: {
                "range": [self.metric_min, self.metric_max],
                "tickvals": metric_ticks,
            },
        }

        dimensions = []
        for column in self.df.columns:
            values, prefix, tick_values, unique_strings = self.scale_data(
                self.df,
                column,
            )

            # update the label to be prettier
            column_dict = labels_dict.get(column, {})
            dimension_dict = {
                "label": prefix + column,
                "values": values,
                "tickvals": column_dict.get("tickvals", tick_values),
                "ticktext": unique_strings,
                "range": column_dict.get("range", None),
            }

            dimensions.append(dimension_dict)

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=self.metric_data,
                    colorscale=px.colors.diverging.Tealrose,
                    cmax=self.metric_max,
                    cmin=self.metric_min,
                ),
                dimensions=dimensions,
            ),
        )
        return fig

    def concat_strings(self, values):
        new_list = []
        for value in values:
            new_list.append(["".join([f"{s}_" for s in value])[:-1]])
        return np.array(new_list).squeeze()

    def make_column_numeric(self, df, column):
        # try converting object column to float
        try:
            df = df.astype({column: float})
            unique_strings = None

        except Exception:
            # if string, remove spaces + encode
            str_values = df[column].values
            str_values_concat = self.concat_strings(str_values)
            unique_strings = np.unique(str_values_concat)
            df.loc[:, column] = str_values_concat

            # If the column contains strings, make integers
            for i, string in enumerate(unique_strings):
                mask = df[column] == string
                df.loc[mask, column] = i + 1

            # convert type to float
            df = df.astype({column: float})

        return df, unique_strings

    def scale_data(self, df, column):
        df, unique_strings = self.make_column_numeric(df, column)

        max_val = df[column].max()
        min_val = df[column].min()
        log_diff = np.log10(max_val) - np.log10(min_val)
        log_diff = 0.0 if np.isinf(log_diff) else log_diff

        prefix = ""
        values = df[column].values

        # Tick values default to all unique entries
        tick_values = []
        for value in np.unique(values):
            val_rounded = sigfig.round(value, sigfigs=3)
            tick_values.append(val_rounded)

        perturbations = np.zeros_like(values)
        PERT_FRAC = 0.02

        # Convert to log space if min-max delta too large
        MAX_LOG_DIFF = 1.0
        linear_column = column in self.linear_columns
        if log_diff >= MAX_LOG_DIFF and not linear_column:
            values = np.log10(values)
            tick_values = np.log10(tick_values)
            prefix = "log10 "

        # add noise to the data for enhanced visibility
        perturbations = []
        for value in values:
            if value == 0.0:
                value = 0.1
            pert = np.random.normal(0, np.abs(value * PERT_FRAC))
            perturbations.append(pert)

        # unless it's the results
        if self.metric == column:
            perturbations = np.zeros_like(values)

            # tick values can't be each unique entry
            # so divide into 8
            tick_values = []
            for value in np.linspace(min_val, max_val, 8):
                val_rounded = sigfig.round(value, sigfigs=3)
                tick_values.append(val_rounded)

        # clip results to sit within tick bounds
        min_tick = np.min(tick_values)
        max_tick = np.max(tick_values)
        values += np.array(perturbations)
        values = np.clip(values, min_tick, max_tick)
        return values, prefix, tick_values, unique_strings


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


def main(grav_type):
    directory = os.path.dirname(StatOD.__file__)

    df = pd.read_pickle(
        # directory + "/../Data/Dataframes/hparam_search_noiseless_test.data",
        directory + "/../Data/Dataframes/hparam_search_060623_v3.data",
        # directory + "/../Data/Dataframes/hparam_060523.data",
        # + "/../Data/Dataframes/output_filter_060523.data",
    )

    df.dropna(subset="hparams", inplace=True)
    df = make_trajectory_metric(df)
    df = metrics_to_columns(df)
    df = hparams_to_columns(df)

    # filter out only the top 10
    if grav_type == "pm":
        query = "Planes_percent_error_avg < 10 and hparams_pinn_file == 'pm'"
        file_name = "hparams_point_mass"
        metric_max = 10

    if grav_type == "poly":
        query = "Planes_percent_error_avg < 10 and hparams_pinn_file == 'poly'"
        file_name = "hparams_poly"
        metric_max = 10

    df = df.query(query)

    name_dict = {
        # "hparams_q_value": "Process Noise",
        # "hparams_epochs": "Epochs",
        # "hparams_learning_rate": "Learning Rate",
        # "hparams_batch_size": "Batch Size",
        # "hparams_train_fcn": "Training Fcn",
        # "hparams_data_fraction": "Traj Fraction",
        "hparams_pinn_file": "Gravity Model",
        "hparams_measurement_noise": "Measurement Noise",
        "Planes_percent_error_avg": "Percent mean",
        # "Planes_percent_error_std": "Std Error",
        # "Planes_percent_error_max": "Max Error",
        # "Planes_high_error_pixel": "Frac High Pixel",
        "Extrapolation_inter_avg": "Interpolation",
        "Extrapolation_extra_avg": "Extrapolation",
        "Trajectory_avg_dX": "Trajectory",
    }
    df = df.rename(columns=name_dict)
    hparams_df = df[list(name_dict.values())]

    fig = ParallelCoordinatePlot(hparams_df, metric_max=metric_max).run()

    DPI_factor = 3
    DPI = 100  # standard DPI for matplotlib
    fig.update_layout(
        # autosize=True,
        height=2.7 * DPI * DPI_factor,
        width=6.5 * DPI * DPI_factor,
        template="none",
        font={
            "family": "serif",
            "size": 20,  # *DPI_factor
        },
    )
    directory = os.path.dirname(StatOD.__file__) + "/../Plots/"
    write_image(
        fig,
        directory + f"{file_name}.pdf",
        format="pdf",
        width=6.5 * DPI * DPI_factor,
        height=3 * DPI * DPI_factor,
    )
    write_html(
        fig,
        directory + f"{file_name}.html",
    )

    fig.show()


if __name__ == "__main__":
    main("pm")
    main("poly")
