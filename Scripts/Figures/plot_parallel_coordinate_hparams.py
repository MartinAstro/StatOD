import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import StatOD


def main2():
    # full_df = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF_4.data")

    # df1 = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF.data")
    # df2 = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF_2.data")
    # df3 = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF_3.data")
    # df4 = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF_4.data")

    # full_df = pd.concat((df1, df2, df3, df4))
    # full_df = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/df_fixed_IAC.data")

    full_df = pd.read_pickle(
        os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/hparams_rotating.data",
    )

    hparams_df = pd.DataFrame()
    i = 0
    for i in range(len(full_df)):
        hparams = full_df.iloc[i]["hparams"]
        hparams["results"] = np.nanmean(hparams["results"])
        # hparams['results'] = np.nanmedian(hparams['results'])

        df_row = pd.DataFrame(hparams, index=[i])
        hparams_df = pd.concat((hparams_df, df_row))
        i += 1

    print(
        f"Idx of lowest result {np.where(hparams['results'] == hparams['results'].min())}",
    )

    mask = hparams_df["train_fcn"] == "pinn_a"
    hparams_df.loc[mask, "train_fcn"] = 1
    hparams_df.loc[~mask, "train_fcn"] = 2
    hparams_df.loc[:, "train_fcn"] = hparams_df.loc[:, "train_fcn"].astype(float)
    dimensions = []

    artificial_max = 10

    # results_mask = hparams_df['results'] < 8
    # hparams_df = hparams_df[results_mask]

    labels_dict = {
        "results": r"Average % Error",
        "q_value": "q",
        "epochs": "Epochs",
        "batch_size": "Batch Size",
        "train_fcn": "PINN Constraint",
        "learning_rate": "Learning Rate",
        "tau": "Tau",
        "boundary_condition_data": "BC Data",
    }

    for column in hparams_df.columns:
        log_scale = False
        prefix = ""
        values = hparams_df[column].values

        log_diff = np.log10(np.max(values)) - np.log10(np.min(values))

        # if the data spans more than 1.5 orders of magnitude
        # make it log scale
        if log_diff > 1.5 and column != "results":
            values = np.log10(values)
            prefix = "log10 "
            log_scale = True

        # gather the unique tick values
        tick_values = np.round(np.unique(values), 2)

        # introduce some variability into the values to enhance
        # plot visibility
        if column != "results":
            for i in range(len(values)):
                value = values[i]
                if log_scale:
                    # random must accept positive value in std (- log values don't work)
                    values[i] += np.random.normal(0, np.abs(value * 0.01))
                else:
                    # make the variability the same on a linear plot
                    values[i] += np.random.normal(0, np.max(values) * 0.01)

        # update the label to be prettier
        column = labels_dict[column]
        dimension_dict = {
            "label": prefix + column,
            "values": values,
            "tickvals": tick_values,
        }
        if column == "PINN Constraint":

            dimension_dict = {
                "label": column,
                "values": values,
                "tickvals": [1, 2],
                "ticktext": ["PINN A", "PINN ALC"],
            }
        if column == "BC Data":
            values = np.nan_to_num(values, neginf=1)
            dimension_dict = {
                "label": column,
                "values": values,
                "tickvals": [0, 1],
                "ticktext": ["False", "True"],
            }
        if column == r"Average % Error":
            min_result = np.round(np.min(values), 3)
            np.round(np.max(values), 3)

            dimension_dict = {
                # 'range' : [np.min(values), np.max(values)],
                "range": [np.min(values), artificial_max],
                "label": column,
                "values": values,
                "tickvals": np.round(np.linspace(min_result, artificial_max, 8), 2),
            }
        dimensions.append(dimension_dict)

    # Log projection : https://stackoverflow.com/questions/48421782/plotly-parallel-coordinates-plot-axis-styling

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=hparams_df["results"],
                colorscale=px.colors.diverging.Tealrose,
                cmid=7.6,
                cmax=artificial_max,
                cmin=hparams_df["results"].min(),
            ),
            dimensions=dimensions,
        ),
    )

    DPI_factor = 2
    DPI = 100  # standard DPI for matplotlib
    fig.update_layout(
        # autosize=True,
        height=2.7 * DPI * DPI_factor,
        width=6.5 * DPI * DPI_factor,
        template="none",
        font={
            "family": "serif",
            "size": 10 * DPI_factor,
        },
    )

    os.path.dirname(StatOD.__file__) + "/../Plots/"
    # write_image(fig, figure_path + "hparams.pdf", format='pdf', width=6.5*DPI*DPI_factor, height=3*DPI*DPI_factor)

    fig.show()


if __name__ == "__main__":
    # main()
    main2()
