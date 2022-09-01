import plotly.graph_objects as go
import pandas as pd
import os
import StatOD
import numpy as np

import plotly.express as px
from plotly.io import write_image


def main():
    full_df = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm_hparams_EKF.data")

    hparams_df = pd.DataFrame()
    i = 0
    for i in range(len(full_df)):
        hparams = full_df.iloc[i]['hparams']
        hparams['results'] = np.nanmean(hparams['results'])
        df_row = pd.DataFrame(hparams, index=[i])
        hparams_df = pd.concat((hparams_df, df_row))
        i += 1

    mask = (hparams_df['train_fcn'] == "pinn_a")
    hparams_df.loc[mask, 'train_fcn'] = 0
    hparams_df.loc[~mask, 'train_fcn'] = 1
    hparams_df.loc[:, 'train_fcn'] = hparams_df.loc[:, 'train_fcn'].astype(float)
    dimensions = []
    for column in hparams_df.columns:

        dimension_dict = {
            'label' : column,
            'values' : pd.unique(hparams_df[column]),
        }
        if column == "train_fcn":

            dimension_dict = {
            'label' : column,
            'values' : pd.unique(hparams_df[column]),
            'tickvals' : [0,1],
            'ticktext' : ['PINN A', 'PINN ALC']
            }
        dimensions.append(dimension_dict)

    for column in hparams_df.columns:
        if (column == 'results') or (column == 'train_fcn'):
            continue 
        values = hparams_df.loc[:, column]
        for i in range(len(values)):
            min_value = np.min(values)
            value = values[i]
            value_mod = value + np.random.normal(0, min_value*0.05) # perturb the value slightly for better plotting
            values[i] = value_mod
        hparams_df.loc[:, column] = values



    fig = px.parallel_coordinates(hparams_df, color="results", 
                                labels={
                                    "results": r"Average % Error",
                                    "q_value": "q", 
                                    "epochs": "Epochs",
                                    "batch_size": "Batch Size", 
                                    "train_fcn" : "PINN Constraint", 
                                    "learning_rate" : "Learning Rate", 
                                    "tau" : "Tau"
                                    },
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                )


    DPI_factor = 2
    DPI = 100 # standard DPI for matplotlib
    fig.update_layout(
        # autosize=True,
        height=3*DPI*DPI_factor,
        width=6.5*DPI*DPI_factor,
        template='none',
        font={
            'family' : 'serif',
            'size' : 10*DPI_factor 
        })

    figure_path = os.path.dirname(StatOD.__file__) + "/../Plots/"
    write_image(fig, figure_path + "hparams.pdf", format='pdf', width=6.5*DPI*DPI_factor, height=3*DPI*DPI_factor)

    fig.show()

if __name__ == "__main__":
    main()