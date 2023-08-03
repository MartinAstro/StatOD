import os

import GravNN
import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Networks.Model import load_config_and_model

from Scripts.VisualizationTools.ExperimentPanelVisualizer import (
    ExperimentPanelVisualizer,
)


def main():
    # load the output dataframe

    gravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__)) + "/../"
    df_file = gravNN_dir + "Data/Dataframes/output_filter_060123.data"

    # load the dataframe
    df = pd.read_pickle(df_file)

    # load model and config
    config, model = load_config_and_model(df, df.id.values[-1], only_weights=True)

    # run the visualization suite on the model
    vis = ExperimentPanelVisualizer(model)
    vis.plot()
    plt.show()


if __name__ == "__main__":
    main()
