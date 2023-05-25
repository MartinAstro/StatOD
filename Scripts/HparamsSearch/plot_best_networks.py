import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.Networks.Model import load_config_and_model

from Scripts.Scenarios.helper_functions import *
from Scripts.Visualization.GravityPlanesVisualizer import GravityPlanesVisualizer
from StatOD.dynamics import *
from StatOD.visualization.visualizations import *

plt.switch_backend("WebAgg")


def plot(df):
    model_ids = df.id.values
    for model_id in model_ids:
        config, model = load_config_and_model(model_id, df)
        planes_exp = PlanesExperiment(
            model.gravity_model,
            model.config,
            [
                -model.config["planet"][0].radius * 2,
                model.config["planet"][0].radius * 2,
            ],
            100,
        )
        planes_exp.config["gravity_data_fcn"] = [get_hetero_poly_data]
        planes_exp.run()

        # Plot experiment results
        planes_vis = GravityPlanesVisualizer(planes_exp, halt_formatting=True)
        planes_vis.run(
            max_error=10,
        )

        bs = config["batch_size"][0]
        epochs = config["epochs"][0]
        frac = config["data_fraction"][0]

        title = f"{bs}_{epochs}_{frac}"
        plt.title(title)

    plt.show()

    return


if __name__ == "__main__":
    directory = os.path.dirname(StatOD.__file__)

    df = pd.read_pickle(
        directory + "/../Data/Dataframes/hparam_test.data",
    )

    # filter out only the top 10
    query = "hparams_percent_error_avg < 10"
    df = df.query(query)
    plot(df)
