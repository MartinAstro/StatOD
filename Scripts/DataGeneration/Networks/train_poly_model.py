import os
from pprint import pprint

from GravNN.Networks.Configs import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
import StatOD

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def plot_planes(model, config):
    planet = config["planet"][0]
    points = 100
    radius_bounds = [-5 * planet.radius, 5 * planet.radius]
    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        points,
        remove_error=True,
    )
    planes_exp.run()
    print(np.nanmean(planes_exp.percent_error_acc))
    print(np.nanmax(planes_exp.percent_error_acc))

    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=100, annotate_stats=True, log=True)
    plt.show()


def main():
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    df_file = statOD_dir + "Data/Dataframes/eros_poly_071123.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [45000],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent", "rms"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "batch_size": [2**14],
        "epochs": [2500],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "gravity_data_fcn": [get_poly_data],
        "fuse_models": [True],
    }
    config.update(hparams)
    run(config, df_file)

    # config, model = load_config_and_model(df_file, config['id'][0])
    # plot_planes(model, config)


def run(config, df_file=None):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)
    saver = ModelSaver(model, history=history)
    saver.save(df_file=df_file)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
