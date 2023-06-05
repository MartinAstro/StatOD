import multiprocessing as mp
import os
from pprint import pprint

import numpy as np
from GravNN.GravityModels.PointMass import get_pm_data
from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():

    threads = 4

    df_file = "Data/Dataframes/eros_filter_poly.data"
    df_file = "Data/Dataframes/eros_filter_poly_dropout.data"
    df_file = "Data/Dataframes/eros_pm_053123.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [4500],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent"]],
        # "loss_fcns": [["percent", "rms"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "batch_size": [2**16],
        "epochs": [10000],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "gravity_data_fcn": [get_pm_data],
        "fuse_models" : [False],
        # "dropout": [0.1],
        # "override": [True],
    }
    args = configure_run_args(config, hparams)

    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):

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
    model.train(data)
    # model.predict(np.array([[0.0, 0.0, 0.0]]))
    saver = ModelSaver(model, history=None)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
