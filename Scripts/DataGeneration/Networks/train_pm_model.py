import os
from pprint import pprint

from GravNN.GravityModels.PointMass import get_pm_data
from GravNN.Networks.Configs import *

import StatOD

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    df_file = statOD_dir + "Data/Dataframes/eros_pm_061023.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [50000],
        "N_train": [4500],
        "N_val": [5000],
        "num_units": [20],
        "loss_fcns": [["percent"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "batch_size": [2**16],
        "epochs": [10],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "gravity_data_fcn": [get_pm_data],
        # "fuse_models": [False],
        # "enforce_bc": [False],
        # "scale_nn_potential": [False],
    }
    config.update(hparams)
    run(config, df_file)


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
    if config["epochs"][0] == 0:
        model.predict(data.train_data)
        history = None
    else:
        history = model.train(data)

    saver = ModelSaver(
        model,
        history=history,
    )
    saver.save(df_file=df_file)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
