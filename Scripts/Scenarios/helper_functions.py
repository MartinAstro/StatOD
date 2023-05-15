import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

import StatOD


def get_trajectory_data(data_file):
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + f"Data/Trajectories/{data_file}.data", "rb") as f:
        data = pickle.load(f)
    return data


def format_args(hparams):
    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args = []
    session_num = 0
    for hparam_inst in permutations_dicts:
        print("--- Starting trial: %d" % session_num)
        print({key: value for key, value in hparam_inst.items()})
        session_num += 1
        args.append((hparam_inst,))
    return args


def save_results(df_file, configs):
    import pandas as pd

    for config in configs:
        config = dict(sorted(config.items(), key=lambda kv: kv[0]))
        config["PINN_constraint_fcn"] = [
            config["PINN_constraint_fcn"][0],
        ]  # Can't have multiple args in each list
        df = pd.DataFrame().from_dict(config).set_index("timetag")

        try:
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except:
            df.to_pickle(df_file)


def compute_BN(tVec, omega):
    theta = tVec * omega
    C00 = np.cos(theta)
    C01 = -np.sin(theta)
    C10 = np.sin(theta)
    C11 = np.cos(theta)
    Cij = np.zeros_like(C00)
    C22 = np.zeros_like(C00) + 1

    C = np.block(
        [
            [[C00], [C01], [Cij]],
            [[C10], [C11], [Cij]],
            [[Cij], [Cij], [C22]],
        ],
    )
    C = np.transpose(C, axes=[2, 0, 1])

    return C
