import numpy as np
import pandas as pd
from GravNN.Networks.Constraints import get_PI_constraint
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.transformations import cart2sph, invert_projection

from StatOD.utils import dict_values_to_list


class pinnGravityModel:
    def __init__(
        self,
        df_file,
        custom_data_dir="",
        dtype="float32",
        learning_rate=None,
        dim_constants=None,
    ):

        # tf.config.run_functions_eagerly(True)
        df = pd.read_pickle(custom_data_dir + df_file)
        config, gravity_model = load_config_and_model(
            df.id.values[-1],
            df,
            custom_dtype=dtype,
            only_weights=True,
        )

        self.dtype = dtype
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config["planet"][0]
        if learning_rate is not None:
            self.config["learning_rate"][0] = learning_rate
        self.optimizer = gravity_model.optimizer.learning_rate = self.config[
            "learning_rate"
        ][0]
        if dim_constants is None:
            self.dim_constants = {
                "m_star": 1.0,
                "t_star": 1.0,
                "l_star": 1.0,
            }
        else:
            self.dim_constants = dim_constants

        removed_pm = config.get("remove_point_mass", [False])[0]
        deg_removed = config.get("deg_removed", [-1])[0]
        if removed_pm or deg_removed > -1:
            self.removed_pm = True
        else:
            self.removed_pm = False

    def compute_acceleration(self, X):
        X_dim = X * self.dim_constants["l_star"]
        R = np.array(X_dim).reshape((-1, 3)).astype(np.float64)

        # this is currently a_r > 0
        a_model = self.gravity_model.compute_acceleration(R).numpy()

        if not self.removed_pm:
            a_model = a_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            a_pm_sph = np.zeros((len(R), 3))
            a_pm_sph[:, 0] = -self.planet.mu / r**2
            r_sph = cart2sph(R)
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            a_model = (a_pm_xyz + a_model).squeeze()  # a_r < 0

        a_star = self.dim_constants["l_star"] / self.dim_constants["t_star"] ** 2
        a_non_dim = a_model / a_star
        return a_non_dim

    def generate_dadx(self, X):
        X_dim = X * self.dim_constants["l_star"]
        R = np.array(X_dim).reshape((-1, 3)).astype(np.float64)
        dadx_dim = self.gravity_model.compute_dU_dxdx(R).numpy()  # this is also > 0
        dadx_non_dim = dadx_dim / (1.0 / self.dim_constants["t_star"] ** 2)
        return dadx_non_dim.squeeze()

    def compute_potential(self, X):
        X_dim = X * self.dim_constants["l_star"]
        R = np.array(X_dim).reshape((-1, 3)).astype(np.float64)
        U_model = self.gravity_model.compute_potential(R).numpy()  # this is also > 0
        if not self.removed_pm:
            U_model = U_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            U_pm = np.zeros((len(R), 1))
            U_pm[:, 0] = -self.planet.mu / r
            U_model = (U_pm + U_model).squeeze()

        u_star = self.dim_constants["l_star"] ** 2 / self.dim_constants["t_star"] ** 2
        U_model_non_dim = U_model / u_star
        return U_model_non_dim

    def set_PINN_training_fcn(self, PINN_constraint_fcn):
        PINN_variables = get_PI_constraint(PINN_constraint_fcn)
        self.gravity_model.eval = PINN_variables
        self.config["PINN_constraint_fcn"] = [PINN_constraint_fcn]

        # necessary to set or avoid XLA
        self.gravity_model.__init__(self.config, self.gravity_model.network)
        self.gravity_model.compile(optimizer=self.gravity_model.optimizer)

    def train(self, X_dim, Y_dim, **kwargs):

        # save X_dim and Y_dim to a pickle file
        import pickle

        data = {"X": X_dim, "Y": Y_dim}
        with open("Scripts/Scratch/training_data_III_v2.data", "wb") as f:
            pickle.dump(data, f)

        # non-dimensionalize / preprocess (in case different scheme was used)
        X_process = self.gravity_model.x_preprocessor(X_dim).numpy()
        Y_process = self.gravity_model.a_preprocessor(Y_dim).numpy()
        if self.config["PINN_constraint_fcn"][0] == "pinn_al":
            Y_L = np.full((len(Y_process), 1), 0.0)
            Y_process = np.hstack((Y_process, Y_L))

        # Make a dataset object, but don't preprocess the data (already taken
        # care of above using the pretrained network preferences).
        dataset = DataSet()
        dataset.config = dict_values_to_list(kwargs)
        dataset.config.update({"dtype": [self.dtype]})
        dataset.from_raw_data(X_process, Y_process, percent_validation=0.01)

        # override default training config with kwargs
        self.gravity_model.config.update(**kwargs)
        self.gravity_model.train(dataset, initialize_optimizer=False)

    def save(self, df_file, data_dir):
        # save the network and config data using PINN-GM API
        self.gravity_model.save(df_file, data_dir)


class sphericalHarmonicModel:
    def __init__(self, model):
        self.gravity_model = model

    def compute_acceleration(self, X):
        return self.gravity_model.compute_acceleration(X)

    def compute_potential(self, X):
        return self.gravity_model.compute_potential(X)
