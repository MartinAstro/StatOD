import numpy as np
import pandas as pd
from GravNN.Networks.Constraints import get_PI_constraint
from GravNN.Networks.Data import DataSet
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.utils import configure_optimizer
from GravNN.Support.transformations import cart2sph, invert_projection

from StatOD.utils import dict_values_to_list


class pinnGravityModel:
    def __init__(
        self,
        df_file,
        custom_data_dir="",
        learning_rate=None,
        dim_constants=None,
    ):

        # tf.keras.mixed_precision.Policy("float64")
        df = pd.read_pickle(custom_data_dir + df_file)
        config, gravity_model = load_config_and_model(
            df.id.values[-1],
            df,
            custom_dtype="float64",
        )
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config["planet"][0]
        if learning_rate is not None:
            self.config["learning_rate"][0] = learning_rate
        self.optimizer = configure_optimizer(self.config, None)
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

        # # configure preprocessing layers
        # x_transformer = config["x_transformer"][0]
        # u_transformer = config["u_transformer"][0]
        # a_transformer = config["a_transformer"][0]

        # x_star = x_transformer.scale_
        # u_star = u_transformer.scale_
        # a_star = a_transformer.scale_

        # # TODO: Fix this, it's very counter intuitive.
        # x_preprocessor = PreprocessingLayer(0, x_star, tf.float64)
        # u_postprocessor = PostprocessingLayer(0, u_star, tf.float64)
        # a_preprocessor = PreprocessingLayer(0, a_star, tf.float64)
        # a_postprocessor = PostprocessingLayer(0, a_star, tf.float64)

        # self.gravity_model.x_preprocessor = x_preprocessor
        # self.gravity_model.u_postprocessor = u_postprocessor
        # self.gravity_model.a_preprocessor = a_preprocessor
        # self.gravity_model.a_postprocessor = a_postprocessor

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
        # self.gravity_model.scale_loss = PINN_variables[1]
        # self.gravity_model.adaptive_constant = tf.Variable(PINN_variables[2], dtype=self.config['dtype'][0])
        self.config["PINN_constraint_fcn"] = [PINN_constraint_fcn]

    def train(self, X, Y, **kwargs):
        # Make sure Y_DMC has the gravity model accelerations added to it
        # tf.config.run_functions_eagerly(True)

        # COM data ?
        # X_com = np.full((10, 3), 0.1) # can't be zero exactly otherwise zeros cause issues
        # Y_com = np.full((10, 3), 0.0)
        # X = np.vstack((X, X_com))
        # Y = np.vstack((Y, Y_com))

        # dimensionalize
        X_dim = X * (self.dim_constants["l_star"])
        A_dim = Y * (self.dim_constants["l_star"] / self.dim_constants["t_star"] ** 2)

        # non-dimensionalize / preprocess (in case different scheme was used)
        X_process = self.gravity_model.x_preprocessor(X_dim).numpy()
        Y_process = self.gravity_model.a_preprocessor(A_dim).numpy()
        if self.config["PINN_constraint_fcn"][0] == "pinn_alc":
            Y_LC = np.full((len(Y_process), 4), 0.0)
            Y_process = np.hstack((Y_process, Y_LC))

            # # Use the COM as a constraint
            # internal_density = kwargs.get('internal_density', None)
            # if internal_density is not None:
            #     internal_density_non_dim = internal_density * self.dim_constants['l_star']**3
            #     X_com = np.full((10, 3), 1E-3) # can't be zero exactly otherwise zeros cause issues
            #     Y_com = np.full((10, 7), [0,0,0,internal_density_non_dim, 0,0,0])
            #     X_process = np.vstack((X_process, X_com))
            #     Y_process = np.vstack((Y_process, Y_com))

        batch_size = kwargs.get("batch_size", 32)
        dataset = DataSet()
        dataset.config = dict_values_to_list(kwargs)
        dataset.from_raw_data(X_process, Y_process, percent_validation=0.01)
        # dataset = generate_dataset(X_process, Y_process, batch_size, dtype=self.config['dtype'][0])
        # dataset.shuffle(buffer_size=batch_size)
        self.gravity_model.compile(optimizer=self.optimizer, loss="mse")
        self.gravity_model.fit(
            dataset.train_data,
            batch_size=batch_size,
            epochs=kwargs.get("epochs", [5])[0],
            use_multiprocessing=True,
        )

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
