from multiprocessing.sharedctypes import Value
import numpy as np
from sympy import init_printing, print_latex

from tqdm import tqdm
from colorama import Fore


class ProgressBar:
    def __init__(self, max_value, enable=False):
        self.max_value = max_value
        self.last_update = 0
        self.enable = enable
        self.p = self.pbar()

    def pbar(self):
        return tqdm(
            total=self.max_value,
            desc='Progress: ',
            disable=not self.enable)
            #,
            #bar_format="%s{l_bar}{bar}|%s" % (Fore.YELLOW, Fore.RESET))

    def update(self, update_value):
        if update_value < self.max_value:
            self.p.update(update_value-self.last_update)
            self.last_update = update_value
        else:
            self.p.update(self.max_value - self.last_update)
            self.last_update = self.max_value

    def markComplete(self):
        if self.update == self.max_value:
            return
        self.p.update(self.max_value-self.last_update)

    def close(self):
        self.p.close()



def print_expression(expression, prefix=""):
    init_printing() 

    for index, x in np.ndenumerate(expression):
        if x == 0:
            continue

        subscript = "_{"
        for i in index:
            subscript += str(i)
        subscript += "}"

        prefix_inst = prefix + subscript

        print(prefix_inst, end=" & =")
        print_latex(x)

def latlon2cart(R,lat,lon):
    """Convert Latitude and Longitude to Cartesian Position

    Args:
        R (km): Radius of body in km
        lat (deg): Latitude in radians [0, pi]
        lon (deg): Longitude in radians [0, 2*pi]
    """
    if lat < 0:
        ValueError("Latitude must be between 0 and pi")

    x = R * np.cos(lon) * np.sin(lat)
    y = R * np.sin(lon) * np.sin(lat)
    z = R * np.cos(lat)

    return np.array([x,y,z])

# def ECEF_2_ECI(lat, lon):
#     """Convert a vector represented in [r_hat, phi_hat, theta_hat]
#     into one of [x_hat, y_hat, z_hat]. See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates 

#     Args:
#         lat ([float]): [latitude in radians]
#         lon ([float]): [longitude in radians]

#     Returns:
#         [np.array]: [NB]
#     """
#     theta = lat
#     phi = lon
#     NB = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
#                     [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
#                     [-np.sin(phi)             , np.cos(phi)              , 0]])
    
#     return NB

def ECEF_2_ECI(t, X_ECEF, omega, theta_0):
    rot_angle = np.array([omega*t + theta_0]).reshape((-1,))
    omega_vec = np.array([0,0,omega]) # WARNING: Assumes spin only about z-axis!

    Z_rot = np.array([ [[np.cos(rot_angle[i]), -np.sin(rot_angle[i]), 0],
                        [np.sin(rot_angle[i]),  np.cos(rot_angle[i]), 0],
                        [0,                   0,                      1]]
                        for i in range(len(rot_angle))
                      ])

    x_ECEF = X_ECEF[0:3]

    # rotate about Z axis
    x_ECI = Z_rot@x_ECEF
    v_ECI = np.cross(omega_vec, x_ECI)

    return np.hstack((x_ECI, v_ECI)).squeeze()


from GravNN.Support.transformations import cart2sph, invert_projection, project_acceleration
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Layers import PreprocessingLayer, PostprocessingLayer
from GravNN.Networks.Data import generate_dataset
from GravNN.Networks.utils import configure_optimizer, _get_loss_fcn, _get_PI_constraint
import pandas as pd
import tensorflow as tf

class pinnGravityModel():
    def __init__(self, df_file, custom_data_dir="", learning_rate=None, dim_constants=None):
        df = pd.read_pickle(custom_data_dir + df_file)
        config, gravity_model = load_config_and_model(df.iloc[-1]['id'], df)
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config['planet'][0]
        if learning_rate is not None:
            self.config['learning_rate'][0] = learning_rate
        self.optimizer = configure_optimizer(self.config, None)
        if dim_constants is None:
            self.dim_constants = {
                "m_star" : 1.0,
                "t_star" : 1.0,
                "l_star" : 1.0,
            }
        else:
            self.dim_constants = dim_constants

        removed_pm = config.get('remove_point_mass', [False])[0]
        deg_removed = config.get('deg_removed', [-1])[0]
        if removed_pm or deg_removed > -1:
            self.removed_pm = True
        else:
            self.removed_pm = False

        # configure preprocessing layers
        x_transformer = config['x_transformer'][0]
        u_transformer = config['u_transformer'][0]
        a_transformer = config['a_transformer'][0]

        # TODO: Fix this, it's very counter intuitive. 
        x_preprocessor = PostprocessingLayer(x_transformer.data_min_, x_transformer.data_range_, tf.float64) # normalizing layer
        u_postprocessor = PreprocessingLayer(u_transformer.data_min_, u_transformer.data_range_, tf.float64) # unnormalize layer
        a_preprocessor = PostprocessingLayer(a_transformer.data_min_, a_transformer.data_range_, tf.float64) # normalizing layer
        a_postprocessor = PreprocessingLayer(a_transformer.data_min_, a_transformer.data_range_, tf.float64) # unormalizing layer

        self.gravity_model.x_preprocessor = x_preprocessor
        self.gravity_model.u_postprocessor = u_postprocessor
        self.gravity_model.a_preprocessor = a_preprocessor
        self.gravity_model.a_postprocessor = a_postprocessor

    def generate_acceleration(self, X):
        X_dim = X*self.dim_constants['l_star']
        R = np.array(X_dim).reshape((-1,3)).astype(np.float32)
        a_model = self.gravity_model.generate_acceleration(R).numpy() # this is currently a_r > 0

        if not self.removed_pm:
            a_model = a_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            a_pm_sph = np.zeros((len(R), 3))
            a_pm_sph[:,0] = -self.planet.mu/r**2
            r_sph = cart2sph(R)
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            a_model = (a_pm_xyz + a_model).squeeze() # a_r < 0

        a_non_dim = a_model / (self.dim_constants['l_star'] / self.dim_constants['t_star']**2)
        return a_non_dim


    def generate_dadx(self, X):
        X_dim = X*self.dim_constants['l_star']
        R = np.array(X_dim).reshape((-1,3)).astype(np.float32)
        dadx_dim = self.gravity_model.generate_dU_dxdx(R).numpy() # this is also > 0
        dadx_non_dim = dadx_dim / (1.0 / self.dim_constants['t_star']**2)
        return dadx_non_dim.squeeze()

    def generate_potential(self, X):
        X_dim = X*self.dim_constants['l_star']
        R = np.array(X_dim).reshape((-1,3)).astype(np.float32)
        U_model = self.gravity_model.generate_potential(R).numpy() # this is also > 0
        if not self.removed_pm:
            U_model = U_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            U_pm = np.zeros((len(R), 1))
            U_pm[:,0] = -self.planet.mu/r
            U_model = (U_pm + U_model).squeeze()

        U_model_non_dim = U_model / (self.dim_constants['l_star']**2/ self.dim_constants['t_star']**2)
        return U_model_non_dim

    def set_PINN_training_fcn(self, PINN_constraint_fcn):
        PINN_variables = _get_PI_constraint(PINN_constraint_fcn)
        self.gravity_model.eval = PINN_variables[0]
        self.gravity_model.scale_loss = PINN_variables[1]
        self.gravity_model.adaptive_constant = tf.Variable(PINN_variables[2], dtype=tf.float32)
        self.config['PINN_constraint_fcn'] = [PINN_constraint_fcn]

    def train(self, X, Y, **kwargs):
        # tf.config.run_functions_eagerly(True)
        A = Y + self.generate_acceleration(X)
        X_dim = X*(self.dim_constants['l_star'])
        A_dim = A*(self.dim_constants['l_star'] / self.dim_constants['t_star']**2)
        X_process = self.gravity_model.x_preprocessor(X_dim).numpy()
        Y_process = self.gravity_model.a_preprocessor(A_dim).numpy()
        if self.config['PINN_constraint_fcn'][0] == "pinn_alc":
            Y_LC = np.full((len(Y_process), 4), 0.0)
            Y_process = np.hstack((Y_process, Y_LC))

        batch_size = kwargs.get("batch_size", 32)
        dataset = generate_dataset(X_process, Y_process, batch_size)
        dataset.shuffle(buffer_size=batch_size)
        self.gravity_model.compile(optimizer=self.optimizer, loss='mse')
        self.gravity_model.fit(
            dataset,
            batch_size=batch_size,
            epochs=kwargs.get("epochs", 5),
            use_multiprocessing=True,
        )

    def save(self, df_file, data_dir):
        # save the network and config data using PINN-GM API
        self.gravity_model.save(df_file, data_dir)


def get_jac_sparsity_matrix():
    jac_sparsity = np.zeros((42,42))

    xd_x = np.zeros((3,3))
    xd_xd = np.eye(3)
    xd_phi = np.zeros((3,36))

    xdd_x = np.full((3,3), 1)
    xdd_xd = np.zeros((3,3))
    xdd_phi = np.zeros((3,36))

    phid_x = np.zeros((36, 3))


    phid_x[18:21,:] = np.full((3,3), 1)
    phid_x[24:27,:] = np.full((3,3), 1)
    phid_x[30:33,:] = np.full((3,3), 1)

    # This won't be true for non-conservative systems 
    phid_xd = np.zeros((36,3))

    phid_phi = np.zeros((36,36))
    phid_phi[3, 0] = 1
    phid_phi[4, 1] = 1
    phid_phi[5, 2] = 1

    phid_phi[9, 6+0] = 1
    phid_phi[10, 6+1] = 1
    phid_phi[11, 6+2] = 1

    phid_phi[15, 12+0] = 1
    phid_phi[16, 12+1] = 1
    phid_phi[17, 12+2] = 1


    def idx(i,j,N):
        # convert 2d matrix idx to flat idx
        idx = i*N + j
        return idx

    N = 6
    # phid_phi[phid_idx, phi_idx]

    # d/d_phi (d/dx (a) phi[3:6,0:3])
    phid_phi[idx(3,0,N), idx(3,0,N)] = 1
    phid_phi[idx(3,0,N), idx(4,0,N)] = 1
    phid_phi[idx(3,0,N), idx(5,0,N)] = 1

    phid_phi[idx(3,1,N), idx(3,1,N)] = 1
    phid_phi[idx(3,1,N), idx(4,1,N)] = 1
    phid_phi[idx(3,1,N), idx(5,1,N)] = 1

    phid_phi[idx(3,2,N), idx(3,2,N)] = 1
    phid_phi[idx(3,2,N), idx(4,2,N)] = 1
    phid_phi[idx(3,2,N), idx(5,2,N)] = 1


    phid_phi[idx(4,0,N), idx(3,0,N)] = 1
    phid_phi[idx(4,0,N), idx(4,0,N)] = 1
    phid_phi[idx(4,0,N), idx(5,0,N)] = 1

    phid_phi[idx(4,1,N), idx(3,1,N)] = 1
    phid_phi[idx(4,1,N), idx(4,1,N)] = 1
    phid_phi[idx(4,1,N), idx(5,1,N)] = 1
    
    phid_phi[idx(4,2,N), idx(3,2,N)] = 1
    phid_phi[idx(4,2,N), idx(4,2,N)] = 1
    phid_phi[idx(4,2,N), idx(5,2,N)] = 1


    phid_phi[idx(5,0,N), idx(3,0,N)] = 1
    phid_phi[idx(5,0,N), idx(4,0,N)] = 1
    phid_phi[idx(5,0,N), idx(5,0,N)] = 1

    phid_phi[idx(5,1,N), idx(3,1,N)] = 1
    phid_phi[idx(5,1,N), idx(4,1,N)] = 1
    phid_phi[idx(5,1,N), idx(5,1,N)] = 1
    
    phid_phi[idx(5,2,N), idx(3,2,N)] = 1
    phid_phi[idx(5,2,N), idx(4,2,N)] = 1
    phid_phi[idx(5,2,N), idx(5,2,N)] = 1



    jac_sparsity = np.block([
        [xd_x,  xd_xd,   xd_phi],
        [xdd_x, xdd_xd,  xdd_phi],
        [phid_x,phid_xd, phid_phi]
    ])

    return jac_sparsity