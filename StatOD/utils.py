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
import pandas as pd
import tensorflow as tf

class pinnGravityModel():
    def __init__(self, df_file):
        df = pd.read_pickle(df_file)
        config, gravity_model = load_config_and_model(df.iloc[-1]['id'], df)
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config['planet'][0]
        removed_pm = config.get('remove_point_mass', [False])[0]
        deg_removed = config.get('deg_removed', [-1])[0]
        if removed_pm or deg_removed > -1:
            self.removed_pm = True
        else:
            self.removed_pm = False

        # configure preprocessing layers
        x_transformer = config['x_transformer'][0]
        u_transformer = config['u_transformer'][0]

        x_preprocessor = PreprocessingLayer(x_transformer.min_, x_transformer.scale_, tf.float64)
        u_postprocessor = PostprocessingLayer(u_transformer.min_, u_transformer.scale_, tf.float64)

        self.gravity_model.x_preprocessor = x_preprocessor
        self.gravity_model.u_postprocessor = u_postprocessor

    def generate_acceleration(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        a_model = self.gravity_model.generate_acceleration(R).numpy() # this is currently a_r > 0
        # a_model *= -1 # to make dynamics work, this will be multiplied by -1    
        
        if not self.removed_pm:
            return a_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            a_pm_sph = np.zeros((len(R), 3))
            a_pm_sph[:,0] = -self.planet.mu/r**2
            r_sph = cart2sph(R)
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            return (a_pm_xyz + a_model).squeeze() # a_r < 0


    def generate_dadx(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        dadx = self.gravity_model.generate_dU_dxdx(R).numpy() # this is also > 0
        return dadx.squeeze()

    def generate_potential(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        U_model = self.gravity_model.generate_potential(R).numpy() # this is also > 0
        if not self.removed_pm:
            return U_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            U_pm = np.zeros((len(R), 1))
            U_pm[:,0] = -self.planet.mu/r
            return (U_pm + U_model).squeeze()