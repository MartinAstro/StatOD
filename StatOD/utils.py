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
