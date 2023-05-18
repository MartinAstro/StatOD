import numpy as np
from sympy import init_printing, print_latex
from tqdm import tqdm


class ProgressBar:
    def __init__(self, max_value, enable=False):
        self.max_value = max_value
        self.last_update = 0
        self.enable = enable
        self.p = self.pbar()

    def pbar(self):
        return tqdm(total=self.max_value, desc="Progress: ", disable=not self.enable)
        # ,
        # bar_format="%s{l_bar}{bar}|%s" % (Fore.YELLOW, Fore.RESET))

    def update(self, update_value):
        if update_value < self.max_value:
            self.p.update(update_value - self.last_update)
            self.last_update = update_value
        else:
            self.p.update(self.max_value - self.last_update)
            self.last_update = self.max_value

    def markComplete(self):
        if self.update == self.max_value:
            return
        self.p.update(self.max_value - self.last_update)

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


def latlon2cart(R, lat, lon):
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

    return np.array([x, y, z])


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
    rot_angle = np.array([omega * t + theta_0]).reshape((-1,))
    omega_vec = np.array([0, 0, omega])  # WARNING: Assumes spin only about z-axis!

    Z_rot = np.array(
        [
            [
                [np.cos(rot_angle[i]), -np.sin(rot_angle[i]), 0],
                [np.sin(rot_angle[i]), np.cos(rot_angle[i]), 0],
                [0, 0, 1],
            ]
            for i in range(len(rot_angle))
        ],
    )

    x_ECEF = X_ECEF[0:3]

    # rotate about Z axis
    x_ECI = Z_rot @ x_ECEF
    v_ECI = np.cross(omega_vec, x_ECI)

    return np.hstack((x_ECI, v_ECI)).squeeze()


def get_jac_sparsity_matrix():
    jac_sparsity = np.zeros((42, 42))

    xd_x = np.zeros((3, 3))
    xd_xd = np.eye(3)
    xd_phi = np.zeros((3, 36))

    xdd_x = np.full((3, 3), 1)
    xdd_xd = np.zeros((3, 3))
    xdd_phi = np.zeros((3, 36))

    phid_x = np.zeros((36, 3))

    phid_x[18:21, :] = np.full((3, 3), 1)
    phid_x[24:27, :] = np.full((3, 3), 1)
    phid_x[30:33, :] = np.full((3, 3), 1)

    # This won't be true for non-conservative systems
    phid_xd = np.zeros((36, 3))

    phid_phi = np.zeros((36, 36))
    phid_phi[3, 0] = 1
    phid_phi[4, 1] = 1
    phid_phi[5, 2] = 1

    phid_phi[9, 6 + 0] = 1
    phid_phi[10, 6 + 1] = 1
    phid_phi[11, 6 + 2] = 1

    phid_phi[15, 12 + 0] = 1
    phid_phi[16, 12 + 1] = 1
    phid_phi[17, 12 + 2] = 1

    def idx(i, j, N):
        # convert 2d matrix idx to flat idx
        idx = i * N + j
        return idx

    N = 6
    # phid_phi[phid_idx, phi_idx]

    # d/d_phi (d/dx (a) phi[3:6,0:3])
    phid_phi[idx(3, 0, N), idx(3, 0, N)] = 1
    phid_phi[idx(3, 0, N), idx(4, 0, N)] = 1
    phid_phi[idx(3, 0, N), idx(5, 0, N)] = 1

    phid_phi[idx(3, 1, N), idx(3, 1, N)] = 1
    phid_phi[idx(3, 1, N), idx(4, 1, N)] = 1
    phid_phi[idx(3, 1, N), idx(5, 1, N)] = 1

    phid_phi[idx(3, 2, N), idx(3, 2, N)] = 1
    phid_phi[idx(3, 2, N), idx(4, 2, N)] = 1
    phid_phi[idx(3, 2, N), idx(5, 2, N)] = 1

    phid_phi[idx(4, 0, N), idx(3, 0, N)] = 1
    phid_phi[idx(4, 0, N), idx(4, 0, N)] = 1
    phid_phi[idx(4, 0, N), idx(5, 0, N)] = 1

    phid_phi[idx(4, 1, N), idx(3, 1, N)] = 1
    phid_phi[idx(4, 1, N), idx(4, 1, N)] = 1
    phid_phi[idx(4, 1, N), idx(5, 1, N)] = 1

    phid_phi[idx(4, 2, N), idx(3, 2, N)] = 1
    phid_phi[idx(4, 2, N), idx(4, 2, N)] = 1
    phid_phi[idx(4, 2, N), idx(5, 2, N)] = 1

    phid_phi[idx(5, 0, N), idx(3, 0, N)] = 1
    phid_phi[idx(5, 0, N), idx(4, 0, N)] = 1
    phid_phi[idx(5, 0, N), idx(5, 0, N)] = 1

    phid_phi[idx(5, 1, N), idx(3, 1, N)] = 1
    phid_phi[idx(5, 1, N), idx(4, 1, N)] = 1
    phid_phi[idx(5, 1, N), idx(5, 1, N)] = 1

    phid_phi[idx(5, 2, N), idx(3, 2, N)] = 1
    phid_phi[idx(5, 2, N), idx(4, 2, N)] = 1
    phid_phi[idx(5, 2, N), idx(5, 2, N)] = 1

    jac_sparsity = np.block(
        [
            [xd_x, xd_xd, xd_phi],
            [xdd_x, xdd_xd, xdd_phi],
            [phid_x, phid_xd, phid_phi],
        ],
    )

    return jac_sparsity


# Iterate through a dictionary and for any value that is not a list, make it a list
def dict_values_to_list(d):
    for k, v in d.items():
        # check if value is a function, if so save name
        if callable(v):
            d[k] = [v.__name__]
            continue
        if not isinstance(v, list):
            d[k] = [v]
    return d
