import numpy as np
from sympy import *


def f_C22(t, x, args):
    x, y, z, vx, vy, vz = x
    R, mu, C20, S20, C21, S21, C22, S22 = args

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    r_mag = sqrt(x**2 + y**2 + z**2)

    longitude = atan2(y, x)
    sin_phi = z / r_mag
    P_20 = assoc_legendre(2, 0, sin_phi) * (
        C20 * cos(0 * longitude) + S20 * sin(0 * longitude)
    )
    P_21 = assoc_legendre(2, 1, sin_phi) * (
        C21 * cos(1 * longitude) + S21 * sin(1 * longitude)
    )
    P_22 = assoc_legendre(2, 2, sin_phi) * (
        C22 * cos(2 * longitude) + S22 * sin(2 * longitude)
    )

    U_0 = -(mu / r_mag)
    U_2 = -(mu / r_mag) * (R / r_mag) ** 2 * (P_20 + P_21 + P_22)
    U = U_0 + U_2

    a_x = -differentiate(U, x)
    a_y = -differentiate(U, y)
    a_z = -differentiate(U, z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()


##########################
# Simple Cartesian State #
##########################
def f_point_mass(x, args):
    x, y, z, vx, vy, vz = x
    mu, x_body, y_body, z_body = args

    r_mag = sqrt((x - x_body) ** 2 + (y - y_body) ** 2 + (z - z_body) ** 2)

    U0 = -mu / r_mag

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)

    a_x = -U0_x
    a_y = -U0_y
    a_z = -U0_z

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()


def f_J2(t, x, args):
    x, y, z, vx, vy, vz = x
    R, mu, J2 = args

    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2 * l - 1) * arg * P(arg, l - 1) - (l - 1) * P(arg, l - 2)) / l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z / r_mag, 2)

    U0 = -mu / r_mag
    U2 = (mu / r_mag) * (R / r_mag) ** 2 * P2 * J2

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    a_x = -(U0_x + U2_x)
    a_y = -(U0_y + U2_y)
    a_z = -(U0_z + U2_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()


def f_J3(t, x, args):
    x, y, z, vx, vy, vz = x
    R, mu, J2, J3 = args

    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2 * l - 1) * arg * P(arg, l - 1) - (l - 1) * P(arg, l - 2)) / l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z / r_mag, 2)
    P3 = P(z / r_mag, 3)

    U0 = -mu / r_mag
    U2 = (mu / r_mag) * (R / r_mag) ** 2 * P2 * J2
    U3 = (mu / r_mag) * (R / r_mag) ** 3 * P3 * J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()


############################
# Augmented State Dynamics #
############################
def f_aug_J2(t, x, args):
    x, y, z, vx, vy, vz, mu, J2 = x
    (R,) = args

    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2 * l - 1) * arg * P(arg, l - 1) - (l - 1) * P(arg, l - 2)) / l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z / r_mag, 2)

    U0 = -mu / r_mag
    U2 = (mu / r_mag) * (R / r_mag) ** 2 * P2 * J2

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    a_x = -(U0_x + U2_x)
    a_y = -(U0_y + U2_y)
    a_z = -(U0_z + U2_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0]).tolist()


def f_aug_J3(t, x, args):
    x, y, z, vx, vy, vz, mu, J2, J3 = x
    (R,) = args

    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2 * l - 1) * arg * P(arg, l - 1) - (l - 1) * P(arg, l - 2)) / l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z / r_mag, 2)
    P3 = P(z / r_mag, 3)

    U0 = -mu / r_mag
    U2 = (mu / r_mag) * (R / r_mag) ** 2 * P2 * J2
    U3 = (mu / r_mag) * (R / r_mag) ** 3 * P3 * J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0, 0.0]).tolist()


###########################
# Consider State Dynamics #
###########################
def f_consider_J3(t, x, args):
    x, y, z, vx, vy, vz, mu, J2 = x
    R, J3 = args

    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2 * l - 1) * arg * P(arg, l - 1) - (l - 1) * P(arg, l - 2)) / l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z / r_mag, 2)
    P3 = P(z / r_mag, 3)

    U0 = -mu / r_mag
    U2 = (mu / r_mag) * (R / r_mag) ** 2 * P2 * J2
    U3 = (mu / r_mag) * (R / r_mag) ** 3 * P3 * J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0]).tolist()
