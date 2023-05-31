import copy

import numpy as np
from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
from GravNN.GravityModels.PointMass import PointMass


def compute_semimajor(X, mu):
    def cross(x, y):
        return np.cross(x, y)

    r = X[0:3]
    v = X[3:6]
    h = cross(r, v)
    p = np.dot(h, h) / mu
    e = cross(v, h) / mu - r / np.linalg.norm(r)
    a = p / (1 - np.linalg.norm(e) ** 2)
    return a


def generate_heterogeneous_model(planet, shape_model):
    poly_r0_gm = HeterogeneousPoly(planet, shape_model)

    # Force the following mass inhomogeneity
    mass_0 = copy.deepcopy(planet)
    mass_0.mu = -2 * mass_0.mu / 20
    r_offset_0 = [0, 0, 0]

    mass_1 = copy.deepcopy(planet)
    mass_1.mu = mass_1.mu / 20
    r_offset_1 = [mass_1.radius / 2, 0, 0]

    mass_2 = copy.deepcopy(planet)
    mass_2.mu = mass_2.mu / 20
    r_offset_2 = [-mass_2.radius / 2, 0, 0]

    point_mass_0 = PointMass(mass_0)
    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    poly_r0_gm.add_point_mass(point_mass_0, r_offset_0)
    poly_r0_gm.add_point_mass(point_mass_1, r_offset_1)
    poly_r0_gm.add_point_mass(point_mass_2, r_offset_2)

    return poly_r0_gm
