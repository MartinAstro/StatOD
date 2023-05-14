import copy
import pickle

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import HeterogeneousPoly
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from scipy.integrate import solve_ivp

from Scripts.AsteroidScenarios.helper_functions import compute_BN
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from StatOD.models import pinnGravityModel
from StatOD.utils import ProgressBar


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
    mass_1 = copy.deepcopy(planet)
    mass_1.mu = mass_1.mu / 10
    r_offset_1 = [mass_1.radius / 3, 0, 0]

    mass_2 = copy.deepcopy(planet)
    mass_2.mu = -mass_2.mu / 10
    r_offset_2 = [-mass_2.radius / 3, 0, 0]

    point_mass_1 = PointMass(mass_1)
    point_mass_2 = PointMass(mass_2)

    poly_r0_gm.add_point_mass(point_mass_1, r_offset_1)
    poly_r0_gm.add_point_mass(point_mass_2, r_offset_2)

    return poly_r0_gm


def f_ivp(t, Z, model, pbar, omega):
    R_N = Z[0:3] * 1e3  # convert to meters
    V_N = Z[3:6] * 1e3  # convert from km/s -> m/s

    BN = compute_BN(t, omega).squeeze()
    R_B = BN @ R_N

    # m / s^2
    R_B = R_B.reshape((-1, 3))
    A_B = model.compute_acceleration(R_B).reshape((-1))
    A_N = BN.T @ A_B.squeeze()

    pbar.update(t)
    return np.hstack((V_N, A_N)) / 1e3  # convert from m -> km


def generate_rotating_asteroid_trajectory(
    X0_km_N,
    filename,
    model_file,
    timestep=30,
    orbits=2.5,
):
    # compute orbit period
    ep = ErosParams()
    a = compute_semimajor(X0_km_N, ep.mu)
    if np.isnan(a):
        a = 100
    n = np.sqrt(ep.mu / a**3)
    T = 2 * np.pi / n

    # generate true gravity model
    eros = Eros()
    gravity_model_true = generate_heterogeneous_model(eros, eros.obj_8k)

    # integrate trajectory
    t_f = T * orbits
    pbar = ProgressBar(t_f, enable=True)
    t_mesh = np.arange(0, t_f, step=timestep)
    sol = solve_ivp(
        f_ivp,
        [0, t_f],
        X0_km_N,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_mesh,
        args=(gravity_model_true, pbar, ep.omega),
    )

    # compute body frame accelerations along trajectory
    R_N = sol.y[0:3, :].T * 1e3
    BN = compute_BN(sol.t, ep.omega)
    R_B = np.einsum("ijk,ik->ij", BN, R_N)
    acc_true_B_m = gravity_model_true.compute_acceleration(R_B).reshape((-1, 3))

    # compute body frame accelerations along trajectory using other models
    statOD_dir = os.path.dirname(StatOD.__file__)
    gravity_model_poly = Polyhedral(eros, eros.obj_8k)
    gravity_model_pinn = pinnGravityModel(
        f"{statOD_dir}/../Data/Dataframes/{model_file}.data",
    )

    acc_poly_B_m = gravity_model_poly.compute_acceleration(R_B).reshape((-1, 3))
    acc_pinn_B_m = gravity_model_pinn.compute_acceleration(R_B).reshape((-1, 3))

    # compute inertial frame accelerations along trajectory
    NB = np.transpose(BN, axes=[0, 2, 1])
    acc_true_N_m = np.einsum("ijk,ik->ij", NB, acc_true_B_m)
    acc_poly_N_m = np.einsum("ijk,ik->ij", NB, acc_poly_B_m)
    acc_pinn_N_m = np.einsum("ijk,ik->ij", NB, acc_pinn_B_m)

    # convert to kilometers
    acc_true_km = acc_true_N_m / 1e3
    acc_poly_km = acc_poly_N_m / 1e3
    acc_pinn_km = acc_pinn_N_m / 1e3

    N = len(X0_km_N)
    data = {
        "t": sol.t,
        "X": sol.y[:N, :].T,  # state in km and km/s
        "X_B": R_B / 1e3,  # position in km
        "A_B": acc_true_B_m / 1e3,  # acceleration in km/s^2
        # # true unmodeled acceleration in km/s^2
        "W": acc_true_km - acc_poly_km,
        # # estimated unmodeled acceleration in km/s^2
        "W_pinn": acc_true_km - acc_pinn_km,
    }
    with open(f"Data/Trajectories/{filename}.data", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    ep = ErosParams()
    X0_m_N = np.array(
        [
            3.16800e04,
            0.00e00,
            -3.00e03,
            0.1,
            -2.75,
            2.25,
        ],
    )
    # high altitude
    a = ep.R * 1000 * 2  # m
    r0 = np.array([a, 0, a])  # m
    r_mag = np.linalg.norm(r0)
    v_mag = np.sqrt(ep.mu * 1e9 * (2.0 / r_mag - 1.0 / a))  # m/s
    v0 = np.array([0, v_mag, 0])  # m/s
    X0_m_N = np.append(r0, v0)

    # Julio's params (the best)
    # 34000,
    # 0.001,
    # np.pi / 4,
    # np.deg2rad(48.2),
    # np.deg2rad(347.8),
    # np.deg2rad(85.3),
    X0_m_N = np.array(
        [
            -19243.595703125,
            21967.5078125,
            17404.74609375,
            -2.939612865447998,
            -1.1707247495651245,
            -1.7654979228973389,
        ],
    )

    X0_km_N = X0_m_N / 1e3

    # generate_rotating_asteroid_trajectory(
    #     X0_km_N,
    #     "traj_rotating_gen_III",
    #     "eros_filter_poly",
    #     timestep=60,
    #     orbits=3,
    # )
    # generate_rotating_asteroid_trajectory(
    #     X0_km_N,
    #     "traj_rotating_gen_III_constant",
    #     "eros_constant_poly",
    #     timestep=60,
    #     orbits=3,
    # )
    # generate_rotating_asteroid_trajectory(
    #     X0_km_N,
    #     "traj_rotating_gen_III_constant_no_fuse",
    #     "eros_constant_poly_no_fuse",
    #     timestep=60,
    #     orbits=3,
    # )

    filename = "traj_rotating_gen_III_constant_dropout"
    generate_rotating_asteroid_trajectory(
        X0_km_N,
        "traj_rotating_gen_III_constant_dropout",
        "eros_constant_poly_dropout",
        timestep=60,
        orbits=10,
    )

    from Scripts.Plots.plot_asteroid_trajectory import main

    main(f"Data/Trajectories/{filename}.data")

    from Scripts.Measurements.generate_position_measurements import (
        generate_measurements,
    )

    generate_measurements(f"Data/Trajectories/{filename}.data")
