import os
import pickle

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
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


def compute_rot_acc(R, V, omega):
    x = R[..., 0]
    y = R[..., 1]
    # z = R[...,2]

    x_d = V[..., 0]
    y_d = V[..., 1]
    # z_d = V[...,2]

    A_rot_x = omega**2 * x + 2 * omega * y_d
    A_rot_y = omega**2 * y - 2 * omega * x_d
    A_rot_z = np.zeros_like(A_rot_x)
    A_rot = np.vstack(
        (
            A_rot_x.reshape((1, -1)),
            A_rot_y.reshape((1, -1)),
            A_rot_z.reshape((1, -1)),
        ),
    ).T
    return A_rot.squeeze()


def generate_rotating_asteroid_trajectory(
    X0_km_N,
    filename,
    model_file,
    timestep=30,
    orbits=2.5,
):
    ep = ErosParams()  # max radius is 16 km

    a = compute_semimajor(X0_km_N, ep.mu)  # km
    if np.isnan(a):
        a = 100
    n = np.sqrt(ep.mu / a**3)
    T = 2 * np.pi / n
    t_f = T * orbits
    omega = ep.omega

    eros = Eros()
    gravity_model = Polyhedral(eros, eros.obj_8k)
    gravity_model_pm = PointMass(eros)
    gravity_model_pinn = pinnGravityModel(
        os.path.dirname(StatOD.__file__) + f"/../Data/Dataframes/{model_file}.data",
    )

    N = len(X0_km_N)
    pbar = ProgressBar(t_f, enable=True)

    def f_ivp(t, Z):
        R_N = Z[0:3] * 1e3  # convert to meters
        V_N = Z[3:6] * 1e3  # convert from km/s -> m/s

        BN = compute_BN(t, omega).squeeze()
        R_B = BN @ R_N

        # m / s^2
        A_B = gravity_model.compute_acceleration(
            R_B.reshape((-1, 3)),
            pbar=False,
        ).reshape(
            (-1),
        )
        # A_B = gravity_model_pm.compute_acceleration(R_B.reshape((-1,3))).reshape((-1))
        A_N = BN.T @ A_B.squeeze()

        # A_rotate_N = compute_rot_acc(R_N, V_N, omega)
        # A_N -= A_rotate_N

        pbar.update(t)
        return np.hstack((V_N, A_N)) / 1e3  # convert from m -> km

    t_mesh = np.arange(0, t_f, step=timestep)
    sol = solve_ivp(f_ivp, [0, t_f], X0_km_N, atol=1e-12, rtol=1e-12, t_eval=t_mesh)

    # body frame accelerations
    R_N = sol.y[0:3, :].T * 1e3
    BN = compute_BN(sol.t, omega)
    R_B = np.einsum("ijk,ik->ij", BN, R_N)

    acc_poly_B_m = gravity_model.compute_acceleration(R_B, pbar=False).reshape((-1, 3))
    acc_pm_B_m = gravity_model_pm.compute_acceleration(R_B).reshape((-1, 3))
    acc_pinn_B_m = gravity_model_pinn.compute_acceleration(R_B).reshape((-1, 3))

    NB = np.transpose(BN, axes=[0, 2, 1])
    acc_poly_N_m = np.einsum("ijk,ik->ij", NB, acc_poly_B_m)
    acc_pm_N_m = np.einsum("ijk,ik->ij", NB, acc_pm_B_m)
    acc_pinn_N_m = np.einsum("ijk,ik->ij", NB, acc_pinn_B_m)

    # # add rotational accelerations
    # R_N = sol.y[0:3, :].T.reshape((-1,3))
    # V_N = sol.y[3:6, :].T.reshape((-1,3))
    # acc_rot_N_m = compute_rot_acc(R_N, V_N, omega)
    # acc_poly_N_m -= acc_rot_N_m
    # acc_pm_N_m -= acc_rot_N_m
    # acc_pinn_N_m -= acc_rot_N_m

    # convert to kilometers
    acc_poly_km = acc_poly_N_m / 1e3
    acc_pm_km = acc_pm_N_m / 1e3
    acc_pinn_km = acc_pinn_N_m / 1e3

    data = {
        "t": sol.t,
        "X": sol.y[:N, :].T,  # in km and km/s
        "X_B": R_B / 1e3,
        "A_B": acc_poly_B_m / 1e3,
        "W": acc_poly_km - acc_pm_km,
        "W_pinn": acc_poly_km - acc_pinn_km,
    }
    with open(f"Data/Trajectories/{filename}.data", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    ep = ErosParams()
    X0_m_N = np.array(
        [
            3.16800000e04,
            0.00000000e00,
            -3.00000000e03,
            0.1,
            -2.75,
            2.25,
        ],
    )
    X0_km_N = X0_m_N / 1e3

    generate_rotating_asteroid_trajectory(
        X0_km_N,
        "traj_rotating_gen_III",
        "eros_filter_poly",
        timestep=60,
        orbits=3,
    )

    from Scripts.Plots.plot_asteroid_trajectory import main

    main()
