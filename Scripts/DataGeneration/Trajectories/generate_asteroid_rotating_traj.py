import pickle

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from scipy.integrate import solve_ivp

import StatOD
from Scripts.Factories.DynArgsFactory import DynArgsFactory
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from StatOD.models import pinnGravityModel
from StatOD.utils import (
    ProgressBar,
    compute_BN,
    compute_semimajor,
    generate_heterogeneous_model,
)


class ModelWrapper:
    def __init__(self, model, dim_constants):
        self.model = model
        self.dim_constants = dim_constants

    def compute_acceleration(self, X):
        l_star = self.dim_constants["l_star"]
        t_star = self.dim_constants["t_star"]

        ms2 = l_star / t_star**2
        X_dim = X * self.dim_constants["l_star"]

        return self.model.compute_acceleration(X_dim.reshape((-1, 3)) * 1e3) / 1e3 / ms2

    def generate_dadx(self, X):
        t_star = self.dim_constants["t_star"]
        X_dim = X * self.dim_constants["l_star"]

        c = 1.0 / t_star**2
        return self.model.compute_dfdx(X_dim.reshape((-1, 3)) * 1e3) / c


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


# @njit(cache=False)
def dynamics_ivp_f_only(t, Z, f, dfdx, f_args, pbar):
    X_inst = Z
    f_inst = np.array(f(t, X_inst, f_args))
    pbar.update(t)
    return f_inst.reshape((-1))


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
    dim_constants = {"l_star": 1.0, "t_star": 1.0}
    eros = Eros()
    gravity_model_true = generate_heterogeneous_model(eros, eros.obj_8k)
    gravity_model_true = ModelWrapper(gravity_model_true, dim_constants)
    # gravity_model_true = Polyhedral(eros, eros.obj_8k)

    # integrate trajectory
    t_f = T * orbits
    pbar = ProgressBar(t_f, enable=True)

    f_args = DynArgsFactory().get_HF_args(gravity_model_true)
    f, dfdx, q, q_args = get_DMC_HF_zero_order()
    f_integrate = dynamics_ivp_f_only

    Z0 = np.hstack((X0_km_N, np.zeros((3,))))

    t_mesh = np.arange(0, t_f, step=timestep)
    sol = solve_ivp(
        f_integrate,
        [0, t_f],
        Z0,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_mesh,
        args=(f, dfdx, f_args, pbar),
    )

    # compute body frame accelerations along trajectory
    R_N = sol.y[0:3, :].T * 1e3
    BN = compute_BN(sol.t, ep.omega)
    R_B = np.einsum("ijk,ik->ij", BN, R_N)
    acc_grav_B_m = gravity_model_true.compute_acceleration(R_B).reshape((-1, 3))

    # compute body frame accelerations along trajectory using other models
    statOD_dir = os.path.dirname(StatOD.__file__)
    gravity_model_pinn = pinnGravityModel(
        f"{statOD_dir}/../Data/Dataframes/{model_file}.data",
    )

    acc_pinn_B_m = gravity_model_pinn.compute_acceleration(R_B).reshape((-1, 3))

    # compute inertial frame accelerations along trajectory
    NB = np.transpose(BN, axes=[0, 2, 1])
    acc_grav_N_m = np.einsum("ijk,ik->ij", NB, acc_grav_B_m)
    acc_pinn_N_m = np.einsum("ijk,ik->ij", NB, acc_pinn_B_m)

    Z_i = sol.y.T
    Zd = np.array(
        [
            f_integrate(t_mesh[i], Z_i[i], f, dfdx, f_args, pbar)
            for i in range(len(t_mesh))
        ],
    )
    acc_true_km = Zd[:, 3:6]

    # convert to kilometers
    acc_grav_km = acc_grav_N_m / 1e3
    acc_pinn_km = acc_pinn_N_m / 1e3

    N = len(X0_km_N)
    data = {
        "t": sol.t,
        "X": sol.y[:N, :].T,  # state in km and km/s
        "X_B": R_B,  # position in km
        "A_B": acc_grav_B_m,  # acceleration in km/s^2
        # # true unmodeled acceleration in km/s^2
        "W": acc_true_km - acc_grav_km,
        # # estimated unmodeled acceleration in km/s^2
        "W_pinn": acc_true_km - acc_pinn_km,
    }

    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    with open(f"{statOD_dir}/Data/Trajectories/{filename}.data", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
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

    pinn_file = "eros_pm_053123"
    filename = f"traj_{pinn_file}"
    generate_rotating_asteroid_trajectory(
        X0_km_N,
        filename,
        pinn_file,
        timestep=60,
        orbits=10,
    )

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    from Scripts.DataGeneration.Measurements.generate_position_measurements import (
        generate_measurements,
    )

    generate_measurements(f"{statOD_dir}Data/Trajectories/{filename}.data")
