import pickle

import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from scipy.integrate import solve_ivp

import StatOD
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from StatOD.models import pinnGravityModel
from StatOD.utils import (
    ProgressBar,
    compute_BN,
    compute_semimajor,
    generate_heterogeneous_model,
)


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

    pinn_file = "eros_poly_053123"
    filename = f"traj_{pinn_file}"
    generate_rotating_asteroid_trajectory(
        X0_km_N,
        filename,
        pinn_file,
        timestep=60,
        orbits=3,
    )

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../"
    from Scripts.DataGeneration.Measurements.generate_position_measurements import (
        generate_measurements,
    )

    generate_measurements(f"{statOD_dir}Data/Trajectories/{filename}.data")
