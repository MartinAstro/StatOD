import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_sym_model
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

        X_dim_m = X_dim.reshape((-1, 3)) * 1e3
        A_dim_ms2 = self.model.compute_acceleration(X_dim_m)
        A_dim_km_s2 = A_dim_ms2 / 1e3
        A_non_dim = A_dim_km_s2 / ms2

        return A_non_dim

    def generate_dadx(self, X):
        t_star = self.dim_constants["t_star"]
        X_dim = X * self.dim_constants["l_star"]

        c = 1.0 / t_star**2
        return self.model.compute_dfdx(X_dim.reshape((-1, 3)) * 1e3) / c


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
    dim_constants = {"t_star": 1.0, "l_star": 1.0}
    eros = Eros()
    gravity_model_true = generate_heterogeneous_sym_model(eros, eros.obj_8k)
    gravity_model_true = ModelWrapper(gravity_model_true, dim_constants)
    # gravity_model_true = Polyhedral(eros, eros.obj_8k)

    # integrate trajectory
    t_f = T * orbits
    pbar = ProgressBar(t_f, enable=True)

    f_args = DynArgsFactory().get_HF_args(gravity_model_true)
    f, dfdx, q, q_args = get_DMC_HF_zero_order()
    f_integrate = dynamics_ivp_f_only

    w0 = np.array([0.0, 0.0, 0.0])
    Z0_km = np.hstack((X0_km_N, w0))

    t_mesh = np.arange(0, t_f, step=timestep)
    sol = solve_ivp(
        f_integrate,
        [0, t_f],
        Z0_km,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_mesh,
        args=(f, dfdx, f_args, pbar),
    )

    # compute body frame accelerations along trajectory
    R_N_km = sol.y[0:3, :].T
    BN = compute_BN(sol.t, ep.omega)
    R_B_km = np.einsum("ijk,ik->ij", BN, R_N_km)
    R_B_m = R_B_km * 1e3

    # You have to pass in km to the true model
    acc_grav_only_B_km = gravity_model_true.compute_acceleration(R_B_km).reshape(
        (-1, 3),
    )
    acc_grav_only_B_m = acc_grav_only_B_km * 1e3

    # compute body frame accelerations along trajectory using other models
    statOD_dir = os.path.dirname(StatOD.__file__)
    gravity_model_pinn = pinnGravityModel(
        f"{statOD_dir}/../Data/Dataframes/{model_file}.data",
    )

    acc_pinn_B_m = gravity_model_pinn.compute_acceleration(R_B_m).reshape((-1, 3))

    Z_i = sol.y.T
    Zd = np.array(
        [
            f_integrate(t_mesh[i], Z_i[i], f, dfdx, f_args, pbar)
            for i in range(len(t_mesh))
        ],
    )
    acc_true_N_km = Zd[:, 3:6]

    # compute inertial frame accelerations along trajectory
    NB = np.transpose(BN, axes=[0, 2, 1])
    acc_grav_only_N_m = np.einsum("ijk,ik->ij", NB, acc_grav_only_B_m)
    acc_pinn_N_m = np.einsum("ijk,ik->ij", NB, acc_pinn_B_m)
    acc_true_B_km = np.einsum("ijk,ik->ij", BN, acc_true_N_km)

    # convert to kilometers
    R_B_km = R_B_m / 1e3

    acc_grav_only_N_km = acc_grav_only_N_m / 1e3
    acc_pinn_N_km = acc_pinn_N_m / 1e3
    acc_grav_only_B_km = acc_grav_only_B_m / 1e3
    acc_pinn_B_km = acc_pinn_B_m / 1e3

    N = len(X0_km_N)
    data = {
        "t": sol.t,
        "X": sol.y[:N, :].T,  # state in km and km/s
        "X_B": R_B_km,  # position in km
        "A_B": acc_grav_only_B_km,  # hetero model acceleration in km/s^2
        "A_true_B": acc_true_B_km,  # acceleration in km/s^2
        # # true unmodeled acceleration in km/s^2
        "W": acc_true_N_km - acc_grav_only_N_km,
        # # estimated unmodeled acceleration in km/s^2
        "W_pinn": acc_true_N_km - acc_pinn_N_km,
    }

    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    data
    acc_pinn_B_km

    idx = -1
    print(acc_true_B_km[idx])
    print(acc_grav_only_B_km[idx])
    print(acc_pinn_B_km[idx])

    with open(f"{statOD_dir}/Data/Trajectories/{filename}.data", "wb") as f:
        pickle.dump(data, f)


def get_elliptic_state(pinn_file):
    # Julio's parameters but e = 0.1
    X0_m_N = np.array(
        [
            -24336.171875,
            19408.798828125,
            13827.63671875,
            -2.717930555343628,
            -1.3936699628829956,
            -1.9365609884262085,
        ],
    )
    filename = f"traj_{pinn_file}_elliptic"
    return X0_m_N, filename


def get_circular_state(pinn_file):
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
    filename = f"traj_{pinn_file}"
    return X0_m_N, filename


def main(pinn_file):
    # X0_m_N, filename = get_circular_state(pinn_file)
    # X0_m_N, filename = get_elliptic_state(pinn_file)

    # Load in the initial conditions
    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    with open(f"{statOD_dir}/Data/InitialConditions/ICs.data", "rb") as f:
        ICs = pickle.load(f)

    idx = 0
    X0_m_N = np.array(ICs[idx]["X"])
    a = ICs[idx]["a"]
    e = ICs[idx]["e"]

    filename = f"traj_{pinn_file}_{a}_{e}"

    X0_km_N = X0_m_N / 1e3

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


if __name__ == "__main__":
    pinn_file = "eros_pm_053123"
    pinn_file = "eros_pm_061023"
    pinn_file = "eros_poly_061023"
    # pinn_file = "eros_poly_061023_dropout"
    main(pinn_file)
