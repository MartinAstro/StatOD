import pickle

import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.PointMass import PointMass
from scipy.integrate import solve_ivp
from utils import *

from Scripts.Factories.DynArgsFactory import DynArgsFactory
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from StatOD.utils import ProgressBar, compute_semimajor


def generate_trajectory(X, f, orbits, timestep):
    # compute orbit period
    ep = ErosParams()
    a = compute_semimajor(X0_km_N, ep.mu)
    if np.isnan(a):
        a = 100
    n = np.sqrt(ep.mu / a**3)
    T = 2 * np.pi / n

    # Specify gravity model
    dim_constants = {
        "l_star": 1.0,
        "t_star": 1.0,
        "m_star": 1.0,
    }
    gravity_model = ModelWrapper(PointMass(Eros()), dim_constants)

    # Specify f_args
    f_args = DynArgsFactory().get_HF_args(gravity_model)

    N = len(X) + 3  # for DMC
    phi = np.eye(N).reshape((-1))
    w0 = np.zeros((3,))
    Z0 = np.hstack((X, w0, phi))
    f_fcn, dfdx_fcn, q_fcn, q_args = get_DMC_HF_zero_order()

    # integrate trajectory
    t_f = T * orbits
    pbar = ProgressBar(t_f, enable=True)
    t_mesh = np.arange(0, t_f, step=timestep)

    sol = solve_ivp(
        f_ivp,
        [0, t_f],
        Z0,
        atol=1e-14,
        rtol=1e-14,
        t_eval=t_mesh,
        args=(f_fcn, dfdx_fcn, f_args, N, pbar),
    )

    X_list = sol.y[:N, :].T
    phi = sol.y[N:, :].T.reshape((-1, N, N))
    plot_true_and_estimated_X(X_list, phi)

    data = {
        "t": sol.t,
        "X": X_list,
        "phi": phi,
    }
    statOD_dir = os.path.dirname(StatOD.__file__) + "/.."
    with open(f"{statOD_dir}/Data/Trajectories/{f.__name__}.data", "wb") as f:
        pickle.dump(data, f)

    # plot_trajectory(data["X"])

    test_phi(data, 1)
    test_phi(data, 10)
    test_phi(data, -1)

    plt.show()


if __name__ == "__main__":
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
    generate_trajectory(
        X0_km_N,
        f_PINN_DMC_HF_zero_order,
        timestep=60,
        orbits=1,
    )
