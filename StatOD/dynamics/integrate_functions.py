import numpy as np
from numba import njit, prange


@njit(cache=False)
def dynamics_ivp(t, Z, f, dfdx, f_args):
    N = int(1 / 2 * (np.sqrt(4 * len(Z) + 1) - 1))
    X_inst = Z[0:N]
    phi_inst = Z[N:].reshape((N, N))

    f_inst = np.array(f(t, X_inst, f_args)).reshape((N))
    dfdx_inst = np.array(dfdx(t, X_inst, f_inst, f_args)).reshape((N, N))

    phi_dot = dfdx_inst @ phi_inst
    Zd = np.hstack((f_inst, phi_dot.reshape((-1))))
    return Zd


# @njit(cache=False)
def dynamics_ivp_no_jit(t, Z, f, dfdx, f_args):
    N = int(1 / 2 * (np.sqrt(4 * len(Z) + 1) - 1))
    X_inst = Z[0:N]
    phi_inst = Z[N:].reshape((N, N))

    f_inst = np.array(f(t, X_inst, f_args)).reshape((N))
    dfdx_inst = np.array(dfdx(t, X_inst, f_inst, f_args)).reshape((N, N))

    phi_dot = dfdx_inst @ phi_inst
    Zd = np.hstack((f_inst, phi_dot.reshape((-1))))
    return Zd


@njit(cache=False)
def consider_dynamics_ivp(t, Z, f, dfdx, dfdc, args, N, M, consider_mask):
    X_inst = Z[0:N]
    C_inst = Z[N : N + M]
    phi_inst = Z[N + M : (N + M + N**2)].reshape((N, N))
    theta_inst = Z[(N + M + N**2) :].reshape((N, M))

    required_args = np.append(X_inst, args[~consider_mask])

    f_inst = np.array(f(t, X_inst, args)).reshape((N))
    dfdx_inst = np.array(dfdx(t, X_inst, f_inst, args)).reshape((N, N))
    dfdc_inst = np.array(dfdc(t, C_inst, f_inst, required_args)).reshape((N, M))

    phi_dot = dfdx_inst @ phi_inst
    theta_dot = dfdx_inst @ theta_inst + dfdc_inst
    Zd = np.hstack(
        (f_inst, np.zeros_like(C_inst), phi_dot.reshape((-1)), theta_dot.reshape((-1))),
    )
    return Zd


@njit(cache=False)
def dynamics_ivp_unscented(t, Z, f, dfdx, f_args):
    L = len(Z)
    N = int(1 / 4.0 * (np.sqrt(8 * L + 1) - 1))
    sigma_points = Z.reshape((2 * N + 1, N))
    Zd = np.zeros((L,))
    for k in range(2 * N + 1):
        X_inst = sigma_points[k]
        f_inst = np.array(f(t, X_inst, f_args)).reshape((N))
        Zd[k * N : (k + 1) * N] = f_inst
    return Zd


def dynamics_ivp_unscented_no_jit(t, Z, f, dfdx, f_args):
    L = len(Z)
    N = int(1 / 4.0 * (np.sqrt(8 * L + 1) - 1))
    sigma_points = Z.reshape((2 * N + 1, N))
    Zd = np.zeros((L,))
    for k in range(2 * N + 1):
        X_inst = sigma_points[k]
        f_inst = np.array(f(t, X_inst, f_args)).reshape((N))
        Zd[k * N : (k + 1) * N] = f_inst
    return Zd


@njit(cache=False, parallel=True)
def dynamics_ivp_particle(t, Z, f, N, f_args):
    X_inst = Z.reshape((N, -1))
    Zd = np.zeros_like(X_inst)
    for i in prange(len(X_inst)):
        Zd[i] = f(X_inst[i], f_args)
    return Zd.reshape(-1)


from StatOD.data import get_earth_position


@njit(cache=False)
def dynamics_ivp_proj2(t, z, f_fcn, dfdx_fcn, f_args):
    N = int(1 / 2 * (np.sqrt(4 * len(z) + 1) - 1))
    x = z[:N]
    phi = z[N:].reshape((N, N))
    J0 = 2456296.25

    r_E = get_earth_position(J0 + t / (24 * 3600))
    f_args[0:3] = r_E
    f = np.array(f_fcn(t, x, f_args)).reshape((N,))
    dfdx = np.array(dfdx_fcn(t, x, f, f_args)).reshape((N, N))

    phi_dot = dfdx @ phi
    z_dot = np.hstack((f, phi_dot.reshape((-1))))
    return z_dot
