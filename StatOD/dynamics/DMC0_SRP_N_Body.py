import numpy as np
from sympy import *

from StatOD.utils import compute_BN

# Frames of Note
# B: Body Fixed Frame
# P: Body Inertial Frame
# N: Sun Inertial


def f_PINN_DMC_HF_zero_order(t, x, args):
    X_sc_P = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_P = args[1:7].astype(float)
    X_sun_P = args[7:10].astype(float)
    A2M = float(args[10])
    radiant_flux = float(args[11])
    mu_sun = float(args[12])
    Cr = float(args[13])
    c = float(args[14])
    # t = float(args[15])
    omega = float(args[16])

    x_pos = X_sc_P[0:3] - X_body_P[0:3]  # either km or [-]
    x_vel = X_sc_P[3:6] - X_body_P[3:6]

    # Gravity
    # scaling occurs within the gravity model
    # come out as non_dim from km + s
    BP = compute_BN(t, omega).squeeze()
    x_pos_B = BP @ x_pos
    x_dd_grav_B = model.compute_acceleration(x_pos_B).reshape((-1,))
    x_dd_grav = BP.T @ x_dd_grav_B

    # 3BP Perturbation
    x_sun_P = X_sun_P[0]
    y_sun_P = X_sun_P[1]
    z_sun_P = X_sun_P[2]

    x_sc_sun = X_sc_P[0] + x_sun_P
    y_sc_sun = X_sc_P[1] + y_sun_P
    z_sc_sun = X_sc_P[2] + z_sun_P

    r_P_sun_mag = np.sqrt(x_sun_P**2 + y_sun_P**2 + z_sun_P**2)
    r_sc_sun_mag = np.sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    a_x_grav_sun = mu_sun * (x_sc_sun / r_sc_sun_mag**3 - x_sun_P / r_P_sun_mag**3)
    a_y_grav_sun = mu_sun * (y_sc_sun / r_sc_sun_mag**3 - y_sun_P / r_P_sun_mag**3)
    a_z_grav_sun = mu_sun * (z_sc_sun / r_sc_sun_mag**3 - z_sun_P / r_P_sun_mag**3)
    # a_x_grav_sun = 0.0
    # a_y_grav_sun = 0.0
    # a_z_grav_sun = 0.0

    # # SRP
    r_sc_sun_mag_m = r_sc_sun_mag * 1e3
    irradiance = radiant_flux / (4 * np.pi * r_sc_sun_mag_m**2)
    P = irradiance / c
    coef = Cr * P * A2M  # SRP in m/s^2
    coef /= 1e3  # SRP in km/s^2
    a_x_SRP = coef * (x_sc_sun / r_sc_sun_mag)
    a_y_SRP = coef * (y_sc_sun / r_sc_sun_mag)
    a_z_SRP = coef * (z_sc_sun / r_sc_sun_mag)
    # a_x_SRP = 0.0
    # a_y_SRP = 0.0
    # a_z_SRP = 0.0

    # Totals
    x_dd = x_dd_grav[0] + a_x_grav_sun + a_x_SRP + w_vec[0]
    y_dd = x_dd_grav[1] + a_y_grav_sun + a_y_SRP + w_vec[1]
    z_dd = x_dd_grav[2] + a_z_grav_sun + a_z_SRP + w_vec[2]

    x_acc = np.hstack((x_dd, y_dd, z_dd))

    w_d = np.zeros_like(w_vec)

    return np.hstack((x_vel, x_acc, w_d))


def dfdx_PINN_DMC_HF_zero_order(t, x, f, args):
    # f argument is needed to make interface standard
    X_sc_P = x[0:6]
    x[6:]

    model = args[0]
    X_body_P = args[1:7].astype(float)
    X_sun_P = args[7:10].astype(float)
    A2M = float(args[10])
    radiant_flux = float(args[11])
    mu_sun = float(args[12])
    Cr = float(args[13])
    c = float(args[14])
    # t = float(args[15])
    omega = float(args[16])

    x_pos = X_sc_P[0:3] - X_body_P[0:3]  # either km or [-]
    BN = compute_BN(t, omega).squeeze()
    x_pos_P = BN @ x_pos
    dfdx_P = model.generate_dadx(x_pos_P).reshape((3, 3))  # [(m/s^2) / m] = [1/s^2]
    dfdx = BN.T @ dfdx_P @ BN

    # Add 3rd Body
    x = X_sc_P[0]
    y = X_sc_P[1]
    z = X_sc_P[2]

    x_s = X_sun_P[0]
    y_s = X_sun_P[1]
    z_s = X_sun_P[2]

    x_sc_sun = x + x_s
    y_sc_sun = y + y_s
    z_sc_sun = z + z_s

    r_sc_s = np.sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    def diag_3BP(i_sc, i_sun, r):
        return mu_sun * (1 / r**3 - 3 * i_sc * (i_sc - i_sun) / r**5)

    def offdiag_3BP(i_sc, j_sc, j_sun, r):
        return mu_sun * -3 * i_sc * (j_sc - j_sun) / r**5

    x_diag = diag_3BP(x, x_s, r_sc_s)
    y_diag = diag_3BP(y, y_s, r_sc_s)
    z_diag = diag_3BP(z, z_s, r_sc_s)
    dfdx_3BP = np.array(
        [
            [x_diag, offdiag_3BP(x, y, y_s, r_sc_s), offdiag_3BP(x, z, z_s, r_sc_s)],
            [offdiag_3BP(y, x, x_s, r_sc_s), y_diag, offdiag_3BP(y, z, z_s, r_sc_s)],
            [offdiag_3BP(z, x, x_s, r_sc_s), offdiag_3BP(z, y, y_s, r_sc_s), z_diag],
        ],
    )
    # dfdx_3BP = np.zeros_like(dfdx_3BP)
    dfdx += dfdx_3BP

    ###############
    # Add SRP
    ###############

    r_sc_sun_mag_m = r_sc_s * 1e3
    irradiance = radiant_flux / (4 * np.pi * r_sc_sun_mag_m**2)

    def diag_SRP(i_sc, i_s, rScS):
        coef = -A2M * Cr * irradiance / c  # m/s^2
        coef /= 1e3  # km/s^2
        return (
            coef
            * (3 * (i_sc - i_s) ** 2 - rScS**2)
            / (rScS**2 * np.sqrt(rScS**2))
        )

    def offdiag_SRP(i_sc, j_sc, i_s, j_s, rScS):
        coef = -3 * A2M * Cr * irradiance / c
        coef /= 1e3  # km/s^2
        return coef * ((i_sc - i_s) * (j_sc - j_s)) / (rScS**2) ** (3 / 2)

    x_diag = diag_SRP(x, x_s, r_sc_s)
    y_diag = diag_SRP(y, y_s, r_sc_s)
    z_diag = diag_SRP(z, z_s, r_sc_s)
    rS = r_sc_s
    dfdx_SRP = np.array(
        [
            [x_diag, offdiag_SRP(x, y, x_s, y_s, rS), offdiag_SRP(x, z, x_s, z_s, rS)],
            [offdiag_SRP(y, x, y_s, x_s, rS), y_diag, offdiag_SRP(y, z, y_s, z_s, rS)],
            [offdiag_SRP(z, x, z_s, x_s, rS), offdiag_SRP(z, y, z_s, y_s, rS), z_diag],
        ],
    )
    # dfdx_SRP = np.zeros_like(dfdx_SRP)
    dfdx += dfdx_SRP

    # Build full dfdx matrix
    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3, 3))
    dfdx = np.block(
        [
            [zero_3x3, dfdx_vel],
            [dfdx, zero_3x3],
        ],
    )

    dfdw = np.zeros((3, 3))
    zeros_6x3 = np.zeros((6, 3))
    intermediate_6x3 = np.zeros((6, 3))
    intermediate_6x3[3:] = np.eye(3)

    dfdz = np.block(
        [
            [dfdx, intermediate_6x3],
            [zeros_6x3.T, dfdw],
        ],
    )
    return dfdz


def get_Q_DMC_HF_zero_order(t, dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    dfdx = args[0]
    f_args = args[1]
    A = dfdx(t, x, None, f_args)

    phi = np.eye(N) + A * dt

    B = np.zeros((N, M))

    B[6, 0] = 1
    B[7, 1] = 1
    B[8, 2] = 1

    Q_i_i_m1 = phi @ B @ DCM.T @ Q @ DCM @ B.T @ phi.T * dt

    return Q_i_i_m1.tolist()


# def get_Q_DMC_HF_zero_order(dt, x, Q, DCM, args):
#     N = len(x)
#     M = Q.shape[0]

#     A = zeros(N, N)

#     # velocities
#     A[0, 3] = 1
#     A[1, 4] = 1
#     A[2, 5] = 1

#     # acceleration is just DMC
#     A[3, 6] = 1
#     A[4, 7] = 1
#     A[5, 8] = 1

#     # TODO: Revisit this. If not commented, it causes the Q
#     # matrix to shrink the covariance instead of increase it.
#     # For now, make the linear approximation used in SNC instead.

#     # A[3:6,0:3] = model.generate_dadx(x[0:3])
#     # A[6,6] = -1/tau
#     # A[7,7] = -1/tau
#     # A[8,8] = -1/tau

#     phi = eye(N) + A * dt
#     # phi = exp(A * dt) # only for LTI

#     B = zeros(N, M)

#     B[6, 0] = 1
#     B[7, 1] = 1
#     B[8, 2] = 1

#     integrand = phi * B * DCM.T * Q * DCM * B.T * phi.T

#     Q_i_i_m1 = np.zeros((N, N), dtype=np.object)
#     for i in range(N):  # f[i] differentiated
#         for j in range(i, N):  # w.r.t. X[j]
#             integrated = integrate(integrand[i, j], dt)
#             Q_i_i_m1[i, j] = integrated
#             Q_i_i_m1[j, i] = integrated

#     # numba can't work with arrays of sympy ints and floats in same matrix
#     # so just force sympy ints to be floats
#     Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
#     Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

#     return Q_i_i_m1.tolist()


def get_DMC_HF_zero_order():
    q_fcn = get_Q_DMC_HF_zero_order
    f_fcn = f_PINN_DMC_HF_zero_order
    dfdx_fcn = dfdx_PINN_DMC_HF_zero_order
    q_args = []
    return f_fcn, dfdx_fcn, q_fcn, q_args
