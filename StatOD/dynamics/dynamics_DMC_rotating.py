import numpy as np
from sympy import *

from Scripts.Scenarios.helper_functions import compute_BN


def get_Q_DMC(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    (tau,) = args

    A = zeros(N, N)

    # velocities
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1

    # acceleration is just DMC
    A[3, 6] = 1
    A[4, 7] = 1
    A[5, 8] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it.
    # For now, make the linear approximation used in SNC instead.

    A[6, 6] = -1 / tau
    A[7, 7] = -1 / tau
    A[8, 8] = -1 / tau

    # phi = eye(N) + A*dt
    phi = exp(A * dt)

    B = zeros(N, M)

    B[6, 0] = 1
    B[7, 1] = 1
    B[8, 2] = 1

    integrand = phi * B * DCM.T * Q * DCM * B.T * phi.T

    Q_i_i_m1 = np.zeros((N, N), dtype=np.object)
    for i in range(N):  # f[i] differentiated
        for j in range(i, N):  # w.r.t. X[j]
            integrated = integrate(integrand[i, j], dt)
            Q_i_i_m1[i, j] = integrated
            Q_i_i_m1[j, i] = integrated

    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()


# DMC without dynamics
def f_rot_PINN_DMC_zero_order(x, args):
    X_sc_ECI = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    omega = float(args[-1])

    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]  # either km or [-]
    x_vel = X_sc_ECI[3:6] - X_body_ECI[3:6]

    # scaling occurs within the gravity model
    x_acc_m = model.compute_acceleration(x_pos).reshape((-1,))

    x, y, z = x_pos
    x_d, y_d, z_d = x_vel

    x_dd = x_acc_m[0] - omega**2 * x - 2 * omega * y_d + w_vec[0]
    y_dd = x_acc_m[1] - omega**2 * y + 2 * omega * x_d + w_vec[1]
    z_dd = x_acc_m[2] + w_vec[2]

    x_acc = np.hstack((x_dd, y_dd, z_dd))

    w_d = np.zeros_like(w_vec)

    return np.hstack((x_vel, x_acc, w_d))


def dfdx_rot_PINN_DMC_zero_order(x, f, args):
    # f argument is needed to make interface standard
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    omega = float(args[-1])

    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]  # either km or [-]

    f_x = model.generate_dadx(x_pos).reshape((3, 3))  # [(m/s^2) / m] = [1/s^2]

    f_x_rotation = np.zeros((3, 3))
    f_x_rotation[0, 0] = -(omega**2)
    f_x_rotation[1, 1] = -(omega**2)

    f_x += f_x_rotation

    f_x_d = np.zeros((3, 3))
    f_x_d[0, 1] = -2 * omega
    f_x_d[1, 0] = +2 * omega

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3, 3))
    dfdx = np.block(
        [
            [zero_3x3, dfdx_vel],
            [f_x, f_x_d],
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


##################
# Inertial Frame #
##################

# DMC without dynamics
def f_rot_N_PINN_DMC_zero_order(x, args):
    X_sc_ECI = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_ECI = args[1:-2].astype(float)

    t = float(args[-2])
    omega = float(args[-1])

    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]  # either km or [-]
    x_vel = X_sc_ECI[3:6] - X_body_ECI[3:6]

    # scaling occurs within the gravity model
    BN = compute_BN(t, omega).squeeze()
    x_pos_B = BN @ x_pos
    x_acc_m_B = model.compute_acceleration(x_pos_B).reshape((-1,))
    x_acc_m = BN.T @ x_acc_m_B

    x_dd = x_acc_m[0] + w_vec[0]
    y_dd = x_acc_m[1] + w_vec[1]
    z_dd = x_acc_m[2] + w_vec[2]

    x_acc = np.hstack((x_dd, y_dd, z_dd))

    w_d = np.zeros_like(w_vec)

    return np.hstack((x_vel, x_acc, w_d))


def dfdx_rot_N_PINN_DMC_zero_order(x, f, args):
    # f argument is needed to make interface standard
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:-2].astype(float)
    t = float(args[-2])
    omega = float(args[-1])

    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]  # either km or [-]
    BN = compute_BN(t, omega).squeeze()
    x_pos_B = BN @ x_pos
    f_x_B = model.generate_dadx(x_pos_B).reshape((3, 3))  # [(m/s^2) / m] = [1/s^2]
    f_x = BN.T @ f_x_B @ BN

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3, 3))
    dfdx = np.block(
        [
            [zero_3x3, dfdx_vel],
            [f_x, zero_3x3],
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


def get_Q_DMC_zero_order(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N, N)

    # velocities
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1

    # acceleration is just DMC
    A[3, 6] = 1
    A[4, 7] = 1
    A[5, 8] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it.
    # For now, make the linear approximation used in SNC instead.

    # A[3:6,0:3] = model.generate_dadx(x[0:3])
    # A[6,6] = -1/tau
    # A[7,7] = -1/tau
    # A[8,8] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A * dt)

    B = zeros(N, M)

    B[6, 0] = 1
    B[7, 1] = 1
    B[8, 2] = 1

    integrand = phi * B * DCM.T * Q * DCM * B.T * phi.T

    Q_i_i_m1 = np.zeros((N, N), dtype=np.object)
    for i in range(N):  # f[i] differentiated
        for j in range(i, N):  # w.r.t. X[j]
            integrated = integrate(integrand[i, j], dt)
            Q_i_i_m1[i, j] = integrated
            Q_i_i_m1[j, i] = integrated

    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

def get_Q_DMC_zero_order_paper(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N, N)

    # velocities
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it.
    # For now, make the linear approximation used in SNC instead.

    # A[3:6,0:3] = model.generate_dadx(x[0:3])
    # A[6,6] = -1/tau
    # A[7,7] = -1/tau
    # A[8,8] = -1/tau

    phi = eye(N) - A*dt

    B = zeros(N, M)

    B[6, 0] = 1
    B[7, 1] = 1
    B[8, 2] = 1

    integrand = phi * B * Q * B.T * phi.T

    Q_i_i_m1 = np.zeros((N, N), dtype=np.object)
    for i in range(N):  # f[i] differentiated
        for j in range(i, N):  # w.r.t. X[j]
            integrated = integrate(integrand[i, j], dt)
            Q_i_i_m1[i, j] = integrated
            Q_i_i_m1[j, i] = integrated

    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

def get_Q_DMC_zero_order_approx(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    Q_i_i_m1 = np.zeros((N, N), dtype=np.object)
    Q_i_i_m1[-M:, -M:] = Q*dt
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()


def get_rot_DMC_zero_order():
    # q_fcn = get_Q_DMC_zero_order
    # q_fcn = get_Q_DMC_zero_order_approx
    q_fcn = get_Q_DMC_zero_order_paper
    f_fcn = f_rot_N_PINN_DMC_zero_order
    dfdx_fcn = dfdx_rot_N_PINN_DMC_zero_order
    q_args = []
    return f_fcn, dfdx_fcn, q_fcn, q_args
