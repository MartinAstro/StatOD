from sympy import *
import numpy as np

def get_Q_DMC(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, = args

    A = zeros(N,N)

    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it. 
    # For now, make the linear approximation used in SNC instead.
    
    A[6,6] = -1/tau
    A[7,7] = -1/tau
    A[8,8] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A*dt)
    
    B = zeros(N,M)

    B[6,0] = 1
    B[7,1] = 1
    B[8,2] = 1

    integrand = phi*B*DCM.T*Q*DCM*B.T*phi.T

    Q_i_i_m1 = np.zeros((N,N), dtype=np.object)
    for i in range(N): # f[i] differentiated
        for j in range(i,N): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Q_i_i_m1[i,j] = integrated
            Q_i_i_m1[j,i] = integrated

            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()


# DMC with dynamics
def f_PINN_DMC(x, args):
    X_sc_ECI = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])

    x_pos = (X_sc_ECI[0:3] - X_body_ECI[0:3]) # either km or [-]
    x_vel = (X_sc_ECI[3:6] - X_body_ECI[3:6])

    # scaling occurs within the gravity model 
    x_acc_m = model.generate_acceleration(x_pos).reshape((-1,))

    x_acc = x_acc_m + w_vec
    
    w_d = -1.0/tau*w_vec

    return np.hstack((x_vel, x_acc, w_d))

def dfdx_PINN_DMC(x, f, args):
    # f argument is needed to make interface standard 
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])
    
    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]# either km or [-]

    dfdx_acc_m = model.generate_dadx(x_pos).reshape((3,3)) #[(m/s^2) / m] = [1/s^2]

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3,3))
    dfdx = np.block([[zero_3x3, dfdx_vel],[dfdx_acc_m, zero_3x3]])

    dfdw = np.eye(3) * -1.0/tau
    zeros_6x3 = np.zeros((6,3))
    intermediate_6x3 = np.zeros((6,3))
    intermediate_6x3[3:] = np.eye(3)

    dfdz = np.block(
        [
            [dfdx, intermediate_6x3],
            [zeros_6x3.T, dfdw]
        ]
    )
    return dfdz

def get_Q_PINN_DMC(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, model = args
    
    # Crudest option: Assume that t-t0 ~= 0 s.t. STM = I
    # phi = np.eye(N)
    # B = np.zeros((N,M))
    # B[6:9,0:3] = np.eye(3)
    # integrand = phi@B@DCM.T@Q@DCM@B.T@phi.T
    # Q_i_i_m1 = integrand*dt

    # More accurate option: approximate the STM 
    A = np.zeros((N,N))

    A[0:3, 3:6] = np.eye(3)    # velocities

    # acceleration is DMC + gravity
    A[3:6,0:3] = model.generate_dadx(x[0:3])
    A[3:6,6:9] = np.eye(3)

    # DMC dynamics
    A[6:9,6:9] = -1/tau*np.eye(3)

    # Can't analytically integrate, so approximate STM
    phi = np.eye(N) + A*dt

    B = np.zeros((N,M))
    B[6:9,0:3] = np.eye(3)

    integrand = phi@B@DCM.T@Q@DCM@B.T@phi.T
    Q_i_i_m1 = integrand*dt

    return Q_i_i_m1

def get_DMC_first_order(tau=120):
    q_fcn = get_Q_DMC
    f_fcn = f_PINN_DMC
    dfdx_fcn = dfdx_PINN_DMC
    tau = 120
    return f_fcn, dfdx_fcn, q_fcn, tau

# DMC without dynamics
def f_PINN_DMC_wo_tau(x, args):
    X_sc_ECI = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])

    x_pos = (X_sc_ECI[0:3] - X_body_ECI[0:3]) # either km or [-]
    x_vel = (X_sc_ECI[3:6] - X_body_ECI[3:6])

    # scaling occurs within the gravity model 
    x_acc_m = model.generate_acceleration(x_pos).reshape((-1,))

    x_acc = x_acc_m + w_vec
    
    w_d = np.zeros_like(w_vec)

    return np.hstack((x_vel, x_acc, w_d))

def dfdx_PINN_DMC_wo_tau(x, f, args):
    # f argument is needed to make interface standard 
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])
    
    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]# either km or [-]

    dfdx_acc_m = model.generate_dadx(x_pos).reshape((3,3)) #[(m/s^2) / m] = [1/s^2]

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3,3))
    dfdx = np.block([
        [zero_3x3, dfdx_vel],
        [dfdx_acc_m, zero_3x3]
        ])

    dfdw = np.zeros((3,3))
    zeros_6x3 = np.zeros((6,3))
    intermediate_6x3 = np.zeros((6,3))
    intermediate_6x3[3:] = np.eye(3)

    dfdz = np.block(
        [
            [dfdx, intermediate_6x3],
            [zeros_6x3.T, dfdw]
        ]
    )
    return dfdz

def get_Q_DMC_wo_tau(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, = args

    A = zeros(N,N)

    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it. 
    # For now, make the linear approximation used in SNC instead.
    
    # A[3:6,0:3] = model.generate_dadx(x[0:3])
    # A[6,6] = -1/tau
    # A[7,7] = -1/tau
    # A[8,8] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A*dt)
    
    B = zeros(N,M)

    B[6,0] = 1
    B[7,1] = 1
    B[8,2] = 1

    integrand = phi*B*DCM.T*Q*DCM*B.T*phi.T

    Q_i_i_m1 = np.zeros((N,N), dtype=np.object)
    for i in range(N): # f[i] differentiated
        for j in range(i,N): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Q_i_i_m1[i,j] = integrated
            Q_i_i_m1[j,i] = integrated

            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

def get_Q_DMC_wo_tau_model(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, model = args

    A = np.zeros((N,N))

    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    A[3:6,0:3] = model.generate_dadx(x[0:3])
    # A[6,6] = -1/tau
    # A[7,7] = -1/tau
    # A[8,8] = -1/tau

    phi = np.eye(N) + A*dt
    # phi = exp(A*dt)
    
    B = np.zeros((N,M))

    B[6,0] = 1
    B[7,1] = 1
    B[8,2] = 1

    Q_i_i_m1 = phi@B@DCM.T@Q@DCM@B.T@phi.T*dt

    return Q_i_i_m1

def get_Gamma_SRIF_DMC(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N,N)
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    phi = eye(N) + A*dt
    
    # control only influence x_dd
    B = zeros(N,M)
    B[6,0] = 1
    B[7,1] = 1
    B[8,2] = 1

    integrand = phi*B

    Gamma_i_i_m1 = np.zeros((N,M), dtype=np.object)
    for i in range(len(Gamma_i_i_m1)): # f[i] differentiated
        for j in range(len(Gamma_i_i_m1[0])): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Gamma_i_i_m1[i,j] = integrated            
            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Gamma_i_i_m1[np.where(Gamma_i_i_m1 == 0)] = 0.0
    Gamma_i_i_m1[np.where(Gamma_i_i_m1 == 1)] = 1.0

    return Gamma_i_i_m1.tolist()

def get_DMC_zero_order():
    q_fcn = get_Q_DMC_wo_tau
    f_fcn = f_PINN_DMC_wo_tau
    dfdx_fcn = dfdx_PINN_DMC_wo_tau
    tau = 0
    return f_fcn, dfdx_fcn, q_fcn, tau
