
from sympy import *
import numba
import numpy as np
import StatOD
import pickle
import inspect
import os
def process_noise(x, Q, Q_fcn, args, cse_func=cse, use_numba=True):
    n = len(x) # state
    m = len(Q)
    k = len(args)

    # symbolic arguments
    dt = symbols('dt')
    x_args = np.array(symbols('x:'+str(n))) # state
    Q_args = MatrixSymbol("Q", m, m) # Continuous Process Noise
    DCM_args = MatrixSymbol("DCM", m, m)
    misc_args = np.array(symbols('arg:'+str(k)), dtype=np.object)

    # Load or rerun the symbolic expressions
    fcn_name = f"{Q_fcn.__name__}_{m}" 
    dir_name = os.path.dirname(StatOD.__file__) + "/.."
    try:
        # Look for a cached version of the sympy function 
        os.makedirs(f"{dir_name}/.cachedir/process_noise/", exist_ok=True)
        with open(f"{dir_name}/.cachedir/process_noise/{fcn_name}.data", "rb") as f:
            Q_fcn_loaded_src = pickle.load(f)
            Q_sym_loaded = pickle.load(f)

        # Check that the code of the original function hasn't changed
        if inspect.getsource(Q_fcn) == Q_fcn_loaded_src:
            Q_sym = Q_sym_loaded
        else:
            raise ValueError()
    except:
        # If the code has changed, or there wasn't a cached symbolic expression, (re)generate one.
        Q_sym = Q_fcn(dt, x_args, Q_args, DCM_args, misc_args)
        with open(f"{dir_name}/.cachedir/process_noise/{fcn_name}.data", "wb") as f:
            pickle.dump(inspect.getsource(Q_fcn), f)
            pickle.dump(Q_sym, f)

    lambdify_Q = lambdify([dt, x_args, Q_args, DCM_args, misc_args], Q_sym, cse=cse_func, modules='numpy')

    if use_numba:
        Q_func = numba.njit(lambdify_Q, cache=False) # can't cache file b/c it exists within an .egg directory
    else:
        Q_func = lambdify_Q

    # Force JIT compilation so that fcn can be saved using joblib. 
    dt_tmp = 0
    x_tmp = np.arange(0, n, 1)
    Q_tmp = np.zeros((m,m)) # Continuous Process Noise
    DCM_tmp = np.eye(m)
    misc_tmp = np.arange(0,k,1)

    tmp = Q_func(dt_tmp, 
                x_tmp,
                Q_tmp,
                DCM_tmp,
                misc_tmp)

    return Q_func

################
# Noise Models #
################

def get_Q(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N,N)
    for i in range(N // 2):
        A[i, N//2 + i] = 1
    # A[0,3] = 1
    # A[1,4] = 1
    # A[2,5] = 1

    phi = eye(N) + A*dt
    
    # control only influence x_dd
    B = zeros(N,M)
    for i in range(N // 2):
        B[N//2 + i, i] = 1
    # B[3,0] = 1
    # B[4,1] = 1
    # B[5,2] = 1

    integrand = phi*B*DCM*Q*DCM.T*B.T*phi.T

    Q_i_i_m1 = np.zeros((N,N), dtype=np.object)
    for i in range(N): # f[i] differentiated
        for j in range(i, N): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Q_i_i_m1[i,j] = integrated
            Q_i_i_m1[j,i] = integrated
            
            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

def get_Q_original(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N,N)

    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    phi = eye(N) + A*dt
    
    # control only influence x_dd
    B = zeros(N,M)

    B[3,0] = 1
    B[4,1] = 1
    B[5,2] = 1

    integrand = phi*B*DCM*Q*DCM.T*B.T*phi.T

    Q_i_i_m1 = np.zeros((N,N), dtype=np.object)
    for i in range(N): # f[i] differentiated
        for j in range(i, N): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Q_i_i_m1[i,j] = integrated
            Q_i_i_m1[j,i] = integrated
            
            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

def get_Gamma_SRIF(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]

    A = zeros(N,N)
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    phi = eye(N) + A*dt
    
    # control only influence x_dd
    B = zeros(N,M)
    B[3,0] = 1
    B[4,1] = 1
    B[5,2] = 1

    integrand = phi*B#*Q#*B.T*phi.T

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
