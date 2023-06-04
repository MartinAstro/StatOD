import os

import numba
import numpy as np
from sympy import *

os.environ["NUMBA_CACHE_DIR"] = "./numba_cache_tmp"



#####################
# Function Wrappers #
#####################
def dfdx(x, f, args):
    m = len(f)
    n = len(x)
    dfdx = np.zeros((m,n), dtype=np.object)

    for i in range(m): # f[i] differentiated
        for j in range(n): # w.r.t. X[j]
            # dfdx[i,j] = simplify(diff(f[i], x[j]))
            dfdx[i,j] = diff(f[i], x[j])
            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    dfdx[np.where(dfdx == 0.0)] = 0.0
    dfdx[np.where(dfdx == 1.0)] = 1.0
    return dfdx.tolist()

def dynamics(x, f, args, cse_func=cse, use_numba=True, consider=None):
    n = len(x) # state
    k = len(args) # non-state arguments

    # n = len(['x', 'y', 'z', 'vx', 'vy', 'vz']) # state
    # k = len(['R', 'mu', 'J2', 'J3']) # non-state arguments

    # symbolic arguments
    t_arg = symbols('t')
    f_args = np.array(symbols('f:'+str(n)))
    x_args = np.array(symbols('x:'+str(n))) # state
    c_args = np.array(symbols('arg:'+str(k))) # parameters

    f_sym = f(t_arg, x_args, c_args)   
    dfdx_sym = dfdx(t_arg, x_args, f_sym, c_args)

    # Define X, R as the inputs to expression
    lambdify_f = lambdify([t_arg, x_args, c_args], f_sym, cse=cse_func, modules='numpy')
    lambdify_dfdx = lambdify([t_arg, x_args, f_args, c_args], dfdx_sym, cse=cse_func, modules='numpy')

    # return func_f, func_dfdx
    if use_numba:
        f_func = numba.njit(lambdify_f, cache=False)
        dfdx_func = numba.njit(lambdify_dfdx, cache=False)
    else:
        f_func = lambdify_f
        dfdx_func = lambdify_dfdx

    t_tmp = 1.0
    x_tmp = np.arange(1,n+1,1) # make values different
    f_tmp = np.arange(2,n+2,1) # to minimize risk of 
    c_tmp = np.arange(3,k+3,1) # div by zero
    f_func(t_tmp, x_tmp, c_tmp)
    dfdx_func(t_tmp, x_tmp, f_tmp, c_tmp)

    # Generate consider dynamics if requested
    if consider is not None:
        assert len(consider) == k # ensure that consider variable is of length args
        consider = np.array(consider).astype(bool)
        c_arg_subset = c_args[consider]
        required_args = np.append(x_args, c_args[~consider])
        dfdc_sym = dfdx(t_arg, c_arg_subset, f_sym, required_args)
        lambdify_dfdc = lambdify([t_arg, c_arg_subset, f_args, required_args], dfdc_sym, cse=cse_func, modules='numpy')
        dfdc_func = numba.njit(lambdify_dfdc, cache=False) if use_numba else lambdify_dfdc
        required_tmp = np.append(t_arg, x_tmp, c_tmp[~consider])
        dfdc_func(t_arg, c_tmp[consider], f_tmp, required_tmp)
        return f_func, dfdx_func, dfdc_func

    return f_func, dfdx_func




# if __name__ == "__main__":
#     import timeit
#     R = 6378.0
#     mu = 398600.4415 
#     J2 = 0.00108263
#     x = np.array([
#         -3515.4903270335103, 8390.716310243395, 4127.627352553683,
#         -4.357676322178153, -3.3565791387645487, 3.111892927869902
#         ])
#     f = f_J2
#     args = np.array([R, mu, J2])
#     f, dfdx = dynamics(x, f, args)

#     f_i = f(x, args)
#     dfdx_i = dfdx(x, f_i, args)

#     print(np.mean(timeit.repeat(lambda : f(x,args), repeat=100, number=1000)))
#     print(np.mean(timeit.repeat(lambda : dfdx(x, f_i, args), repeat=100, number=1000)))

    