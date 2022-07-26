import numpy as np
from StatOD.constants import EarthParams
from StatOD.dynamics import *
from scipy.integrate import solve_ivp


import pickle

def generate_trajectory_J2():
    ep = EarthParams()

    a = 10000
    n = np.sqrt(ep.mu/a**3)
    T = 2*np.pi/n
    
    # Generate State
    X = np.array([-3515.4903270335103, 8390.716310243395, 4127.627352553683,
                  -4.357676322178153, -3.3565791387645487, 3.111892927869902])
    N = len(X)

    # Generate STM
    phi = np.eye(N).reshape((-1))

    # Get dynamics
    f_args = np.array([ep.R, ep.mu, ep.J2])
    f, dfdX = dynamics(X, f_J2, f_args) # Return function

    def f_ivp(t, Z):
        X_inst = Z[0:N]
        phi_inst = Z[N:].reshape((N,N))

        Xd = f(X_inst, f_args)
        phi_dot = dfdX(X_inst, Xd, f_args)@phi_inst

        Zd = np.hstack((Xd, phi_dot.reshape((-1))))
        return Zd

    Z0 =  np.hstack((X, phi))
    t_f = T*15
    t_mesh = np.arange(0, t_f, 10)
    sol = solve_ivp(f_ivp, [0, t_f], Z0, atol=1E-12, rtol=1E-12, t_eval=t_mesh)

    data = {
        "t" : sol.t,
        "X" : sol.y[:N, :].T,
        "phi" : sol.y[N:,:].reshape((-1,N,N))
    }
    with open('Data/Trajectories/trajectory_J2.data', 'wb') as f:
        pickle.dump(data, f)




def generate_trajectory_J3():
    ep = EarthParams()

    a = 10000
    n = np.sqrt(ep.mu/a**3)
    T = 2*np.pi/n
    
    # Generate State
    X = np.array([-3515.4903270335103, 8390.716310243395, 4127.627352553683,
                  -4.357676322178153, -3.3565791387645487, 3.111892927869902])
    N = len(X)

    # Generate STM
    phi = np.eye(N).reshape((-1))

    # Get dynamics
    f_args = np.array([ep.R, ep.mu, ep.J2, ep.J3])
    f, dfdX = dynamics(X, f_J3, f_args) # Return function

    def f_ivp(t, Z):
        X_inst = Z[0:N]
        phi_inst = Z[N:].reshape((N,N))

        Xd = f(X_inst, f_args)
        phi_dot = dfdX(X_inst, Xd, f_args)@phi_inst

        Zd = np.hstack((Xd, phi_dot.reshape((-1))))
        return Zd

    Z0 =  np.hstack((X, phi))
    t_f = T*15
    t_mesh = np.arange(0, t_f, 10)
    sol = solve_ivp(f_ivp, [0, t_f], Z0, atol=1E-12, rtol=1E-12, t_eval=t_mesh)
    
    data = {
        "t" : sol.t,
        "X" : sol.y[:N, :].T,
        "phi" : sol.y[N:,:].reshape((-1,N,N))
    }
    with open('Data/Trajectories/trajectory_J3.data', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    generate_trajectory_J2()
    generate_trajectory_J3()
