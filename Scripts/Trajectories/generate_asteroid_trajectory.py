import numpy as np
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from scipy.integrate import solve_ivp
import pickle
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.CelestialBodies.Asteroids import Eros
from StatOD.utils import ProgressBar

def generate_asteroid_trajectory():
    ep = ErosParams() # max radius is 16 km 

    a = 23 # km 
    n = np.sqrt(ep.mu/a**3)
    T = 2*np.pi/n
    t_f = T*10

    eros = Eros()
    gravity_model = Polyhedral(eros, eros.obj_8k)

    # Generate State in km and km/s
    X0 = np.array([2.40000000e+04, 0.0, 0.0, 
                   0.0, 4.70033081e+00, 4.71606150e-01])/1E3 
    N = len(X0)

    pbar = ProgressBar(t_f, enable=True)
    def f_ivp(t, Z):
        X = Z[0:3].reshape((-1, 3)) * 1E3 # convert to meters
        V = Z[3:6] * 1E3# convert from km/s -> m/s
        A = gravity_model.compute_acceleration(X, pbar=False).reshape((-1)) # m / s^2
        pbar.update(t)
        return np.hstack((V, A)) / 1E3 # convert from m -> km

    t_mesh = np.arange(0, t_f, 60)
    sol = solve_ivp(f_ivp, [0, t_f], X0, atol=1E-12, rtol=1E-12, t_eval=t_mesh)

    data = {
        "t" : sol.t,
        "X" : sol.y[:N, :].T, # in km and km/s
    }
    with open('Data/Trajectories/trajectory_asteroid.data', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    generate_asteroid_trajectory()