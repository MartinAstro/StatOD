import numpy as np
from StatOD.constants import ErosParams
from StatOD.dynamics import *
from scipy.integrate import solve_ivp
import pickle
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.PointMass import PointMass
from GravNN.CelestialBodies.Asteroids import Eros
from StatOD.utils import ProgressBar, pinnGravityModel
import GravNN
import os

def generate_asteroid_trajectory(X0_km, filename, timestep=30, orbits=2.5):
    ep = ErosParams() # max radius is 16 km 

    a = 30 # km 
    n = np.sqrt(ep.mu/a**3)
    T = 2*np.pi/n
    t_f = T*orbits

    eros = Eros()
    gravity_model = Polyhedral(eros, eros.obj_8k)
    gravity_model_pm = PointMass(eros)
    gravity_model_pinn = pinnGravityModel(os.path.dirname(GravNN.__file__) + \
        # "/../Data/Dataframes/eros_point_mass_v2.data", 
        "/../Data/Dataframes/eros_point_mass_v3.data")

    N = len(X0_km)
    pbar = ProgressBar(t_f, enable=True)
    def f_ivp(t, Z):
        X = Z[0:3].reshape((-1, 3)) * 1E3 # convert to meters
        V = Z[3:6] * 1E3# convert from km/s -> m/s
        A = gravity_model.compute_acceleration(X, pbar=False).reshape((-1)) # m / s^2
        pbar.update(t)
        return np.hstack((V, A)) / 1E3 # convert from m -> km

    t_mesh = np.arange(0, t_f, step=timestep)
    sol = solve_ivp(f_ivp, [0, t_f], X0_km, atol=1E-12, rtol=1E-12, t_eval=t_mesh)

    pos_m = sol.y[0:3, :].T * 1E3
    acc_poly_km = gravity_model.compute_acceleration(pos_m, pbar=False).reshape((-1,3)) / 1E3
    acc_pm_km = gravity_model_pm.compute_acceleration(pos_m).reshape((-1,3)) / 1E3
    acc_pm_km = gravity_model_pm.compute_acceleration(pos_m).reshape((-1,3)) / 1E3
    acc_pinn_km = gravity_model_pinn.generate_acceleration(pos_m).reshape((-1,3))/1E3


    data = {
        "t" : sol.t,
        "X" : sol.y[:N, :].T, # in km and km/s
        "W" : acc_poly_km - acc_pm_km,
        "W_pinn" : acc_poly_km - acc_pinn_km
    }
    with open(f'Data/Trajectories/{filename}.data', 'wb') as f:
        pickle.dump(data, f)



if __name__ == '__main__':

    # Equitorial
    # X0_km = np.array([
    #     2.40000000e+04, 0.0, 0.0, 
    #     0.0, 4.70033081e+00, 4.71606150e-01
    #     ])/1E3 
    # generate_asteroid_trajectory(X0_km, "trajectory_asteroid_equitorial")

    X0_km = np.array([
        3.16800000e+04,0.00000000e+00,0.00000000e+00,
        0.00000000e+00, 3.60365320e+00, 1.11474057e+00
        ])/1E3 
    generate_asteroid_trajectory(X0_km, "trajectory_asteroid_inclined_high_alt_30_timestep", timestep=30, orbits=3)