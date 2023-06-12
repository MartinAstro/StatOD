# A class called Dataset that stores the training data for the PINN GM.
# This includes the position, X_BC, and the accelerations, Y.
# The class should stores these as attributes, and have the ability to
# append arbitrary data to the training data. There should also be methods
# which take radii bounds and generate random data using RandomDist trajectory
# object.
import numpy as np
from GravNN.GravityModels.PointMass import PointMass
from GravNN.Trajectories.RandomDist import RandomDist

from StatOD.utils import compute_BN


class Dataset:
    def __init__(self, dim_constants):
        self.clear()
        self.dim_constants = dim_constants

    def clear(self):
        self.X_non_dim = np.empty((0, 3), float)
        self.Y_non_dim = np.empty((0, 3), float)
        self.t_non_dim = np.empty((0,), float)

    def append(self, X, Y, t):
        self.X_non_dim = np.concatenate((self.X_non_dim, X), axis=0)
        self.Y_non_dim = np.concatenate((self.Y_non_dim, Y), axis=0)
        self.t_non_dim = np.concatenate((self.t_non_dim, t), axis=0)

    def dimensionalize(self, X, Y, t):
        X_dim = X * (self.dim_constants["l_star"])
        Y_dim = Y * (self.dim_constants["l_star"] / self.dim_constants["t_star"] ** 2)
        t_dim = t * self.dim_constants["t_star"]
        return X_dim, Y_dim, t_dim

    def non_dimensionalize(self, X, Y, t):
        X_non_dim = X / (self.dim_constants["l_star"])
        Y_non_dim = Y / (
            self.dim_constants["l_star"] / self.dim_constants["t_star"] ** 2
        )
        t_non_dim = t / self.dim_constants["t_star"]
        return X_non_dim, Y_non_dim, t_non_dim

    def get_data_dim(self):
        return self.dimensionalize(
            self.X_non_dim.copy(),
            self.Y_non_dim.copy(),
            self.t_non_dim.copy(),
        )

    def get_data_non_dim(self):
        return self.X_non_dim.copy(), self.Y_non_dim.copy(), self.t_non_dim.copy()

    def set_data_dim(self, X_dim, Y_dim, t_dim):
        self.set_data_non_dim(self.non_dimensionalize(X_dim, Y_dim, t_dim))

    def set_data_non_dim(self, X_non_dim, Y_non_dim, t_non_dim):
        self.X_non_dim = X_non_dim
        self.Y_non_dim = Y_non_dim
        self.t_non_dim = t_non_dim

    def generate_estimated_data(self, state_non_dim, model, t_batch, omega):
        X_N = state_non_dim[:, 0:3]
        Y_DMC_N = state_non_dim[:, 6:9].copy()

        BN = compute_BN(t_batch, omega)
        # X_B = BN @ X_N
        X_B = np.einsum("ijk,ik->ij", BN, X_N)
        Y_m_B = model.compute_acceleration(X_B).reshape((-1, 3))
        Y_m_N = np.einsum("ijk,ik->ij", np.transpose(BN, axes=[0, 2, 1]), Y_m_B)

        # add DMC to PINN predictions
        Y_m_N += Y_DMC_N

        t = t_batch.copy()
        self.append(X_N, Y_m_N, t)

    def generate_BC_data(self, radii_bounds, num_samples, **kwargs):
        planet = kwargs.get("planet")[0]
        grav_file = kwargs.get("grav_file")[0]
        X_BC = RandomDist(
            planet,
            radii_bounds,
            num_samples,
            shape_model=grav_file,
        ).generate()
        Y_BC = PointMass(planet).compute_acceleration(X_BC)

        # convert to km
        X_BC /= 1000
        Y_BC /= 1000

        t_BC = np.zeros((len(X_BC),))
        X_BC_non_dim, Y_BC_non_dim, t_BC_non_dim = self.non_dimensionalize(
            X_BC,
            Y_BC,
            t_BC,
        )

        self.append(X_BC_non_dim, Y_BC_non_dim, t_BC_non_dim)

    def generate_synthetic_data(
        self,
        radii_bounds,
        num_samples,
        model,
        **kwargs,
    ):
        planet = kwargs.get("planet")[0]
        grav_file = kwargs.get("grav_file")[0]
        X_synth = RandomDist(
            planet,
            radii_bounds,
            num_samples,
            shape_model=grav_file,
            uniform=True,
        ).generate()
        Y_synth = model.gravity_model.compute_acceleration(X_synth)

        # convert to km
        X_synth /= 1000
        Y_synth /= 1000

        t_synth = np.zeros((len(X_synth),))
        X_synth_non_dim, Y_synth_non_dim, t_synth_non_dim = self.non_dimensionalize(
            X_synth,
            Y_synth,
            t_synth,
        )
        self.append(X_synth_non_dim, Y_synth_non_dim, t_synth_non_dim)

    def generate_COM_data(
        self,
        X_COM,
        num_samples,
        COM_radius=None,
        **kwargs,
    ):
        X_COM_samples = np.full((num_samples, 3), X_COM)

        if COM_radius is not None:
            X_COM_samples += np.random.normal(0, COM_radius, size=X_COM_samples.shape)

        Y_COM_samples = np.zeros_like(X_COM_samples) + 1e-16
        # 1e-10

        # convert to km
        X_COM_samples /= 1000.0
        Y_COM_samples /= 1000.0

        t_COM = np.zeros((len(X_COM_samples),))
        X_COM_non_dim, Y_COM_non_dim, t_COM_non_dim = self.non_dimensionalize(
            X_COM_samples,
            Y_COM_samples,
            t_COM,
        )
        self.append(X_COM_non_dim, Y_COM_non_dim, t_COM_non_dim)
