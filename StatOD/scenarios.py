import time
from abc import abstractclassmethod

import numpy as np

from StatOD.datasets import Dataset
from StatOD.dynamics import (
    dynamics,
    dynamics_ivp_no_jit,
    dynamics_ivp_unscented_no_jit,
    process_noise,
)
from StatOD.filters import FilterLogger
from StatOD.measurements import measurements
from StatOD.utils import *


class ScenarioBaseClass:
    def __init__(self, config):
        self.N_states = config["N_states"][0]
        self.dim_constants = config["dim_constants"][0]
        self.model = config["model"][0]

        self.t_star = self.dim_constants["t_star"]
        self.l_star = self.dim_constants["l_star"]
        self.ms = self.dim_constants["l_star"] / self.dim_constants["t_star"]
        self.ms2 = self.dim_constants["l_star"] / self.dim_constants["t_star"] ** 2

    def initializeMeasurements(self, t_vec, Y_vec, R, h_fcn, h_args_vec):
        if len(R.shape) == 1:  # if diag
            R_matrix_vec = np.repeat([np.diag(R)], len(t_vec), axis=0)
        elif len(R.shape) == 2:  # if matrix
            R_matrix_vec = np.repeat([R], len(t_vec), axis=0)
        elif len(R.shape) == 3:  # if tensor
            R_matrix_vec = R
        else:
            raise ValueError(
                f"Argument R shape is not a compatible dimension: R.shape = {R.shape}",
            )

        self.t = t_vec
        self.Y = Y_vec
        self.R = R_matrix_vec

        z0 = np.zeros((self.N_states,))
        h, dhdx = measurements(z0, h_fcn, h_args_vec[0])

        self.h_args = h_args_vec
        self.h_fcn = h
        self.dhdx_fcn = dhdx

    def initializeDynamics(self, f_fcn, dfdx_fcn, f_args):
        try:  # if sympy, convert to lambdas
            z0 = np.zeros((self.N_states,))
            f_fcn, dfdx_fcn = dynamics(z0, f_fcn, f_args)
        except:
            pass

        self.f_fcn = f_fcn
        self.dfdx_fcn = dfdx_fcn
        self.f_args = f_args
        pass

    def initializeNoise(self, q_fcn, q_args, Q0, Q_dt):
        self.Q0 = Q0

        try:  # if sympy
            z0 = np.zeros((self.N_states,))
            q_fcn = process_noise(z0, Q0, q_fcn, q_args, use_numba=False)
        except Exception as e:
            q_fcn = q_fcn
            print(e)
            pass

        self.q_fcn = q_fcn
        self.q_args = q_args
        self.Q_dt = Q_dt

    def initializeIC(self, t0, x0, P0, dx0=None):
        self.t0 = t0
        self.x0 = x0
        self.P0 = P0
        self.dx0 = dx0

        if len(self.P0.shape) == 1:  # if diag
            self.P0 = np.diag(self.P0)

    def initializeFilter(self, filter_class):
        self.filter_type = filter_class.__name__

        f_dict = {
            "f": self.f_fcn,
            "dfdx": self.dfdx_fcn,
            "f_args": self.f_args,
            "Q_fcn": self.q_fcn,
            "Q": self.Q0,
            "Q_args": self.q_args,
            "Q_dt": self.Q_dt,
        }

        h_dict = {
            "h": self.h_fcn,
            "dhdx": self.dhdx_fcn,
            "h_args": self.h_args,
        }

        logger = FilterLogger(len(self.x0), len(self.t))
        filter = filter_class(
            self.t0,
            self.x0,
            self.dx0,
            self.P0,
            f_dict,
            h_dict,
            logger=logger,
        )

        if self.filter_type == "UnscentedKalmanFilter":
            filter.f_integrate = dynamics_ivp_unscented_no_jit
        else:
            filter.f_integrate = dynamics_ivp_no_jit

        self.filter = filter
        pass

    @abstractclassmethod
    def transform(self, function):
        pass

    def dimensionalize(self):
        self.transform(lambda a, b: a * b)

    def non_dimensionalize(self):
        self.transform(lambda a, b: a / b)

    def run_callbacks(self, callbacks, t_i):
        for callback in callbacks.values():
            callback(self.model.gravity_model, t_i)

    def run(self, train_config):
        train = train_config.get("train", True)
        batch_size = train_config.get("batch_size")
        meas_batch_size = train_config.get("meas_batch_size", batch_size)
        train_config.get("epochs", 32)
        BC_data = train_config.get("BC_data", False)
        rotating = train_config.get("rotating", False)
        rotating_fcn = train_config.get("rotating_fcn", None)
        synthetic_data = train_config.get("synthetic_data", False)
        empty_data = train_config.get("empty_data", False)
        COM_data = train_config.get("COM_samples", 0)
        callbacks = train_config.get("callbacks", {})
        intermediate_callbacks = train_config.get("intermediate_callbacks", False)
        start_time = time.time()

        train_idx_list = []
        total_batches = len(self.Y) // meas_batch_size
        self.model.train_idx = 0
        data = Dataset(self.dim_constants)
        planet = self.model.planet

        # run initial callbacks / metrics
        # self.run_callbacks(callbacks, 0.0)

        for k in range(total_batches + 1):
            # Gather measurements in batch
            start_idx = k * meas_batch_size
            end_idx = (
                None if (k + 1) * meas_batch_size >= len(self.Y) else (k + 1) * meas_batch_size
            )
            t_batch = self.t[start_idx:end_idx]
            Y_batch = self.Y[start_idx:end_idx]
            R_batch = self.R[start_idx:end_idx]
            f_args_batch = self.f_args[start_idx:end_idx]
            h_args_batch = self.h_args[start_idx:end_idx]

            # usee latest trained model to f_args
            f_args_batch[:, 0] = self.model

            # run the filter on the batch of measurements
            self.filter.run(t_batch, Y_batch, R_batch, f_args_batch, h_args_batch)

            # collect network training data
            if empty_data:
                data.clear()

            omega = f_args_batch[0, -1]
            state = self.filter.logger.x_hat_i_plus[start_idx:end_idx]
            data.generate_estimated_data(state, self.model, t_batch, omega)

            if BC_data:
                data.generate_BC_data(
                    train_config.get("bc_radii", np.array([5, 10])) * planet.radius,
                    train_config.get("num_samples", 1000),
                    **self.model.config,
                )
            if synthetic_data:
                data.generate_synthetic_data(
                    train_config.get("syn_radii", np.array([0, 3])) * planet.radius,
                    train_config.get("num_samples", 1000),
                    self.model,
                    **self.model.config,
                )
            if COM_data:
                data.generate_COM_data(
                    train_config.get("X_COM"),
                    train_config.get("COM_samples", 1),
                    train_config.get("COM_radius", 0),
                )

            # rotate the values based on spin of asteroid
            if rotating:
                X_non_dim, Y_non_dim, t_non_dim = data.get_data_non_dim()
                X_non_dim, Y_non_dim = rotating_fcn(
                    t_non_dim,
                    omega,
                    X_non_dim,
                    Y_non_dim,
                )
                data.set_data_non_dim(X_non_dim, Y_non_dim, t_non_dim)

            X_train_dim, Y_train_dim, t_train_dim = data.get_data_dim()

            # Don't train on the last batch of data if it's too small
            if train:  # k != total_batches:
                self.model.train(
                    X_train_dim * 1e3,  # convert to m
                    Y_train_dim * 1e3,  # convert to m/s^2
                    **train_config,
                )
                self.model.train_idx += 1
                train_idx_list.append(end_idx)

            # After the model is trained, run the callbacks
            if intermediate_callbacks:
                self.run_callbacks(callbacks, t_batch[-1])

        if len(train_idx_list) > 0:
            train_idx_list.pop()
        self.time_elapsed = time.time() - start_time
        self.train_idx_list = train_idx_list

        # run callbacks at final step
        self.run_callbacks(callbacks, self.t[-1])

        self.filter.logger.compute_phi_ti_t0()

        print(f"Filter Time Elapsed: {self.time_elapsed}")
        print(f"Total Time Elapsed: {time.time() - start_time}")

        return callbacks

    def save(self):
        pass


class ScenarioPositions(ScenarioBaseClass):
    def __init__(self, config):
        super().__init__(config)

    def transform(self, function):
        # Define dimensionalization constants
        t_star = self.t_star
        l_star = self.l_star
        ms = self.ms
        ms2 = self.ms2

        # transform initial parameters
        self.x0[0:3] = function(self.x0[0:3], self.l_star)
        self.x0[3:6] = function(self.x0[3:6], self.ms)
        self.x0[6:9] = function(self.x0[6:9], self.ms2)

        self.P0[0:3, 0:3] = function(self.P0[0:3, 0:3], self.l_star**2)
        self.P0[3:6, 3:6] = function(self.P0[3:6, 3:6], self.ms**2)
        self.P0[6:9, 6:9] = function(self.P0[6:9, 6:9], self.ms2**2)

        self.t = function(self.t, t_star)
        self.Y = function(self.Y, l_star)
        self.R = function(self.R, l_star**2)
        self.Q0 = function(self.Q0, self.ms2**2)
        self.Q_dt = function(self.Q_dt, self.t_star)
        self.f_args[:, -2] = function(self.f_args[:, -2], self.t_star)  # t_i
        self.f_args[:, -1] = function(
            self.f_args[:, -1],
            (1 / self.t_star),
        )  # omega [rad/s]

        try:
            self.tau = function(self.tau, self.ms2**2)
        except:
            pass

        try:
            # transform the logger if the filter is available
            self.filter.logger.t_i = function(self.filter.logger.t_i, t_star)

            self.filter.logger.x_hat_i_plus[:, 0:3] = function(
                self.filter.logger.x_hat_i_plus[:, 0:3],
                l_star,
            )
            self.filter.logger.x_hat_i_plus[:, 3:6] = function(
                self.filter.logger.x_hat_i_plus[:, 3:6],
                ms,
            )
            self.filter.logger.x_hat_i_plus[:, 6:9] = function(
                self.filter.logger.x_hat_i_plus[:, 6:9],
                ms2,
            )

            self.filter.logger.P_i_plus[:, 0:3, 0:3] = function(
                self.filter.logger.P_i_plus[:, 0:3, 0:3],
                l_star**2,
            )
            self.filter.logger.P_i_plus[:, 3:6, 3:6] = function(
                self.filter.logger.P_i_plus[:, 3:6, 3:6],
                ms**2,
            )
            self.filter.logger.P_i_plus[:, 6:9, 6:9] = function(
                self.filter.logger.P_i_plus[:, 6:9, 6:9],
                ms2**2,
            )
        except:
            pass


class ScenarioRangeRangeRate(ScenarioBaseClass):
    def __init__(self, config):
        super().__init__(config)

    def transform(self, function):
        # Define dimensionalization constants
        t_star = self.t_star
        l_star = self.l_star
        ms = self.ms
        ms2 = self.ms2

        # transform initial parameters
        self.x0[0:3] = function(self.x0[0:3], self.l_star)
        self.x0[3:6] = function(self.x0[3:6], self.ms)
        self.x0[6:9] = function(self.x0[6:9], self.ms2)

        self.P0[0:3, 0:3] = function(self.P0[0:3, 0:3], self.l_star**2)
        self.P0[3:6, 3:6] = function(self.P0[3:6, 3:6], self.ms**2)
        self.P0[6:9, 6:9] = function(self.P0[6:9, 6:9], self.ms2**2)

        self.t = function(self.t, t_star)
        self.Y[:, 0] = function(self.Y[:, 0], l_star)
        self.Y[:, 1] = function(self.Y[:, 1], ms)
        self.R[:, 0, 0] = function(self.R[:, 0, 0], l_star**2)
        self.R[:, 1, 1] = function(self.R[:, 1, 1], ms**2)
        self.Q0 = function(self.Q0, self.ms2**2)
        self.Q_dt = function(self.Q_dt, self.t_star)

        try:
            self.tau = function(self.tau, self.ms2**2)
        except:
            pass

        try:
            # transform the logger if the filter is available
            self.filter.logger.t_i = function(self.filter.logger.t_i, t_star)

            self.filter.logger.x_hat_i_plus[:, 0:3] = function(
                self.filter.logger.x_hat_i_plus[:, 0:3],
                l_star,
            )
            self.filter.logger.x_hat_i_plus[:, 3:6] = function(
                self.filter.logger.x_hat_i_plus[:, 3:6],
                ms,
            )
            self.filter.logger.x_hat_i_plus[:, 6:9] = function(
                self.filter.logger.x_hat_i_plus[:, 6:9],
                ms2,
            )

            self.filter.logger.P_i_plus[:, 0:3, 0:3] = function(
                self.filter.logger.P_i_plus[:, 0:3, 0:3],
                l_star**2,
            )
            self.filter.logger.P_i_plus[:, 3:6, 3:6] = function(
                self.filter.logger.P_i_plus[:, 3:6, 3:6],
                ms**2,
            )
            self.filter.logger.P_i_plus[:, 6:9, 6:9] = function(
                self.filter.logger.P_i_plus[:, 6:9, 6:9],
                ms2**2,
            )
        except:
            pass


class ScenarioHF(ScenarioBaseClass):
    def __init__(self, config):
        super().__init__(config)

    def transform(self, function):
        # Define dimensionalization constants
        t_star = self.t_star
        l_star = self.l_star
        ms = self.ms
        ms2 = self.ms2

        # transform initial parameters
        self.x0[0:3] = function(self.x0[0:3], self.l_star)
        self.x0[3:6] = function(self.x0[3:6], self.ms)
        self.x0[6:9] = function(self.x0[6:9], self.ms2)

        self.P0[0:3, 0:3] = function(self.P0[0:3, 0:3], self.l_star**2)
        self.P0[3:6, 3:6] = function(self.P0[3:6, 3:6], self.ms**2)
        self.P0[6:9, 6:9] = function(self.P0[6:9, 6:9], self.ms2**2)

        self.t = function(self.t, t_star)
        self.Y = function(self.Y, l_star)
        self.R = function(self.R, l_star**2)
        self.Q0 = function(self.Q0, self.ms2**2)
        self.Q_dt = function(self.Q_dt, self.t_star)

        # Arguments
        # self.f_args[:, 0] = function(self.f_args[:,0], self.t_star) # gravity_model,
        self.f_args[:, 1:4] = function(self.f_args[:, 1:4], l_star)  # eros_pos_P,
        self.f_args[:, 4:7] = function(self.f_args[:, 4:7], ms)  # eros_vel_P,
        self.f_args[:, 7:10] = function(self.f_args[:, 7:10], l_star)  # sun_pos_P [km],

        # area_2_mass [m^2/kg],
        self.f_args[:, 10] = function(self.f_args[:, 10], l_star**2)

        # radiant_flux [W] = [kg m^2 / s^3],
        self.f_args[:, 11] = function(self.f_args[:, 11], l_star**2 / t_star**3)

        # sunParams.mu_sun,
        self.f_args[:, 12] = function(
            self.f_args[:, 12],
            self.l_star**3 / self.t_star**2,
        )
        self.f_args[:, 13] = function(self.f_args[:, 13], 1.0)  # Cr,
        self.f_args[:, 14] = function(self.f_args[:, 14], ms)  # sunParams.c [m/s],
        self.f_args[:, 15] = function(self.f_args[:, 15], t_star)  # time,
        self.f_args[:, 16] = function(
            self.f_args[:, 16],
            1 / self.t_star,
        )  # ErosParams().omega,

        try:
            self.tau = function(self.tau, ms2**2)
        except:
            pass

        try:
            # transform the logger if the filter is available
            self.filter.logger.t_i = function(self.filter.logger.t_i, t_star)

            self.filter.logger.x_hat_i_plus[:, 0:3] = function(
                self.filter.logger.x_hat_i_plus[:, 0:3],
                l_star,
            )
            self.filter.logger.x_hat_i_plus[:, 3:6] = function(
                self.filter.logger.x_hat_i_plus[:, 3:6],
                ms,
            )
            self.filter.logger.x_hat_i_plus[:, 6:9] = function(
                self.filter.logger.x_hat_i_plus[:, 6:9],
                ms2,
            )

            self.filter.logger.P_i_plus[:, 0:3, 0:3] = function(
                self.filter.logger.P_i_plus[:, 0:3, 0:3],
                l_star**2,
            )
            self.filter.logger.P_i_plus[:, 3:6, 3:6] = function(
                self.filter.logger.P_i_plus[:, 3:6, 3:6],
                ms**2,
            )
            self.filter.logger.P_i_plus[:, 6:9, 6:9] = function(
                self.filter.logger.P_i_plus[:, 6:9, 6:9],
                ms2**2,
            )
        except:
            pass


class ScenarioHFSymb(ScenarioBaseClass):
    def __init__(self, config):
        super().__init__(config)

    def transform(self, function):
        # Define dimensionalization constants
        t_star = self.t_star
        l_star = self.l_star
        ms = self.ms
        ms2 = self.ms2

        # transform initial parameters
        self.x0[0:3] = function(self.x0[0:3], self.l_star)
        self.x0[3:6] = function(self.x0[3:6], self.ms)
        self.x0[6:9] = function(self.x0[6:9], self.ms2)

        self.P0[0:3, 0:3] = function(self.P0[0:3, 0:3], self.l_star**2)
        self.P0[3:6, 3:6] = function(self.P0[3:6, 3:6], self.ms**2)
        self.P0[6:9, 6:9] = function(self.P0[6:9, 6:9], self.ms2**2)

        self.t = function(self.t, t_star)
        self.Y = function(self.Y, l_star)
        self.R = function(self.R, l_star**2)
        self.Q0 = function(self.Q0, self.ms2**2)
        self.Q_dt = function(self.Q_dt, self.t_star)

        # Arguments
        self.f_args[:, 0] = function(
            self.f_args[:, 0],
            self.l_star**3 / self.t_star**2,
        )  # gravity_model,

        try:
            self.tau = function(self.tau, self.ms2**2)
        except:
            pass

        try:
            # transform the logger if the filter is available
            self.filter.logger.t_i = function(self.filter.logger.t_i, t_star)

            self.filter.logger.x_hat_i_plus[:, 0:3] = function(
                self.filter.logger.x_hat_i_plus[:, 0:3],
                l_star,
            )
            self.filter.logger.x_hat_i_plus[:, 3:6] = function(
                self.filter.logger.x_hat_i_plus[:, 3:6],
                ms,
            )
            self.filter.logger.x_hat_i_plus[:, 6:9] = function(
                self.filter.logger.x_hat_i_plus[:, 6:9],
                ms2,
            )

            self.filter.logger.P_i_plus[:, 0:3, 0:3] = function(
                self.filter.logger.P_i_plus[:, 0:3, 0:3],
                l_star**2,
            )
            self.filter.logger.P_i_plus[:, 3:6, 3:6] = function(
                self.filter.logger.P_i_plus[:, 3:6, 3:6],
                ms**2,
            )
            self.filter.logger.P_i_plus[:, 6:9, 6:9] = function(
                self.filter.logger.P_i_plus[:, 6:9, 6:9],
                ms2**2,
            )
        except:
            pass
