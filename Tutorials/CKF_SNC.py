"""
Kalman Filter with Stochastic Noise Compensation Example
============================================================

"""

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import StatOD
from StatOD.constants import *
from StatOD.data import get_measurements
from StatOD.dynamics import dynamics, f_J2, get_Q, process_noise
from StatOD.filters import FilterLogger, KalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.visualization.visualizations import *

# %%
# To begin, gather the telemetry / measurements
# that will be used by the filter to update the state. In this example, the
# telemetry is the range and range-rate in kilometers of spacecraft in orbit
# whose dynamics include a point mass and $J_2$ gravity contribution.
#
# Note that the telemetry is measured by three stations on the surface of Earth.
# The `get_measurements` function evolves the location of these stations in the
# Earth-Centered Inertial (ECI) frame and returns it alongside the time and
# range/range-rate measurements (`t, Y`).

t, Y, X_stations_ECI = get_measurements(
    "Data/Measurements/range_rangerate_w_J2_w_noise.data"
)

# Reduce the simulation length for faster evaluation.
M_end = len(t) // 5
t = t[:M_end]
Y = Y[:M_end]

# %%
# Initialize the starting state $(x, y, z, v_x, v_y, v_z)$
# in [km] and [km/s] respectively and then add an initial perturbation (`dx0`).
# Also initialize the initial state and measurement covariance (`P0` and `R0`).
# The measurements included gaussian noise on the order of 1 meter in range and
# 1 mm/s in range rate, hence the choice of diagonals for the measurement covariance `R0`.
ep = EarthParams()
cart_state = np.array(
    [
        -3515.4903270335103,
        8390.716310243395,
        4127.627352553683,
        -4.357676322178153,
        -3.3565791387645487,
        3.111892927869902,
    ]
)

dx0 = np.array([0.1, 0.0, 0.0, 1e-4, 0.0, 0.0])
x0 = cart_state + dx0

P0 = np.diag([1, 1, 1, 1e-3, 1e-3, 1e-3]) ** 2
R0 = np.diag([1e-3, 1e-6]) ** 2
t0 = 0.0

# %%
# In this example, stochastic noise compensation is used. The process noise matrix
# $Q_0$ is defined below.
#
# Because the continuous time process noise is different than the discrete process noise matrix
# a process noise function must be defined (`Q_fcn`). This function is computed using `sympy` to
# compute an exact symbolic expression on the fly to accommodate any arbitrary estimation scenario.
# This symbolic expression is then lambdified to be used within the filter. This process is repeated
# for the dynamics and measurement functions.
#
# As such, users must pass representative values of the state and any auxillary arguments to the generating
# function `process_noise`.

Q0 = np.eye(3) * 1e-7**2
Q_args = []
Q_fcn = process_noise(x0, Q0, get_Q, Q_args, use_numba=False)


# %%
# Likewise, the dynamics and measurement functions must be computed symbolically and then lambdified.
# A collection of dynamics function are provided within the `StatOD.Dynamics` file. In this example
# the spacecraft's dynamics are influenced only by point mass gravity and $J_2$. Therefore, the
# `f_J2` symbolic expression is selected, and passed to the generating function `dynamics` alongside
# the additional arguments not included in the state that are needed for the dynamics function (`f_args`).
#
# The measurement functions are generated the same way, passing in a representative state and
# additional arguments into the generating function such that a lambdified symbolic expression
# can be produced and passed to the filter.
f_args = np.array([ep.R, ep.mu, ep.J2])
f, dfdx = dynamics(x0, f_J2, f_args)
f_dict = {
    "f": f,
    "dfdx": dfdx,
    "f_args": f_args,
    "Q_fcn": Q_fcn,
    "Q": Q0,
    "Q_args": Q_args,
}

h_args = X_stations_ECI[0]
h, dhdx = measurements(x0, h_rho_rhod, h_args)
h_dict = {"h": h, "dhdx": dhdx, "h_args": h_args}

# %%
# Now that the lambdified symbolic measurement and dynamics functions are produced,
# the auxiallary arguments not included in the state should be gathered for each measurement.
# For the dynamics model, this means passing in the fixed parameters $(R, \mu, J_2)$ at each
# time step. For the measurement function, this includes the stations position in the ECI frame.

f_args_vec = np.full((len(t), len(f_args)), f_args)
h_args_vec = X_stations_ECI
R_vec = np.repeat(np.array([R0]), len(t), axis=0)

# %%
# To initialize the filter, pass in the initial parameters and corresponding generating functions.
# With the filter initialized, call the `.run()` command with the vectors of time, measurements,
# measurement covariances, dynamics and measurement auxillary arguments.

start_time = time.time()
logger = FilterLogger(len(x0), len(t))
filter = KalmanFilter(t0, x0, dx0, P0, f_dict, h_dict, logger=logger)
filter.run(t, Y[:, 1:], R_vec, f_args_vec, h_args_vec)
print("Time Elapsed: " + str(time.time() - start_time))

# %%
# After the filter finishes, the corresponding state pre- and post- measurement updates are saved
# in the `filter.logger` attribute. These can be used to produce the measurement residuals which
# can then be plotted.

package_dir = os.path.dirname(StatOD.__file__) + "/../"
with open(package_dir + "Data/Trajectories/trajectory_J2.data", "rb") as f:
    traj_data = pickle.load(f)

x_truth = traj_data["X"][:M_end]
y_hat_vec = np.zeros((len(t), 2))
for i in range(len(t)):
    y_hat_vec[i] = filter.predict_measurement(
        logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i]
    )

directory = "Plots/" + filter.__class__.__name__ + "/"
y_labels = np.array([r"$\rho$", r"$\dot{\rho}$"])
vis = VisualizationBase(logger, directory, False)
vis.plot_state_errors(x_truth)
vis.plot_residuals(Y[:, 1:], y_hat_vec, R_vec, y_labels)
plt.show()
