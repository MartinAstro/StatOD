import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from StatOD.data import get_measurements
from StatOD.dynamics import dynamics, f_J2, f_J2_DMC, get_Q, get_Q_DMC, process_noise, f_J3
from StatOD.filters import FilterLogger, KalmanFilter
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.visualizations import *
from StatOD.constants import *

def compute_J3_accelerations(X, mu, R, J3):
     x = X[:,0]
     y = X[:,1]
     z = X[:,2]
     r_mag = np.linalg.norm(X[:,:3], axis=1)
     a_J3 = 1/2*J3*(mu/r_mag**2)*(R/r_mag)**3*np.array([
                                        5*(7*(z/r_mag)**3 - 3*(z/r_mag))*x/r_mag, 
                                        5*(7*(z/r_mag)**3 - 3*(z/r_mag))*y/r_mag, 
                                        3*(1 - 10*(z/r_mag)**2 + 35/3*(z/r_mag)**4)
                                        ])
     return a_J3.T


def main():
    ep = EarthParams()
    cart_state = np.array([-3515.4903270335103, 8390.716310243395, 4127.627352553683,
                           -4.357676322178153, -3.3565791387645487, 3.111892927869902])
                                  
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_w_J3_w_noise.data")

    # Decrease scenario length
    M_end = len(t) // 5
    t = t[:M_end]
    Y = Y[:M_end]

    # Initialize state and filter parameters
    dx0 = np.array([0.1, 0.0, 0.0, 1E-4, 0.0, 0.0]) 
    t0 = 0.0

    q = 1E-10
    Q0 = np.eye(3) * q ** 2
    P = 9952.014054236302  #orbit period
    tau = P / 10

    # Augment with DMC
    t0 = 0.0

    w0 = np.array([0, 0, 0]) 
    dw0 = np.array([0, 0, 0])

    Z0 = np.hstack((cart_state, w0))
    dZ0 = np.hstack((dx0, dw0))
    Z0 += dZ0

    sigma_r = 1**2  # km ^2
    sigma_r_dot = (1E-3) ** 2
    sigma_w = (1E-7) ** 2

    P_diag = np.array(
        [sigma_r, sigma_r, sigma_r, 
         sigma_r_dot, sigma_r_dot, sigma_r_dot,
         sigma_w, sigma_w, sigma_w]
    )
    R_diag = np.array([1e-3, 1e-6]) ** 2

    P0 = np.diag(P_diag)
    R0 = np.diag(R_diag)
    
    # Initialize
    f_args = np.array([ep.R, ep.mu, ep.J2, tau])
    f, dfdx = dynamics(Z0, f_J2_DMC, f_args)

    Q_args = np.array([tau,])
    Q_fcn = process_noise(Z0, Q0, get_Q_DMC, Q_args, use_numba=False)
    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "f_args": f_args,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
    }

    h_args = cart_state
    h, dhdx = measurements(Z0, h_rho_rhod, h_args)
    h_dict = {"h": h, "dhdx": dhdx, "h_args": h_args}


    #########################
    # Generate f/h_args_vec #
    #########################

    f_args_vec = np.full((len(t), len(f_args)), f_args)
    h_args_vec = X_stations_ECI
    R_vec = np.repeat(np.array([R0]), len(t), axis=0)

    ##############
    # Run Filter #
    ##############

    start_time = time.time()
    state_labels = np.array([
        "$x$", "$y$", "$z$", 
        "$v_x$", "$v_y$", "$v_z$",  
        "$a_{J3,x}$", "$a_{J3,y}$", "$a_{J3,z}$"
        ])
    logger = FilterLogger(len(Z0), len(t), state_labels=state_labels)
    filter = KalmanFilter(t0, Z0, dZ0, P0, f_dict, h_dict, logger=logger)
    filter.run(t, Y[:,1:], R_vec, f_args_vec, h_args_vec)
    print("Time Elapsed: " + str(time.time() - start_time))

    ############
    # Plotting #
    ############
    with open('Data/Trajectories/trajectory_J3.data', 'rb') as f:
        traj_data = pickle.load(f)

    x_truth = traj_data['X'][:M_end]

    # Measurement residuals
    y_hat = np.zeros((len(t), 2))
    for i in range(len(t)):
        y_hat[i] = filter.predict_measurement(logger.x_i[i], logger.dx_i_plus[i], h_args_vec[i])

    # state error
    a_vec = compute_J3_accelerations(filter.logger.x_hat_i_plus[:,:6], ep.mu, ep.R, ep.J3)
    Z_truth = np.hstack((x_truth[:M_end], a_vec))

    directory = "Plots/" + filter.__class__.__name__ + "/"
    y_labels = np.array([r'$\rho$', r'$\dot{\rho}$'])
    vis = VisualizationBase(logger, directory, False)
    vis.plot_residuals(Y[:,1:], y_hat, R_vec, y_labels)
    vis.plot_state_errors(Z_truth)

    plt.show()

if __name__ == "__main__":
    main()