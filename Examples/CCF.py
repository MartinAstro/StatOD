import os
import sys
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from StatOD.data import get_measurements
from StatOD.dynamics import *
from StatOD.filters import FilterLogger, SequentialConsiderCovarianceFilter
from StatOD.measurements import *
from StatOD.rotations import ECI_2_RCI
from StatOD.utils import ECEF_2_ECI, latlon2cart
from StatOD.visualizations import *


def get_true_trajectory(t, mu, R, cart_state, include_J3=False):
     from StatOD.dynamics import dynamics, f_J3
     from scipy.integrate import solve_ivp

     J2 = 0.001082626925638815
     J3 = -1.61 * 10 ** -6
     f_J3_args = np.array([R, mu, J2, J3])
     f_J3_fcn, dfdx_J3 = dynamics(cart_state, f_J3, f_J3_args)
     init_sol = solve_ivp(
          lambda t, x: f_J3_fcn(x, f_J3_args),
          [0, t[-1]],
          cart_state,
          atol=1e-14,
          rtol=2.23e-14,
          t_eval=t,
          method="RK45",
     )
     x_original = init_sol.y.T
     if include_J3:
          x_original = np.hstack((x_original, np.ones((len(x_original),1))*mu, np.ones((len(x_original),1))*J2, np.ones((len(x_original),1))*J3))
     else:
          x_original = np.hstack((x_original, np.ones((len(x_original),1))*mu, np.ones((len(x_original),1))*J2))
     return x_original


def main():
   
    ######################
    ## Get Measurements ##
    ######################
    ep = EarthParams()
    cart_state = np.array([-3515.4903270335103, 8390.716310243395, 4127.627352553683,
                           -4.357676322178153, -3.3565791387645487, 3.111892927869902])
                                  
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_w_J3_w_noise.data")

    t0 = 0.0
    M_end = len(t) // 5

    t = t[:M_end]
    Y = Y[:M_end]




    ##############################
    ## Set State and Covariance ##
    ##############################

    x0 = copy.deepcopy(cart_state)
    x0 = np.hstack((x0, [ep.mu, ep.J2]))
    dx0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 += dx0

    c0 = np.array([0.0])
    dc0 = np.array([ep.J3])

    sigma_r = (1E4 / 1E3) ** 2  
    sigma_r_dot = (1E4 / 1E3) ** 2
    sigma_mu = (1E4 / 1E3) ** 2  
    sigma_J2 = (1E4 / 1E3) ** 2
    P_xx_diag = np.array(
        [sigma_r, sigma_r, sigma_r, 
        sigma_r_dot, sigma_r_dot, sigma_r_dot,
        sigma_mu, sigma_J2]
    )
    P_cc_diag = np.array([ep.J3**2])
    R_diag = np.array([1e-3, 1e-6]) ** 2

    P_xx_0 = np.diag(P_xx_diag)
    P_cc_0 = np.diag(P_cc_diag)
    R0 = np.diag(R_diag)

    ########################
    ## Configure Dynamics ##
    ########################

    c_args = np.array([ep.R, 0.0])
    f_consider_mask = np.array([0,1])
    f, dfdx, dfdc = dynamics(x0, f_consider_J3, c_args, consider=f_consider_mask, use_numba=True)

    h_args = X_stations_ECI[0]
    h_args = np.append(h_args, c_args)
    h_consider_mask = np.array([0,0,0,0,0,0,0,1])
    h, dhdx, dhdc = measurements(x0, h_rho_rhod, h_args, consider=h_consider_mask)

    Q_args = []
    Q0 = np.eye(3) * 5e-8 ** 2
    Q_fcn = None

    R_vec = np.repeat(np.array([R0]), len(t), axis=0)

    ######################
    ## Configure Filter ##
    ######################

    # Initialize
    f_dict = {
        "f": f,
        "dfdx": dfdx,
        "dfdc": dfdc,
        "f_args": c_args,
        "f_consider_mask" : f_consider_mask,
        "Q_fcn": Q_fcn,
        "Q": Q0,
        "Q_args": Q_args,
    }

    h_dict = {"h": h, 
              "dhdx": dhdx,
              "dhdc": dhdc,
              "h_args": h_args,
              "h_consider_mask" : h_consider_mask,
              }

    start_time = time.time()
    logger = FilterLogger(len(x0), len(t), len(c0))
    filter = SequentialConsiderCovarianceFilter(t0, x0, dx0, c0, dc0, P_xx_0, P_cc_0, f_dict, h_dict, logger=logger)
    f_args_vec = np.full((len(t), len(c_args)), c_args)
    h_args_vec = X_stations_ECI
    filter.run(t, Y[:,1:], R_vec, f_args_vec, h_args_vec, h_args_append=c_args)
    print("Time Elapsed: " + str(time.time() - start_time))

    ##############
    ## Plotting ##
    ##############
    x_original = get_true_trajectory(t, ep.mu, ep.R, cart_state, include_J3=True)

    y_labels = np.array(["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$", "$\mu$", "$J_2$", "$J_3$"])
    dX_RMS = plot_state_residual_and_covariance_consider(
        logger.t_i,
        filter.logger.x_hat_i_plus,
        x_original[:,:8],
        y_labels,
        covariances=filter.logger.P_i_plus,
        consider_covariances=filter.logger.P_c_i_plus[:,:8,:8],
        use_std=True,
        sigma=2
    )

    def plot_position_error(filter):
        plt.figure()
        plt.semilogy(np.linalg.norm(filter.logger.x_hat_i_plus[:,0:3] - x_original[:,0:3],axis=1), label=r'$\Delta x$')
        plt.semilogy(np.linalg.norm(filter.logger.x_hat_c_i_plus[:,0:3] - x_original[:,0:3],axis=1), label=r'$\Delta x_c$')
        
        plt.semilogy(np.linalg.norm(np.diagonal(filter.logger.P_i_plus[:,0:3, 0:3],axis1=1,axis2=2),axis=1), label='$P$')
        plt.semilogy(np.linalg.norm(np.diagonal(filter.logger.P_c_i_plus[:,0:3, 0:3],axis1=1,axis2=2),axis=1), label='$P_c$')
        plt.legend()

    plot_position_error(filter)

    #########################
    ## Back Prop Estimates ##
    #########################

    end_idx = len(filter.logger.i)-1
    dx0_new = filter.logger.dx_i_plus[-1]
    P0_new = filter.logger.P_c_i_plus[-1]
    N = len(filter.logger.phi_ti_ti_m1[0])
    M = len(filter.logger.theta_ti_ti_m1[0,0])
    for i in range(end_idx, 0, -1):
        phi = filter.logger.phi_ti_ti_m1[i]
        theta = filter.logger.theta_ti_ti_m1[i]
        psi = np.block([[phi, theta], [np.zeros((M,N)), np.eye(M)]])

        phi_inv = np.linalg.inv(phi)
        psi_inv = np.linalg.inv(psi)

        dx0_new = phi_inv@dx0_new
        P0_new = psi_inv@P0_new@psi_inv.T

    print(f"dx is {dx0_new}")
    print(f"P0_new is {P0_new}")

    #############################################
    ## Propagate the "corrected" initial state ##
    #############################################
    
    dx0 = dx0_new
    P_xx_0 = P0_new[:N,:N]
    P_cc_0 = P0_new[N:,N:]

    logger = FilterLogger(len(x0), len(t), len(c0))
    filter = SequentialConsiderCovarianceFilter(t0, x0, dx0, c0, dc0, P_xx_0, P_cc_0, f_dict, h_dict, logger=logger)
    filter.run(t, Y[:,1:], R_vec, f_args_vec, h_args_vec, h_args_append=c_args)

    #######################
    ## Map P_cc_0 to t_f ##
    #######################

    end_idx = len(filter.logger.i)-1
    dx0_new = filter.logger.dx_i_plus[0]
    P0_new = filter.logger.P_c_i_plus[0]
    N = len(filter.logger.phi_ti_ti_m1[0])
    M = len(filter.logger.theta_ti_ti_m1[0,0])
    P_c_i = np.zeros_like(filter.logger.P_c_i_plus)
    P_c_i[0] = P0_new
    for i in range(1, len(filter.logger.i)):
        phi = filter.logger.phi_ti_ti_m1[i]
        theta = filter.logger.theta_ti_ti_m1[i]
        psi = np.block([[phi, theta], [np.zeros((M,N)), np.eye(M)]])
        P_c_i[i] = psi@P_c_i[i-1]@psi.T


    y_labels = np.array(["$x$", "$y$", "$z$", "$v_x$", "$v_y$", "$v_z$", "$\mu$", "$J_2$", "$J_3$"])
    dX_RMS = plot_state_residual_and_covariance_consider(
        logger.t_i,
        filter.logger.x_i,
        x_original[:,:8],
        y_labels,
        covariances=None,
        consider_covariances=P_c_i[:,:8,:8],
        use_std=False,
        sigma=2
    )

    plot_position_error(filter)


    plt.show()


if __name__ == "__main__":
    main()