import pickle
import inspect
import time
from sympy import *
import numba
import sys
from numba import njit, jit, prange
import numpy as np
from sympy import init_printing
from StatOD.utils import print_expression
from StatOD.data import get_earth_position
import StatOD

import os
os.environ["NUMBA_CACHE_DIR"] = "./numba_cache_tmp"

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
    # for i in range(N // 2):
    #     A[i, N//2 + i] = 1
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    phi = eye(N) + A*dt
    
    # control only influence x_dd
    B = zeros(N,M)
    # for i in range(N // 2):
    #     B[N//2 + i, i] = 1
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

def get_Q_DMC(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, = args

    A = zeros(N,N)

    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    # TODO: Revisit this. If not commented, it causes the Q
    # matrix to shrink the covariance instead of increase it. 
    # For now, make the linear approximation used in SNC instead.
    
    # A[6,6] = -1/tau
    # A[7,7] = -1/tau
    # A[8,8] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A*dt)
    
    B = zeros(N,M)

    B[6,0] = 1
    B[7,1] = 1
    B[8,2] = 1

    integrand = phi*B*DCM.T*Q*DCM*B.T*phi.T

    Q_i_i_m1 = np.zeros((N,N), dtype=np.object)
    for i in range(N): # f[i] differentiated
        for j in range(i,N): # w.r.t. X[j]
            integrated = integrate(integrand[i,j], dt)
            Q_i_i_m1[i,j] = integrated
            Q_i_i_m1[j,i] = integrated

            
    # numba can't work with arrays of sympy ints and floats in same matrix
    # so just force sympy ints to be floats
    Q_i_i_m1[np.where(Q_i_i_m1 == 0)] = 0.0
    Q_i_i_m1[np.where(Q_i_i_m1 == 1)] = 1.0

    return Q_i_i_m1.tolist()

# DMC Models
def f_J2_DMC(x, args):
    x, y, z, vx, vy, vz, w0, w1, w2 = x
    R, mu, J2, tau = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    a_x = -(U0_x + U2_x) + w0
    a_y = -(U0_y + U2_y) + w1
    a_z = -(U0_z + U2_z) + w2

    w0_d = -1/tau * w0
    w1_d = -1/tau * w1
    w2_d = -1/tau * w2

    return np.array([vx, vy, vz, a_x, a_y, a_z, w0_d, w1_d, w2_d]).tolist() 


##########################
# Simple Cartesian State #
##########################
def f_point_mass(x, args):
    x, y, z, vx, vy, vz = x
    mu, x_body, y_body, z_body = args

    r_mag = sqrt((x - x_body)**2 + (y - y_body)**2 + (z - z_body)**2)

    U0 = -mu/r_mag
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)

    a_x = -U0_x
    a_y = -U0_y
    a_z = -U0_z

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()

def f_J2(x, args):
    x, y, z, vx, vy, vz = x
    R, mu, J2 = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    a_x = -(U0_x + U2_x)
    a_y = -(U0_y + U2_y)
    a_z = -(U0_z + U2_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()

def f_J3(x, args):
    x, y, z, vx, vy, vz = x
    R, mu, J2, J3 = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)
    P3 = P(z/r_mag, 3)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    U3 = (mu/r_mag)*(R/r_mag)**3*P3*J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist() 

def f_C22(x, args):
    x, y, z, vx, vy, vz = x
    R, mu, C20, S20, C21, S21, C22, S22 = args
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)
    
    r_mag = sqrt(x**2 + y**2 + z**2)
    
    longitude = atan2(y,x)
    sin_phi = z/r_mag
    P_20 = assoc_legendre(2, 0, sin_phi)*(C20*cos(0*longitude) + S20*sin(0*longitude))
    P_21 = assoc_legendre(2, 1, sin_phi)*(C21*cos(1*longitude) + S21*sin(1*longitude))
    P_22 = assoc_legendre(2, 2, sin_phi)*(C22*cos(2*longitude) + S22*sin(2*longitude))

    U_0 = -(mu/r_mag)
    U_2 = -(mu/r_mag)*(R/r_mag)**2*(P_20 + P_21 + P_22)
    U = U_0 + U_2

    a_x = -differentiate(U, x)
    a_y = -differentiate(U, y)
    a_z = -differentiate(U, z)

    return np.array([vx, vy, vz, a_x, a_y, a_z]).tolist()  

############################
# Augmented State Dynamics #
############################
def f_aug_J2(x, args):
    x, y, z, vx, vy, vz, mu, J2 = x
    R, = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    a_x = -(U0_x + U2_x)
    a_y = -(U0_y + U2_y)
    a_z = -(U0_z + U2_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0]).tolist()

def f_aug_J3(x, args):
    x, y, z, vx, vy, vz, mu, J2, J3 = x
    R, = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)
    P3 = P(z/r_mag, 3)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    U3 = (mu/r_mag)*(R/r_mag)**3*P3*J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0, 0.0]).tolist() 

###########################
# Consider State Dynamics #
###########################
def f_consider_J3(x, args):
    x, y, z, vx, vy, vz, mu, J2 = x
    R, J3 = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)
    P3 = P(z/r_mag, 3)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    U3 = (mu/r_mag)*(R/r_mag)**3*P3*J3

    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)
    U3_x, U3_y, U3_z = differentiate(U3, x), differentiate(U3, y), differentiate(U3, z)

    a_x = -(U0_x + U2_x + U3_x)
    a_y = -(U0_y + U2_y + U3_y)
    a_z = -(U0_z + U2_z + U3_z)

    return np.array([vx, vy, vz, a_x, a_y, a_z, 0.0, 0.0]).tolist() 


##########################
# Scenario State Dynamics #
##########################
def f_scenario_1_J2(x, args):
    x, y, z, vx, vy, vz, mu, J2, Cd, R_0x, R_0y, R_0z, R_1x, R_1y, R_1z, R_2x, R_2y, R_2z = x
    area, mass, rho_0, r_0, H, omega, R = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    rho = rho_0*exp((-(r_mag - r_0)/H))

    # Velocity of atmosphere 

    # r_atmos_dot_ECI = r_atmos_prime_ECI(?) + omega_ECEF/ECI x r_atmos_ECI
    # r_atmos_dot_ECI = 0_ECI + (...)
    v_atmos_x = -omega*y
    v_atmos_y = omega*x
    v_atmos_z = 0.0
    
    v_rel_x = vx - v_atmos_x
    v_rel_y = vy - v_atmos_y
    v_rel_z = vz - v_atmos_z

    v_rel_mag = sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
    F_drag = 1.0/2.0*rho*v_rel_mag*Cd*area/mass

    a_x = -(U0_x + U2_x) - F_drag*v_rel_x
    a_y = -(U0_y + U2_y) - F_drag*v_rel_y
    a_z = -(U0_z + U2_z) - F_drag*v_rel_z

    mu_dot = 0.0
    J2_dot = 0.0
    Cd_dot = 0.0

    R_0x_d, R_0y_d, R_0z_d = -omega*R_0y, omega*R_0x, 0.0
    R_1x_d, R_1y_d, R_1z_d = -omega*R_1y, omega*R_1x, 0.0
    R_2x_d, R_2y_d, R_2z_d = -omega*R_2y, omega*R_2x, 0.0


    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    mu_dot, J2_dot, Cd_dot, 
                    R_0x_d, R_0y_d, R_0z_d, 
                    R_1x_d, R_1y_d, R_1z_d, 
                    R_2x_d, R_2y_d, R_2z_d]).tolist() 

def f_scenario_2(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr = x
    x_E_sun, y_E_sun, z_E_sun, A2M, flux, mu, mu_sun, AU, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP

    # P = flux/c
    # scale = (AU/r_sc_sun_mag)**2

    # a_x_SRP = -Cr*P*scale*A2M*(x_sun_sc/r_sc_sun_mag)
    # a_y_SRP = -Cr*P*scale*A2M*(y_sun_sc/r_sc_sun_mag)
    # a_z_SRP = -Cr*P*scale*A2M*(z_sun_sc/r_sc_sun_mag)

    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP


    Cr_dot = 0.0

    # return np.array([x_sc_E, y_sc_E, z_sc_E,
    #                 x_E_sun, y_E_sun, z_E_sun,
    #                 vx, vy, vz, 
    #                 x_sc_sun, y_sc_sun, z_sc_sun, 
    #                 a_x_grav_E, a_y_grav_E, a_z_grav_E, 
    #                 a_x_grav_sun, a_y_grav_sun, a_z_grav_sun, 
    #                 a_x_SRP, a_y_SRP, a_z_SRP, 
    #                 Cr_dot, 
    #                 ]) 

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    ]).tolist() 

def f_scenario_2_mu(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr, mu, mu_sun = x
    x_E_sun, y_E_sun, z_E_sun, A2M, flux, misc, misc_2, AU, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP

    # P = flux/c
    # scale = (AU/r_sc_sun_mag)**2

    # a_x_SRP = -Cr*P*scale*A2M*(x_sun_sc/r_sc_sun_mag)
    # a_y_SRP = -Cr*P*scale*A2M*(y_sun_sc/r_sc_sun_mag)
    # a_z_SRP = -Cr*P*scale*A2M*(z_sun_sc/r_sc_sun_mag)

    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP


    Cr_dot = 0.0
    mu_sun_dot = 0.0
    mu_dot = 0.0

    # return np.array([x_sc_E, y_sc_E, z_sc_E,
    #                 x_E_sun, y_E_sun, z_E_sun,
    #                 vx, vy, vz, 
    #                 x_sc_sun, y_sc_sun, z_sc_sun, 
    #                 a_x_grav_E, a_y_grav_E, a_z_grav_E, 
    #                 a_x_grav_sun, a_y_grav_sun, a_z_grav_sun, 
    #                 a_x_SRP, a_y_SRP, a_z_SRP, 
    #                 Cr_dot, 
    #                 ]) 

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    mu_dot, mu_sun_dot
                    ]).tolist() 

def f_scenario_2_extra(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr, mu, mu_sun, A2M, flux = x
    x_E_sun, y_E_sun, z_E_sun, misc_3, misc_4, misc, misc_2, AU, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP

    # P = flux/c
    # scale = (AU/r_sc_sun_mag)**2

    # a_x_SRP = -Cr*P*scale*A2M*(x_sun_sc/r_sc_sun_mag)
    # a_y_SRP = -Cr*P*scale*A2M*(y_sun_sc/r_sc_sun_mag)
    # a_z_SRP = -Cr*P*scale*A2M*(z_sun_sc/r_sc_sun_mag)

    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP


    Cr_dot = 0.0
    mu_sun_dot = 0.0
    mu_dot = 0.0
    A2M_dot = 0.0
    flux_dot = 0.0

    # return np.array([x_sc_E, y_sc_E, z_sc_E,
    #                 x_E_sun, y_E_sun, z_E_sun,
    #                 vx, vy, vz, 
    #                 x_sc_sun, y_sc_sun, z_sc_sun, 
    #                 a_x_grav_E, a_y_grav_E, a_z_grav_E, 
    #                 a_x_grav_sun, a_y_grav_sun, a_z_grav_sun, 
    #                 a_x_SRP, a_y_SRP, a_z_SRP, 
    #                 Cr_dot, 
    #                 ]) 

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    mu_dot, mu_sun_dot,
                    A2M_dot, flux_dot
                    ]).tolist() 

def f_scenario_2_DMC(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr, w0, w1, w2 = x
    x_E_sun, y_E_sun, z_E_sun, A2M, flux, mu, mu_sun, tau, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP
    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)

    # DMC
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP + w0
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP + w1
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP + w2


    Cr_dot = 0.0

    w0_d = -1/tau * w0
    w1_d = -1/tau * w1
    w2_d = -1/tau * w2

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    w0_d, w1_d, w2_d
                    ]).tolist() 

def get_Q_DMC_scenario_2(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, = args

    A = zeros(N,N)
    
    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    # A[6,6] = 0 -- Constant Cr

    A[7,7] = -1/tau
    A[8,8] = -1/tau
    A[9,9] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A*dt)
    
    B = zeros(N,M)
    B[7,0] = 1
    B[8,1] = 1
    B[9,2] = 1

    integrand = phi*B*DCM.T*Q*DCM*B.T*phi.T

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

def f_scenario_2_DMC_mu(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr, mu, mu_sun, w0, w1, w2 = x
    x_E_sun, y_E_sun, z_E_sun, A2M, flux, misc, misc_2, tau, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP
    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)

    # DMC
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP + w0
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP + w1
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP + w2


    Cr_dot = 0.0
    mu_dot = 0.0
    mu_sun_dot = 0.0

    w0_d = -1/tau * w0
    w1_d = -1/tau * w1
    w2_d = -1/tau * w2

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    mu_dot, mu_sun_dot,
                    w0_d, w1_d, w2_d
                    ]).tolist() 

def get_Q_DMC_scenario_2_mu(dt, x, Q, DCM, args):
    N = len(x)
    M = Q.shape[0]
    tau, = args

    A = zeros(N,N)
    
    # velocities
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1

    # acceleration is just DMC
    A[3,6] = 1
    A[4,7] = 1
    A[5,8] = 1

    # A[6,6] = 0 -- Constant Cr
    # A[7,7] = 0 -- Constant mu
    # A[8,8] = 0 -- Constant mu_sun

    A[9,9] = -1/tau
    A[10,10] = -1/tau
    A[11,11] = -1/tau

    # phi = eye(N) + A*dt
    phi = exp(A*dt)
    
    B = zeros(N,M)
    B[9,0] = 1
    B[10,1] = 1
    B[11,2] = 1

    integrand = phi*B*DCM.T*Q*DCM*B.T*phi.T

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

#################
# PINN Dynamics #
#################

def f_PINN(x, args):
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:].astype(float)
    x_pos_km = (X_sc_ECI[0:3] - X_body_ECI[0:3])
    x_vel_km = (X_sc_ECI[3:6] - X_body_ECI[3:6])

    # gravity model requires meters so convert km -> m
    x_pos_m = x_pos_km*1E3
    x_acc_m = model.generate_acceleration(x_pos_m).reshape((-1,))

    #convert acceleration to km/s^2
    x_acc_km = x_acc_m/1E3

    return np.hstack((x_vel_km, x_acc_km))

def dfdx_PINN(x, f, args):
    # f argument is needed to make interface standard 
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:].astype(float)

    x_pos_km = X_sc_ECI[0:3] - X_body_ECI[0:3]
    x_pos_m = x_pos_km*1E3

    dfdx_acc_m = model.generate_dadx(x_pos_m).reshape((3,3)) #[(m/s^2) / m] = [1/s^2]

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3,3))
    dfdx = np.block([[zero_3x3, dfdx_vel],[dfdx_acc_m, zero_3x3]])
    return dfdx

def f_PINN_DMC(x, args):
    X_sc_ECI = x[0:6]
    w_vec = x[6:]

    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])

    x_pos = (X_sc_ECI[0:3] - X_body_ECI[0:3]) # either km or [-]
    x_vel = (X_sc_ECI[3:6] - X_body_ECI[3:6])

    # scaling occurs within the gravity model 
    x_acc_m = model.generate_acceleration(x_pos).reshape((-1,))

    x_acc = x_acc_m + w_vec
    
    w_d = -1.0/tau*w_vec

    return np.hstack((x_vel, x_acc, w_d))

def dfdx_PINN_DMC(x, f, args):
    # f argument is needed to make interface standard 
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:-1].astype(float)
    tau = float(args[-1])
    
    x_pos = X_sc_ECI[0:3] - X_body_ECI[0:3]# either km or [-]

    dfdx_acc_m = model.generate_dadx(x_pos).reshape((3,3)) #[(m/s^2) / m] = [1/s^2]

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3,3))
    dfdx = np.block([[zero_3x3, dfdx_vel],[dfdx_acc_m, zero_3x3]])

    dfdw = np.eye(3) * -1.0/tau
    zeros_6x3 = np.zeros((6,3))
    intermediate_6x3 = np.zeros((6,3))
    intermediate_6x3[3:] = np.eye(3)

    dfdz = np.block(
        [
            [dfdx, intermediate_6x3],
            [zeros_6x3.T, dfdw]
        ]
    )
    return dfdz

########################
# UKF Example Dynamics #
########################
def f_spring(x, args):
    x, v_x = x
    k, eta = args
    a_x = -k*x
    return np.array([v_x, a_x]).tolist() 

def f_spring_duffing(x, args):
    x, v_x = x
    k, eta = args
    a_x = -k*x - eta*x**3
    return np.array([v_x, a_x]).tolist() 


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
    f_args = np.array(symbols('f:'+str(n)))
    x_args = np.array(symbols('x:'+str(n))) # state
    c_args = np.array(symbols('arg:'+str(k))) # parameters

    f_sym = f(x_args, c_args)   
    dfdx_sym = dfdx(x_args, f_sym, c_args)

    # Define X, R as the inputs to expression
    lambdify_f = lambdify([x_args, c_args], f_sym, cse=cse_func, modules='numpy')
    lambdify_dfdx = lambdify([x_args, f_args, c_args], dfdx_sym, cse=cse_func, modules='numpy')

    # return func_f, func_dfdx
    if use_numba:
        f_func = numba.njit(lambdify_f, cache=False)
        dfdx_func = numba.njit(lambdify_dfdx, cache=False)
    else:
        f_func = lambdify_f
        dfdx_func = lambdify_dfdx

    x_tmp = np.arange(1,n+1,1) # make values different
    f_tmp = np.arange(2,n+2,1) # to minimize risk of 
    c_tmp = np.arange(3,k+3,1) # div by zero
    tmp = f_func(x_tmp, c_tmp)
    tmp = dfdx_func(x_tmp, f_tmp, c_tmp)

    # Generate consider dynamics if requested
    if consider is not None:
        assert len(consider) == k # ensure that consider variable is of length args
        consider = np.array(consider).astype(bool)
        c_arg_subset = c_args[consider]
        required_args = np.append(x_args, c_args[~consider])
        dfdc_sym = dfdx(c_arg_subset, f_sym, required_args)
        lambdify_dfdc = lambdify([c_arg_subset, f_args, required_args], dfdc_sym, cse=cse_func, modules='numpy')
        dfdc_func = numba.njit(lambdify_dfdc, cache=False) if use_numba else lambdify_dfdc
        required_tmp = np.append(x_tmp, c_tmp[~consider])
        tmp = dfdc_func(c_tmp[consider], f_tmp, required_tmp)
        return f_func, dfdx_func, dfdc_func

    return f_func, dfdx_func

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

# dynamics = memory.cache(dynamics)

# Use cached output regardless of the inputs
# process_noise = memory.cache(process_noise)#, ignore=['x', 'f', 'args', 'cse_func', 'use_numba'])

@njit(cache=False)
def dynamics_ivp(t, Z, f, dfdx, f_args):
    N = int(1/2 * (np.sqrt(4*len(Z) + 1) - 1))
    X_inst = Z[0:N]
    phi_inst = Z[N:].reshape((N,N))

    f_inst = np.array(f(X_inst, f_args)).reshape((N))
    dfdx_inst = np.array(dfdx(X_inst, f_inst, f_args)).reshape((N,N))

    phi_dot = dfdx_inst@phi_inst
    Zd = np.hstack((f_inst, phi_dot.reshape((-1))))
    return Zd

# @njit(cache=False)
def dynamics_ivp_no_jit(t, Z, f, dfdx, f_args):
    N = int(1/2 * (np.sqrt(4*len(Z) + 1) - 1))
    X_inst = Z[0:N]
    phi_inst = Z[N:].reshape((N,N))

    f_inst = np.array(f(X_inst, f_args)).reshape((N))
    dfdx_inst = np.array(dfdx(X_inst, f_inst, f_args)).reshape((N,N))

    phi_dot = dfdx_inst@phi_inst
    Zd = np.hstack((f_inst, phi_dot.reshape((-1))))
    return Zd

@njit(cache=False)
def consider_dynamics_ivp(t, Z, f, dfdx, dfdc, args, N, M, consider_mask):
    X_inst = Z[0:N]
    C_inst = Z[N:N+M]
    phi_inst = Z[N+M:(N+M+N**2)].reshape((N,N))
    theta_inst = Z[(N+M+N**2):].reshape((N, M))

    required_args = np.append(X_inst, args[~consider_mask])

    f_inst = np.array(f(X_inst, args)).reshape((N))
    dfdx_inst = np.array(dfdx(X_inst, f_inst, args)).reshape((N,N))
    dfdc_inst = np.array(dfdc(C_inst, f_inst, required_args)).reshape((N,M))

    phi_dot = dfdx_inst@phi_inst
    theta_dot = dfdx_inst@theta_inst + dfdc_inst
    Zd = np.hstack((f_inst, np.zeros_like(C_inst), phi_dot.reshape((-1)), theta_dot.reshape((-1))))
    return Zd
    
@njit(cache=False)
def dynamics_ivp_unscented(t, Z, f, dfdx, f_args):
    L = len(Z)
    N = int(1/4.*(np.sqrt(8*L + 1 ) -1))
    sigma_points = Z.reshape((2*N + 1, N))
    Zd = np.zeros((L,))
    for k in range(2*N+1):
        X_inst = sigma_points[k]
        f_inst = np.array(f(X_inst, f_args)).reshape((N))
        Zd[k*N:(k+1)*N] = f_inst
    return Zd

@njit(cache=False, parallel=True)
def dynamics_ivp_particle(t, Z, f, N, f_args):
    X_inst = Z.reshape((N,-1))
    Zd = np.zeros_like(X_inst)
    for i in prange(len(X_inst)):
        Zd[i] = f(X_inst[i], f_args)
    return Zd.reshape(-1)

@njit(cache=False)
def dynamics_ivp_proj2(t, z, f_fcn, dfdx_fcn, f_args):
    N = int(1/2 * (np.sqrt(4*len(z) + 1) - 1))
    x = z[:N]
    phi = z[N:].reshape((N,N))
    J0 = 2456296.25

    r_E = get_earth_position(J0 + t/(24*3600))
    f_args[0:3] = r_E
    f = np.array(f_fcn(x, f_args)).reshape((N,))
    dfdx = np.array(dfdx_fcn(x, f, f_args)).reshape((N,N))
    
    phi_dot = dfdx@phi
    z_dot = np.hstack((f, phi_dot.reshape((-1))))
    return z_dot

if __name__ == "__main__":
    import timeit
    R = 6378.0
    mu = 398600.4415 
    J2 = 0.00108263
    x = np.array([
        -3515.4903270335103, 8390.716310243395, 4127.627352553683,
        -4.357676322178153, -3.3565791387645487, 3.111892927869902
        ])
    f = f_J2
    args = np.array([R, mu, J2])
    f, dfdx = dynamics(x, f, args)

    f_i = f(x, args)
    dfdx_i = dfdx(x, f_i, args)

    print(np.mean(timeit.repeat(lambda : f(x,args), repeat=100, number=1000)))
    print(np.mean(timeit.repeat(lambda : dfdx(x, f_i, args), repeat=100, number=1000)))

    