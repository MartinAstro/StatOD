from sympy import *
import numpy as np
from StatOD.constants import EarthParams
from StatOD.utils import print_expression, latlon2cart, ECEF_2_ECI
import numba
from numba import njit

#########################
# Measurement Functions #
#########################
def h_rho_rhod(x, args):
    x_i, y_i, z_i, vx_i, vy_i, vz_i  = x[0], x[1], x[2], x[3], x[4], x[5]
    x_s, y_s, z_s, vx_s, vy_s, vz_s = args[0], args[1], args[2], args[3], args[4], args[5]

    rho = sqrt((x_i - x_s)**2 + (y_i - y_s)**2 + (z_i - z_s)**2)
    rho_dot = ((x_i - x_s)*(vx_i - vx_s) + (y_i - y_s)*(vy_i - vy_s) + (z_i - z_s)*(vz_i - vz_s))/rho

    h = np.array([rho, rho_dot])
    return h.tolist()


#######################
# scenario 1 Functions #
#######################
@njit()
def custom_DiracDelta(x):
    if x == 0.0:
        return 1.0
    else:
        return 0.0

def h_rho_rhod_scenario(x, args):
    x, y, z, vx, vy, vz, mu, J2, Cd, R_0x, R_0y, R_0z, R_1x, R_1y, R_1z, R_2x, R_2y, R_2z = x
    station_idx, omega = args

    x_s = R_0x*DiracDelta(station_idx) + R_1x*DiracDelta(station_idx - 1) + R_2x*DiracDelta(station_idx - 2)
    y_s = R_0y*DiracDelta(station_idx) + R_1y*DiracDelta(station_idx - 1) + R_2y*DiracDelta(station_idx - 2)
    z_s = R_0z*DiracDelta(station_idx) + R_1z*DiracDelta(station_idx - 1) + R_2z*DiracDelta(station_idx - 2)

    vx_s = -omega*y_s
    vy_s = omega*x_s
    vz_s = 0.0

    rho = sqrt((x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2)
    rho_dot = ((x - x_s)*(vx - vx_s) + (y - y_s)*(vy - vy_s) + (z - z_s)*(vz - vz_s))/rho

    h = np.array([rho, rho_dot])
    return h.tolist()

def h_rho(x, args):
    x, y, z, vx, vy, vz, mu, J2, Cd, R_0x, R_0y, R_0z, R_1x, R_1y, R_1z, R_2x, R_2y, R_2z = x
    station_idx, omega = args

    x_s = R_0x*DiracDelta(station_idx) + R_1x*DiracDelta(station_idx - 1) + R_2x*DiracDelta(station_idx - 2)
    y_s = R_0y*DiracDelta(station_idx) + R_1y*DiracDelta(station_idx - 1) + R_2y*DiracDelta(station_idx - 2)
    z_s = R_0z*DiracDelta(station_idx) + R_1z*DiracDelta(station_idx - 1) + R_2z*DiracDelta(station_idx - 2)

    vx_s = -omega*y_s
    vy_s = omega*x_s
    vz_s = 0.0

    rho = sqrt((x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2)
    rho_dot = ((x - x_s)*(vx - vx_s) + (y - y_s)*(vy - vy_s) + (z - z_s)*(vz - vz_s))/rho

    h = np.array([rho])
    return h.tolist()

def h_rhod(x, args):
    x, y, z, vx, vy, vz, mu, J2, Cd, R_0x, R_0y, R_0z, R_1x, R_1y, R_1z, R_2x, R_2y, R_2z = x
    station_idx, omega = args

    x_s = R_0x*DiracDelta(station_idx) + R_1x*DiracDelta(station_idx - 1) + R_2x*DiracDelta(station_idx - 2)
    y_s = R_0y*DiracDelta(station_idx) + R_1y*DiracDelta(station_idx - 1) + R_2y*DiracDelta(station_idx - 2)
    z_s = R_0z*DiracDelta(station_idx) + R_1z*DiracDelta(station_idx - 1) + R_2z*DiracDelta(station_idx - 2)

    vx_s = -omega*y_s
    vy_s = omega*x_s
    vz_s = 0.0

    rho = sqrt((x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2)
    rho_dot = ((x - x_s)*(vx - vx_s) + (y - y_s)*(vy - vy_s) + (z - z_s)*(vz - vz_s))/rho

    h = np.array([rho_dot])
    return h.tolist()



########################
# Example 8 Functions #
########################
def spring_observation_1(x, args):
    x, vx = x[0], x[1]
    h = np.array([x,])
    return h.tolist()
    
def spring_observation_2(x, args):
    x, vx = x[0], x[1]
    h = np.array([abs(x), ])
    return h.tolist()


#####################
# Function Wrappers #
#####################
def dhdx(x, h, args):
    m = len(h)
    n = len(x)
    dhdx = np.zeros((m,n), dtype=np.object)

    for i in range(m): # h[i] differentiated
        for j in range(n): # w.r.t. X[j]
            # dhdx[i,j] = simplify(diff(h[i], x[j]))
            dhdx[i,j] = diff(h[i], x[j])

    return dhdx.tolist()

def measurements(x, h, args, cse_func=cse, consider=None):
    n = len(x) # state [x, y, z, vx, vy, vz]
    k = len(args) # non-state arguments [xs, ys, zs, R, mu]

    # symbolic arguments
    x_args = np.array(symbols('x:'+str(n)))
    c_args = np.array(symbols('arg:'+str(k)))

    h_sym = h(x_args, c_args)
    dhdx_sym = dhdx(x_args, h_sym, c_args)

    # Can't resolve length until function has been called
    m = len(h_sym) # measurement [rho, rhod]
    h_args = np.array(symbols('h:'+str(m)))

    # Define X, R as the inputs to expression
    func_h = lambdify([x_args, c_args], h_sym, cse=cse_func, modules=['numpy' , {'DiracDelta': custom_DiracDelta}])
    func_dhdx = lambdify([x_args, h_args, c_args], dhdx_sym, cse=cse_func, modules=['numpy' , {'DiracDelta': custom_DiracDelta}])

    # Generate consider dynamics if requested
    if consider is not None:
        assert len(consider) == k # ensure that consider variable is of length args
        consider = np.array(consider).astype(bool)
        c_arg_subset = c_args[consider]
        required_args = np.append(x_args, c_args[~consider])
        dhdc_sym = dhdx(c_arg_subset, h_sym, required_args)
        lambdify_dhdc = lambdify([c_arg_subset, h_args, required_args], dhdc_sym, cse=cse_func, modules='numpy')
        func_dhdc = lambdify_dhdc
        return func_h, func_dhdx, func_dhdc

    return func_h, func_dhdx


def get_rho_rhod_el(t, X_ECI, X_obs_ECI, elevation_mask):  
    """Get range and range rate

    Args:
        t (np.array or int): epoch in seconds
        X (np.array): spacecraft state
        R (float): radius of body
        lat (float): latitude in radians [0, pi]
        lon (float): longitude in radians [0, 2*pi]
        theta_0 (float): rotation of Earth at epoch 0 in radians [0, 2*pi]
        elevation_mask (float): elevation below which there are no measurements in radians [0, n]

    Returns:
        [tuple]: (rho, rho_dot, el)
    """
    if len(np.shape(t)) == 0:
        t = np.array([t])

    # (r, lat, lon) - > (x, y, z)

    x_obs_ECI = X_obs_ECI[:,0:3]
    v_obs_ECI = X_obs_ECI[:,3:6]

    LOS_vec = X_ECI[:,0:3] - x_obs_ECI
    LOS_dot_vec = X_ECI[:,3:6] - v_obs_ECI

    LOS_norm = LOS_vec/np.linalg.norm(LOS_vec, axis=1).reshape((-1,1))
    x_obs_ECI_norm = x_obs_ECI/np.linalg.norm(x_obs_ECI, axis=1).reshape((-1,1))

    # Theta [0, np.pi]
    theta = np.arccos(np.sum(LOS_norm*x_obs_ECI_norm,axis=1))
    el = np.pi/2 - theta # [-pi/2, pi/2]
    mask = (el < elevation_mask) 

    rho = np.linalg.norm(LOS_vec, axis=1)
    rho_dot = np.sum(LOS_vec*LOS_dot_vec,axis=1)/rho

    rho[np.where(mask)] = np.nan
    rho_dot[np.where(mask)] = np.nan

    return rho, rho_dot, el

def test_simple():
    R = 6378.0
    mu = 398600.4415 
    J2 = 0.00108263
    theta_0 = np.deg2rad(122)
    omega = 2*np.pi/(24*60*60) # spin rate of 24 hours

    x = np.array([
        -3515.4903270335103, 8390.716310243395, 4127.627352553683,
        -4.357676322178153, -3.3565791387645487, 3.111892927869902
        ])
    h = h_rho_rhod


    lat_1, lon_1 = np.pi - (np.deg2rad(-35.398333) + np.pi/2), np.deg2rad(148.981944)
    X_obs_ECEF = latlon2cart(R, lat_1, lon_1) 
    X_obs_ECI = ECEF_2_ECI(0, X_obs_ECEF, omega, theta_0)

    args = X_obs_ECI
    h, dhdx = measurements(x, h, args)

    h_i = h(x, args)
    dhdx_i = dhdx(x, h_i, args)

    print(h_i)
    print(dhdx_i)

def test_scenario():
    mu = 398600.4415 
    R = 6378.1363#*1000.0
    J2 = 0.001082626925638815
    Cd_0 = 2.0
    area = 3.0 / (1E3**2) 
    mass = 970.0
    rho_0 = 3.614E-13 * (1E3)**3 # convert kg/m^3 to kg/km^3
    r_0 = (700000*1E-3 + R) # km
    H = 88667.0*1E-3 # m
    omega = 7.2921158553E-5 #2*np.pi/(24*60*60) # spin rate of 24 hours

    # t_0 = datetime.datetime(1999, 10, 3, 23, 11, 9.1814)
    theta_0 = np.deg2rad(0)

    # (r, lat, lon) - > (x, y, z)
    X_obs_1_ECEF = np.array([-5127510.0, -3794160.0, 0.0]) / 1E3
    X_obs_2_ECEF = np.array([-3860910.0, 3238490.0, 3898094.0]) / 1E3
    X_obs_3_ECEF = np.array([549505.0, -1380872.0, 6182197.0]) / 1E3

    X_obs_1_ECI = ECEF_2_ECI(0, X_obs_1_ECEF, omega, theta_0)[0:3]
    X_obs_2_ECI = ECEF_2_ECI(0, X_obs_2_ECEF, omega, theta_0)[0:3]
    X_obs_3_ECI = ECEF_2_ECI(0, X_obs_3_ECEF, omega, theta_0)[0:3]


    cart_state = np.array([-757700.0, 5222607.0, 4851500.0,
                           -2213.21, -4678.34, 5371.30]) / 1E3


    x0 = np.hstack((cart_state, [mu, J2, Cd_0], X_obs_1_ECI, X_obs_2_ECI, X_obs_3_ECI))
    
    h = h_rho_rhod_scenario
    args = [1, omega]
    h, dhdx = measurements(x0, h, args)

    h_i = h(x0, args)
    dhdx_i = dhdx(x0, h_i, args)

    print(h_i)
    print(dhdx_i)

if __name__ == "__main__":
    # test_simple()
    test_scenario()

