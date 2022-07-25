import numpy as np
import pickle
import os
import StatOD
import numba as nb
from numba import njit

def get_measurements(filepath, t_gap=10):
    with open(filepath, 'rb') as f:
        measurements = pickle.load(f)

    time = measurements['time']
    idx_1 = np.zeros((len(time),)) + 0
    idx_2 = np.zeros((len(time),)) + 1
    idx_3 = np.zeros((len(time),)) + 2

    station_1 = np.vstack((time, idx_1, measurements['rho_1'], measurements['rho_dot_1'])).T
    station_2 = np.vstack((time, idx_2, measurements['rho_2'], measurements['rho_dot_2'])).T
    station_3 = np.vstack((time, idx_3, measurements['rho_3'], measurements['rho_dot_3'])).T

    Y = np.vstack((station_1, station_2, station_3))    
    Y = Y[Y[:, 0].argsort()] # sort by time

    # remove empty rows
    mask = np.any(np.isnan(Y), axis=1)
    Y = Y[~mask]

    t = Y[:,0].squeeze()
    Y = Y[:,1:]
    
    time_gaps = np.where(np.diff(t) > t_gap)[0] + 1
    while len(time_gaps) > 0:
            
        idx = time_gaps[0]
        t_vec = np.arange(t[idx-1], t[idx], t_gap)[1:]
        y_vec = np.zeros((len(t_vec), 3))*np.nan
        y_vec[:,0] = 1

        t = np.insert(t, idx, t_vec)
        Y = np.insert(Y, idx, y_vec, axis=0)

        time_gaps = np.where(np.diff(t) > t_gap)[0] + 1

    return t, Y 

def get_scenario_1_measurements(filepath=os.path.dirname(os.path.abspath(StatOD.__file__)) + '/../Data/scenario_1_observations.txt'):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    k = len(lines[0].split())
    measurements = np.zeros((len(lines), k))

    for i in range(len(lines)):
        measurements[i] = np.array(lines[i].split(), dtype=np.float)

    # Convert station IDs into idx
    station_ids = np.array([101, 337, 394])
    measurements[np.where(measurements[:,1] == station_ids[0]),1] = 0
    measurements[np.where(measurements[:,1] == station_ids[1]),1] = 1
    measurements[np.where(measurements[:,1] == station_ids[2]),1] = 2


    t = measurements[:,0]
    Y = measurements[:,1:] 
    Y[:,1:] /= 1E3
    return t, Y 


def get_scenario_2_measurements(file, t_gap=60):

    file = os.path.dirname(os.path.abspath(StatOD.__file__)) + "/../" + file
    # read in text file
    with open(file,'r') as f:
        lines = f.readlines()
    data = np.zeros((len(lines)-2, 7))

    for i in range(2,len(lines)):
        line = lines[i]
        entries = line.split(',')
        entries = [entry.strip() for entry in entries]
        data[i-2] = np.array(entries[:-1], dtype=float)
    
    # Sort measurements into per station
    t = data[:,0].reshape((-1,1))
    idx = np.zeros_like(t)
    y_station_1 = np.hstack((t, idx+0, data[:,1+0].reshape((-1,1)), data[:,4+0].reshape((-1,1))))
    y_station_2 = np.hstack((t, idx+1, data[:,1+1].reshape((-1,1)), data[:,4+1].reshape((-1,1))))
    y_station_3 = np.hstack((t, idx+2, data[:,1+2].reshape((-1,1)), data[:,4+2].reshape((-1,1))))

    # eliminate measurements that contain only nan's
    y_station_1 = y_station_1[~np.any(np.isnan(y_station_1),axis=1)]
    y_station_2 = y_station_2[~np.any(np.isnan(y_station_2),axis=1)]
    y_station_3 = y_station_3[~np.any(np.isnan(y_station_3),axis=1)]

    Y = np.vstack((y_station_1, y_station_2, y_station_3))
    Y = Y[Y[:, 0].argsort()] # sort by time
    t = Y[:,0]

    # Fill in any intervals that extend more than 10 seconds 
    time_gaps = np.where(np.diff(t) > t_gap)[0] + 1
    while len(time_gaps) > 0:
            
        idx = time_gaps[0]
        t_vec = np.arange(t[idx-1], t[idx], t_gap)[1:]
        y_vec = np.zeros((len(t_vec), len(Y[0])))*np.nan
        y_vec[:,1] = 1

        t = np.insert(t, idx, t_vec)
        Y = np.insert(Y, idx, y_vec, axis=0)

        time_gaps = np.where(np.diff(t) > t_gap)[0] + 1

    return t, Y[:,1:]

@njit()
def COEstoRV(a,e,i,RAAN,omega,nu,mu):
    p = a*(1-e**2); 1# semi-latus rectum, [km]
    r = p/(1+e*np.cos(nu)); # orbit radius, [km]

    h = np.sqrt(mu*a*(1-e**2)); # angular momentum

    x = r*(np.cos(RAAN)*np.cos(omega+nu) - np.sin(RAAN)*np.sin(omega+nu)*np.cos(i)); # x-position, [km]
    y = r*(np.sin(RAAN)*np.cos(omega+nu) + np.cos(RAAN)*np.sin(omega+nu)*np.cos(i)); # y-position, [km]
    z = r*(np.sin(i)*np.sin(omega+nu)); # z-position, [km]

    xdot = x*h*e/(r*p)*np.sin(nu) - h/r*(np.cos(RAAN)*np.sin(omega+nu) + np.sin(RAAN)*np.cos(omega+nu)*np.cos(i)); # x-velocity, [km/s]
    ydot = y*h*e/(r*p)*np.sin(nu) - h/r*(np.sin(RAAN)*np.sin(omega+nu) - np.cos(RAAN)*np.cos(omega+nu)*np.cos(i)); # y-velocity, [km/s]
    zdot = z*h*e/(r*p)*np.sin(nu) + h/r*np.sin(i)*np.cos(omega+nu)
    
    R = np.array([x, y, z]).reshape((3,-1))
    V = np.array([xdot, ydot, zdot]).reshape((3,-1))
    return R, V

@njit()
def get_earth_position(JD):
    # These are in EMO2000
    deg2rad = np.pi/180
    T = (JD - 2451545.0)/36525
    AU = 149597870.700
    mu_s = 132712440017.987


    # Earth
    L = np.array([100.466449,     35999.3728519,  -0.00000568,      0.0])
    a = np.array([1.000001018,    0.0,             0.0,             0.0])
    e = np.array([0.01670862,    -0.000042037,    -0.0000001236,	0.00000000004])
    i = np.array([0.0,            0.0130546,      -0.00000931,     -0.000000034])
    W = np.array([174.873174,    -0.2410908,       0.00004067,     -0.000001327])
    P = np.array([102.937348,     0.3225557,       0.00015026,      0.000000478])
    mu_p = 3.98600432896939e5


    Tvec = np.array([1.0, T, T**2, T**3])

    # Mean longitude of Planet
    L = L@Tvec*deg2rad

    # Semimajor axis of the orbit
    a = a@Tvec*AU

    # Eccentricity of the orbit
    e = e@Tvec

    # Inclination of the orbit
    inc = i@Tvec*deg2rad

    # Longitude of the Ascending Node
    W = W@Tvec*deg2rad

    # Longitude of the Perihelion
    P = P@Tvec*deg2rad

    # Argument of perihelion
    w = P - W

    # Mean anomaly of orbit
    M = L - P

    # True anomaly of orbit
    Ccen = (2*e - e**3/4 + 5/96.0*e**5)*np.sin(M) + (5./4*e**2 - 11./24*e**4)*np.sin(2*M) + \
        (13./12*e**3 - 43./64*e**5)*np.sin(3*M) + 103./96*e**4*np.sin(4*M) + \
        1097./960*e**5*np.sin(5*M)

    nu = M + Ccen

    # from OrbitalElements.coordinate_transforms import oe2cart_tf


    R, V = COEstoRV(a, e, inc, W, w, nu, mu_s)

    # Convert to EME2000 if necessary
    # if frame =='EME2000':
    if True:
        theta = 23.4393*np.pi/180
        C = np.array([[1., 0., 0.],
            [0., np.cos(theta), -np.sin(theta)],
            [0., np.sin(theta), np.cos(theta)]])
        R = C@R
        V = C@V
    
    r_new = R.T[0]#reshape((-1,))
    # v_new = V.T[0]#reshape((-1,))

    return r_new#, v_new

    
def get_example8_measurements(case, noisy=True):
    from scipy.integrate import solve_ivp
    x_true = np.array([0.0,1.0])
    k = 1
    eta = 1000
    def spring(t, x):
        return np.array([x[1], -k*x[0]])
    def spring_duffing(t, x):
        return np.array([x[1], -k*x[0]-eta*x[0]**3])

    if case == 0: 
        dynamics = spring
    else:
        dynamics = spring_duffing

    sol = solve_ivp(dynamics, [0, 2*np.pi], x_true, t_eval=np.linspace(0,2*np.pi, 50))
    x_truth = sol.y.T

    noise = np.random.normal(0, 0.1, size=(len(x_truth),))
    if case == 0 or case == 1:
        y = x_truth[:,0]
    elif case == 2:
        y = np.abs(x_truth[:,0]) 

    return sol.t, y + noise, x_truth, y