
import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.CelestialBodies.Asteroids import Eros
import pickle
import os

def plot_DMC_subplot(x, y1, y2):
    plt.plot(x, y1)
    plt.plot(x, y2)
    criteria1 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 > 0)))).T, axis=1)
    criteria2 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 < 0)))).T, axis=1)
    criteria3 = np.all(np.vstack((np.array(y1 > 0), np.array((y2 < 0)))).T, axis=1)
    criteria4 = np.all(np.vstack((np.array(y1 < 0), np.array((y2 > 0)))).T, axis=1)
    percent_productive = np.round((np.count_nonzero(criteria1) + np.count_nonzero(criteria2)) / len(x) * 100,2)
    plt.gca().annotate(f"Percent Useful: {percent_productive}",xy=(0.75, 0.75), xycoords='axes fraction', size=8)
    plt.gca().fill_between(x, y1, y2, 
                where=criteria1, color='green', alpha=0.3,
                interpolate=True)
    plt.gca().fill_between(x, y1, y2, 
                where=criteria2, color='green', alpha=0.3,
                interpolate=True)
    plt.gca().fill_between(x, y1, y2, 
                where=criteria3, color='red', alpha=0.3,
                interpolate=True)
    plt.gca().fill_between(x, y1, y2, 
                where=criteria4, color='red', alpha=0.3,
                interpolate=True)

def plot_DMC(logger, w_truth):
    plt.figure()
    plt.subplot(311)
    plot_DMC_subplot(logger.t_i, logger.x_hat_i_plus[:,6], w_truth[:,0])
    plt.subplot(312)
    plot_DMC_subplot(logger.t_i, logger.x_hat_i_plus[:,7], w_truth[:,1])
    plt.subplot(313)
    plot_DMC_subplot(logger.t_i, logger.x_hat_i_plus[:,8], w_truth[:,2])

    # Plot magnitude
    DMC_mag = np.linalg.norm(logger.x_hat_i_plus[:,6:9], axis=1)
    plt.figure()
    plt.plot(DMC_mag)
    print(f"Average DMC Mag {np.mean(DMC_mag)}")


def plot_error_planes(planes_exp, max_error, logger):
    visPlanes = PlanesVisualizer(planes_exp)
    plt.rc('text', usetex=False)
    X_traj = logger.x_hat_i_plus[:,0:3]*1E3 / Eros().radius
    x = visPlanes.experiment.x_test
    y = visPlanes.experiment.percent_error_acc
    plt.figure()
    visPlanes.max = max_error
    visPlanes.plot_plane(x,y, plane='xy')
    plt.plot(X_traj[:,0], X_traj[:,1], color='black', linewidth=0.5)
    fig2 = plt.figure()
    visPlanes.plot_plane(x,y, plane='xz')
    plt.plot(X_traj[:,0], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    fig3 = plt.figure()
    visPlanes.plot_plane(x,y, plane='yz')
    plt.plot(X_traj[:,1], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])



def non_dimensionalize(t, z0, Y, P_diag, R_diag, tau, q, dim_constants):
    t_star = dim_constants['t_star']
    l_star = dim_constants['l_star']
    ms = dim_constants['l_star'] / dim_constants['t_star']
    ms2 = dim_constants['l_star'] / dim_constants['t_star']**2

    z0[0:3] /= l_star
    z0[3:6] /= ms
    z0[6:9] /= ms2
    
    P_diag[0:3] /= l_star**2
    P_diag[3:6] /= ms**2
    P_diag[6:9] /= ms2**2

    Y[:,1:] /= l_star

    R_diag /= l_star**2

    t /= t_star
    tau /= t_star
    q /= ms2

    return t, z0, Y, P_diag, R_diag, tau, q, dim_constants


def dimensionalize(logger, t, Y, y_hat_vec, R_vec, dim_constants):

    # dimensionalize 
    t_star = dim_constants['t_star']
    l_star = dim_constants['l_star']
    ms = dim_constants['l_star'] / dim_constants['t_star']
    ms2 = dim_constants['l_star'] / dim_constants['t_star']**2

    t *= t_star

    logger.x_hat_i_plus[:,0:3] *= l_star
    logger.x_hat_i_plus[:,3:6] *= l_star / t_star
    logger.x_hat_i_plus[:,6:9] *= ms2
    t *= t_star
    Y[:,1:] *= l_star
    y_hat_vec[:,0:3] *= l_star
    
    logger.t_i *= t_star
    logger.P_i_plus[:,0:3,0:3] *= l_star**2
    logger.P_i_plus[:,3:6,3:6] *= ms**2
    logger.P_i_plus[:,6:9,6:9] *= ms2**2
    
    R_vec *= l_star**2

    return logger, t, Y, y_hat_vec, R_vec, dim_constants



def get_trajectory_data(data_file):
    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + f'Data/Trajectories/{data_file}.data', 'rb') as f:
        data = pickle.load(f)
    return data

