import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from StatOD.visualizations import VisualizationBase
from StatOD.constants import ErosParams
import StatOD
from GravNN.GravityModels.Polyhedral import Polyhedral, PointMass
from GravNN.CelestialBodies.Asteroids import Eros

def main():
    file_name = 'Data/FilterLogs/DMC_5e-08_100.0.data'

    package_dir = os.path.dirname(StatOD.__file__) + "/../"
    with open(package_dir + 'Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
        traj_data = pickle.load(f)
    with open(package_dir + file_name, 'rb') as f:
        logger = pickle.load(f)

    # Setup
    ep = ErosParams()
    t = logger.t_i
    M_end = np.count_nonzero(t)+1
    eros = Eros()
    gravity_model = Polyhedral(eros, eros.obj_8k)
    gravity_model_pm = PointMass(eros)



    # Get true position
    x_truth = traj_data['X'][:M_end] + ep.X_BE_E
    
    # Get true accelerations beyond point mass 
    acc = gravity_model.compute_acceleration(x_truth[:,0:3])
    acc_pm = gravity_model_pm.compute_acceleration(x_truth[:,0:3])
    w_truth =  acc - acc_pm 

    acc_mag = np.linalg.norm(acc, axis=1)
    w_truth_mag = np.linalg.norm(w_truth, axis=1)

    plt.figure()
    plt.plot(acc_mag)
    plt.ylabel("True |a|") # ~1.13E-9 + 5E-14

    plt.figure()
    plt.plot(w_truth_mag)
    plt.ylabel("True |w|") # ~1E-11


    z_truth = np.hstack((x_truth, w_truth)) 

    logger.t_i = logger.t_i / 3600 # seconds to hours

    directory = "Plots/"
    vis = VisualizationBase(logger, directory, False)

    w_est = np.linalg.norm(logger.x_hat_i_plus[:,6:9],axis=1) # magnitude of w
    w_truth = np.zeros_like(w_est)
    P_est = np.sqrt(np.trace(logger.P_i_plus[:,6:9,6:9], axis1=1, axis2=2))
    vis.plot_state_error(w_est, w_truth, P_est, "$|w|$")
    plt.gca().set_yscale('log')
    plt.gca().set_ylim([1E-9, None])


    fig = plt.gcf()
    fig.axes[0].set_ylabel(r"$|\mathbf{w}|$")
    fig.set_size_inches(3.25, 1.25)

    # descriptor = os.path.basename(file_name).split('.')[0]
    # vis.save(f"{descriptor}_x")
    
    vis.plot_state_errors(z_truth)
    
    plt.show()
    # plt.close('all')



if __name__ == "__main__":
    main()