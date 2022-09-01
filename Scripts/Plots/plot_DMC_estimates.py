import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import StatOD
from StatOD.measurements import h_rho_rhod, measurements
from StatOD.visualizations import VisualizationBase
from StatOD.constants import ErosParams
from StatOD.data import get_measurements

def main():
    ##################################
    # Gather measurement predictions #
    ##################################
    
    plt.rc('font', size= 10.0)

    file_name_1 = 'Data/FilterLogs/DMC_1e-08_150.0.data'
    file_name_2 = 'Data/FilterLogs/DMC_1e-08_140.0.data'
    file_name_3 = 'Data/FilterLogs/DMC_5e-08_100.0.data'

    # file_name_1 = 'Data/FilterLogs/DMC_1e-11_5000.data'
    # file_name_2 = 'Data/FilterLogs/DMC_1e-10_5000.data'
    # file_name_3 = 'Data/FilterLogs/DMC_1e-11_50.data'

    fig_list = [file_name_1, file_name_2, file_name_3]

    ep = ErosParams()
    t, Y, X_stations_ECI = get_measurements("Data/Measurements/range_rangerate_asteroid_wo_noise.data", t_gap=60)
    h_args_vec = X_stations_ECI
    
    idx_min = len(t)

    for idx, file_name in enumerate(fig_list):
        package_dir = os.path.dirname(StatOD.__file__) + "/../"
        with open(package_dir + 'Data/Trajectories/trajectory_asteroid.data', 'rb') as f:
            traj_data = pickle.load(f)
        with open(package_dir + file_name, 'rb') as f:
            logger = pickle.load(f)

        t = logger.t_i
        entries = np.count_nonzero(t)+1
        idx_min = entries if entries < idx_min else idx_min
        M_end = idx_min

        z0 = logger.x_i[0]
        h_args = X_stations_ECI[0]
        h, dhdx = measurements(z0, h_rho_rhod, h_args)


        x_truth = traj_data['X'][:M_end] + ep.X_BE_E
        w_truth =  np.full((len(x_truth),3), 0) # DMC should be zero ideally 
        x_truth = np.hstack((x_truth, w_truth)) 
        y_hat_vec = np.zeros((len(t), 2))
        for i in range(len(t)):
            y_hat_vec[i] = np.array(h(logger.x_hat_i_plus[i], X_stations_ECI[i]))

        logger.t_i = logger.t_i / 3600 # seconds to hours

        directory = "Plots/"
        y_labels = np.array([r'$\rho$', r'$\dot{\rho}$'])
        vis = VisualizationBase(logger, directory, False)
 
        w_est = np.linalg.norm(logger.x_hat_i_plus[:,6:9],axis=1) # magnitude of w
        w_truth = np.zeros_like(w_est)
        P_est = np.sqrt(np.trace(logger.P_i_plus[:,6:9,6:9], axis1=1, axis2=2))
        P_est = np.zeros_like(P_est)+ np.nan
        vis.logger.t_i = vis.logger.t_i[:M_end]
        vis.plot_state_error(w_est[:M_end], w_truth[:M_end], P_est[:M_end], "$|w|$")
        plt.gca().set_yscale('log')
        plt.gca().set_ylim([1E-9, None])


#        vis.plot_state_errors(x_truth)
        fig = plt.gcf()
        # fig = plt.figure(7)
        fig.axes[0].set_ylabel(r"$|\mathbf{w}|$")
        fig.set_size_inches(3.25, 1.25)
        # plt.gca().set_xticks([0, 5, 10, 15, 20])
        plt.gca().ticklabel_format(useOffset=False, style='plain', axis='x')
        basename = file_name.split("/")[-1].split('.')[0]
        q, tau = basename.split("_")[1:]
        plt.annotate(rf"$\tau$ = {tau}, $q$ = {q}", 
                    fontsize=8, 
                    xycoords='axes fraction', 
                    xy=(0.5, 0.75),
                    bbox=dict(boxstyle="round", fc="w", alpha=0.5))


        if idx == len(fig_list) -1:
            fig.set_size_inches(3.25, 1.4)
            fig.axes[0].set_xlabel("Time [hr]")

        descriptor = os.path.basename(file_name).split('.')[0]
        vis.save(f"{descriptor}_x")
        # plt.show()
        plt.close('all')





if __name__ == "__main__":
    main()