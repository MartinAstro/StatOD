import pandas as pd
import numpy as np
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
from GravNN.CelestialBodies.Asteroids import Eros

import matplotlib.pyplot as plt
from GravNN.Networks.Model import load_config_and_model
import os 
import GravNN

from GravNN.GravityModels.Polyhedral import get_poly_data
import StatOD
import pickle




def generate_plot(model, config, radius, density, max_percent, X_traj=None):
    if X_traj is None:
        X_traj = np.full((1,3), np.nan)

    planes_exp = PlanesExperiment(model, config, [-radius, radius], density)
    planes_exp.config['gravity_data_fcn'] = [get_poly_data]
    planes_exp.run()
    vis = PlanesVisualizer(planes_exp)

    plt.rc('font', size=6.0)
    plt.rc('figure', figsize= (6.5/3, 6.5/3))
    # plt.rc('figure', figsize= (2.25, 2.25))
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    vis.max = max_percent
    x = vis.experiment.x_test
    y = vis.experiment.percent_error_acc
    cbar_label = "Acceleration Percent Error"
    fig1 = plt.figure()
    vis.plot_plane(x,y, plane='xy', colorbar_label=cbar_label)
    plt.plot(X_traj[:,0], X_traj[:,1], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    fig2 = plt.figure()
    vis.plot_plane(x,y, plane='xz', colorbar_label=cbar_label)
    plt.plot(X_traj[:,0], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    fig3 = plt.figure()
    vis.plot_plane(x,y, plane='yz', colorbar_label=cbar_label)
    plt.plot(X_traj[:,1], X_traj[:,2], color='black', linewidth=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    print(np.nanmean(planes_exp.percent_error_acc))

    return [fig1, fig2, fig3]


def main():
    planet = Eros()
    model_idx = -1
    radius = planet.radius*4
    density = 50 # 30
    max_percent = 20

    with open("Data/Trajectories/trajectory_asteroid.data", 'rb') as f:
        data = pickle.load(f)
    
    X_m = data['X'][:]*1E3 / planet.radius
    N = len(X_m) // 5
    X_m = X_m[0:N]

    # # PINN trained on Point Mass
    df = pd.read_pickle(os.path.dirname(GravNN.__file__) + "/../Data/Dataframes/eros_point_mass_v2.data")
    model_id = df["id"].values[model_idx] # ALC without extra
    config, model = load_config_and_model(model_id, df)
    fig_list = generate_plot(model, config, radius, density, max_percent)
    for idx, fig in enumerate(fig_list):
        plt.figure(fig.number)
        plt.savefig(os.path.dirname(StatOD.__file__) + f"/../Plots/PM_Error_{idx}.pdf")
        

    # PINN trained using DMC
    df = pd.read_pickle(os.path.dirname(StatOD.__file__) + "/../Data/Dataframes/trained_networks_pm.data")
    # model_id = df["id"].values[32] # best
    # model_id = df["id"].values[83] # 
    model_id = df["id"].values[-1] 
    config, model = load_config_and_model(model_id, df, custom_data_dir=os.path.dirname(StatOD.__file__)+"/../Data")
    fig_list = generate_plot(model, config, radius, density, max_percent, X_m)
    for idx, fig in enumerate(fig_list):
        plt.figure(fig.number)
        plt.savefig(os.path.dirname(StatOD.__file__) + f"/../Plots/PINN_DMC_Error_{idx}.pdf")
    plt.show()

if __name__ == '__main__':
    main()