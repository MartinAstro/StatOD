import os

import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_symmetric_data
from GravNN.GravityModels.PointMass import get_pm_data
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Trajectories import SurfaceDist
from GravNN.Visualization.PolyVisualization import PolyVisualization

import StatOD


def format():
    plt.gca().set_xticklabels("")
    plt.gca().set_yticklabels("")
    plt.gca().set_zticklabels("")
    plt.gca().view_init(elev=35, azim=180 + 45, roll=0)


def main():
    planet = Eros()
    trajectory = SurfaceDist(planet, planet.obj_8k)

    x_hetero_poly, a_hetero_poly, u_hetero_poly = get_hetero_poly_symmetric_data(
        trajectory,
        planet.obj_8k,
        remove_point_mass=[False],
    )

    x_homo_poly, a_homo_poly, u_homo_poly = get_poly_data(
        trajectory,
        planet.obj_8k,
        remove_point_mass=[False],
    )

    x_pm, a_pm, u_pm = get_pm_data(
        trajectory,
        planet.obj_8k,
        remove_point_mass=[False],
        planet=[Eros()],
    )

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"
    vis = PolyVisualization(save_directory=statOD_dir)
    vis.fig_size = (vis.w_full / 4, vis.w_full / 4)

    #######################################
    # Surface Acceleration of heterogenous
    #######################################
    vis.plot_polyhedron(
        planet.obj_8k,
        a_hetero_poly,
        label="Acceleration [m/$s^2$]",
        log=False,
        cmap="viridis",
        cmap_reverse=False,
        percent=False,
        alpha=1,
        cbar_orientation="vertical",
    )
    format()
    vis.save(plt.gcf(), "eros_heterogeneous_surface.pdf")

    #######################################
    # Error of homogenous assumption
    #######################################
    da_norm = np.linalg.norm(a_homo_poly - a_hetero_poly, axis=1)
    a_norm = np.linalg.norm(a_hetero_poly, axis=1)

    a_error = da_norm / a_norm * 100
    a_error = a_error.reshape((-1, 1))
    vis.plot_polyhedron(
        planet.obj_8k,
        a_error,
        label="Acceleration Errors",
        cmap="jet",
        log=False,
        percent=True,
        max_percent=0.1,
        cmap_reverse=False,
        alpha=1,
        cbar_orientation="vertical",
    )
    format()
    vis.save(plt.gcf(), "eros_homo_surface_error.pdf")

    #######################################
    # Error of PM model
    #######################################
    da_norm = np.linalg.norm(a_pm - a_hetero_poly, axis=1)
    a_norm = np.linalg.norm(a_hetero_poly, axis=1)

    a_error = da_norm / a_norm * 100
    a_error = a_error.reshape((-1, 1))
    vis.plot_polyhedron(
        planet.obj_8k,
        a_error,
        label="Acceleration Errors",
        log=False,
        percent=True,
        cmap="jet",
        max_percent=0.1,
        cmap_reverse=False,
        alpha=1,
        cbar_orientation="vertical",
    )
    format()
    vis.save(plt.gcf(), "eros_pm_surface_error.pdf")

    plt.show()


if __name__ == "__main__":
    main()
