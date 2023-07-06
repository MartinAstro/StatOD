import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_symmetric_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
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

    regress_deg = 16
    statOD_dir = os.path.dirname(StatOD.__file__)

    file_name = f"{statOD_dir}/../Data/Products/SH_Eros_model_{regress_deg}.csv"
    model = SphericalHarmonics(file_name, regress_deg)
    a_sh = model.compute_acceleration(x_hetero_poly)

    da_norm = np.linalg.norm(a_sh - a_hetero_poly, axis=1)
    a_norm = np.linalg.norm(a_hetero_poly, axis=1)

    a_error = da_norm / a_norm * 100
    a_error = a_error.reshape((-1, 1))

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"
    vis = PolyVisualization(save_directory=statOD_dir)
    vis.fig_size = (vis.w_full / 4, vis.w_full / 4)

    #######################################
    # Surface Acceleration of heterogenous
    #######################################
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
        cbar=None,
    )
    format()
    vis.save(plt.gcf(), "eros_sh_surface.pdf")

    cmap = plt.gcf().axes[0].collections[0].cmap

    vis.fig_size = (vis.w_full * 4 / 5, vis.w_full / 8)
    fig_cbar, ax = vis.newFig()

    # Create the colorbar
    vmin, vmax = 0, 10
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
        label="Acceleration Error [\%]",
    )
    cbar
    vis.save(plt.gcf(), "colorbar.pdf")

    plt.show()


if __name__ == "__main__":
    main()
