import os

import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Trajectories.SurfaceDist import SurfaceDist

import StatOD
from StatOD.utils import generate_heterogeneous_model

if __name__ == "__main__":
    planet = Eros()

    traj = SurfaceDist(planet, planet.obj_8k)

    model = generate_heterogeneous_model(planet, planet.obj_8k)
    r_offset_0 = model.offset_list[0]
    r_offset_1 = model.offset_list[1]
    r_offset_2 = model.offset_list[2]

    from GravNN.Visualization.PolyVisualization import PolyVisualization

    vis = PolyVisualization()
    vis.plot_polyhedron(planet.obj_8k, None, cmap="Greys", cbar=False, alpha=0.1)
    plt.gca().scatter(r_offset_0[0], r_offset_0[1], r_offset_0[2], s=300, color="blue")
    plt.gca().scatter(r_offset_1[0], r_offset_1[1], r_offset_1[2], s=150, color="red")
    plt.gca().scatter(r_offset_2[0], r_offset_2[1], r_offset_2[2], s=150, color="red")

    plt.gca().set_xticklabels("")
    plt.gca().set_yticklabels("")
    plt.gca().set_zticklabels("")
    plt.gca().view_init(elev=35, azim=180 + 45, roll=0)

    statOD_dir = os.path.dirname(StatOD.__file__) + "/../Plots/"

    vis.save(plt.gcf(), f"{statOD_dir}eros_hetero_density.pdf")

    plt.show()
