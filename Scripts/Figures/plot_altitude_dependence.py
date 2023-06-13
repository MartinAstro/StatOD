import glob
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from GravNN.Visualization.HeatmapVisualizer import Heatmap3DVisualizer


def main():
    # Glob all of the trajectory metrics data
    traj_files = glob.glob("Data/TrajMetrics/*.data")

    # Create a dataframe to store the data
    df = pd.DataFrame()

    # Populate a dataframe with the data, splitting the file names based on semi-major axis and eccentricity
    for traj_file in traj_files:
        # split the file name looking for "_a" and "_e"
        model_info, element_info = traj_file.split("_a")

        # Check if pm or poly in model info
        if "pm" in model_info:
            model = "pm"
        elif "poly" in model_info:
            model = "poly"
        else:
            raise Exception("Model not found")

        # remove ".data" from the element info
        element_info = element_info.replace(".data", "")

        # split into the semi-major axis and eccentricity
        elements = element_info.split("_e")
        a = float(elements[0])
        e = float(elements[1])

        # load the data
        with open(traj_file, "rb") as f:
            metrics = pickle.load(f)

        # merge all data into a single dictionary
        all_metrics = {}

        all_metrics.update(metrics["Extrapolation"][0])
        all_metrics.update(metrics["Planes"][0])

        dX_sum = 0
        for key, value in metrics["Trajectory"][0].items():
            if "dX_sum" in key:
                dX_sum += value
        traj_metric = dX_sum / 4
        all_metrics.update({"Trajectory": traj_metric})

        all_metrics.update(
            {
                "a": a,
                "e": e,
                "model": model,
            },
        )

        # add the data to the dataframe
        df = df.append(all_metrics, ignore_index=True)

    # Use the 3D Histogram Plot
    heatmap_vis = Heatmap3DVisualizer(df)
    heatmap_vis.plot(
        x="a",
        y="e",
        z="Trajectory",
        query="model == 'pm'",
        cbarlabel="Trajectory Error (km)",
        cmap="viridis",
        alpha=1.0,
        newFig=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
