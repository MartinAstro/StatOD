import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


class TruePlanesVisualizer(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)

    def plot(self, percent_error=False, max=None, **kwargs):
        if percent_error:
            # x/a_test is the truth and x/a_pred is the prediction
            x = self.experiment.x_test

            a_pred = self.experiment.a_pred
            a_test = self.experiment.a_test
            da = a_test - a_pred

            a_test_norm = np.linalg.norm(a_test, axis=1, keepdims=True)
            da_norm = np.linalg.norm(da, axis=1, keepdims=True)
            y = da_norm / a_test_norm * 100

            cmap = cm.jet  # cm.RdYlGn.reversed()

        else:
            # x/a_test is the truth and x/a_pred is the prediction
            x = self.experiment.x_test
            y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)
            cmap = cm.viridis

        if max is None:
            self.max = np.nanmean(y) + 1 * np.nanstd(y)
        else:
            self.max = max

        self.newFig()
        plt.subplot(1, 3, 1)
        self.plot_plane(x, y, plane="xy", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(1, 3, 2)
        self.plot_plane(x, y, plane="xz", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")

        plt.subplot(1, 3, 3)
        self.plot_plane(x, y, plane="yz", cbar=False, cmap=cmap, **kwargs)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")
