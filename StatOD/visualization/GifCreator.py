import glob
import os
import unittest

import imageio
import matplotlib.pyplot as plt
import numpy as np


class GifCreator:
    def __init__(self, figures, output_name, duration=10):
        self.figures = figures
        self.output_name = output_name
        self.duration = duration
        self.temp_dir = "./temp_gif_images/"

    def create_gif(self, clean_up=True):
        # Create temporary directory for storing png images
        os.makedirs(self.temp_dir, exist_ok=True)

        images = []
        for idx, fig in enumerate(self.figures):
            temp_file_path = f"{self.temp_dir}{idx}.png"
            fig.savefig(temp_file_path)  # Save figure as png
            images.append(imageio.imread(temp_file_path))  # Read the saved image

        # Create gif from images
        imageio.mimsave(self.output_name, images, duration=self.duration)

        if clean_up:
            self.clean()

    def clean(self):
        # Clean up temporary png images
        files = glob.glob(self.temp_dir + "*.png")
        for f in files:
            os.remove(f)


class TestGifCreator(unittest.TestCase):
    def setUp(self):
        # Create some sample figures
        self.figures = []
        x = np.linspace(0, 2 * np.pi, 100)
        for i in range(10):
            fig, ax = plt.subplots()
            ax.plot(x, np.sin(x + i))
            self.figures.append(fig)

    def test_create_gif(self):
        gif_creator = GifCreator(self.figures, "test.gif")
        gif_creator.create_gif()
        self.assertTrue(os.path.exists("test.gif"))  # Check if the gif file was created
        os.remove("test.gif")


if __name__ == "__main__":
    unittest.main()
