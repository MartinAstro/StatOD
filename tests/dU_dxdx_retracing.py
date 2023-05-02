import os
import time

import numpy as np

import StatOD
from StatOD.models import pinnGravityModel


def main():
    # Write a function that loads a PINN gravity model wrapped by pinnGravityModel class,
    # initializes a list of position testing data, and then runs the compute_dU_dxdx
    # function on the testing data in a for loop.
    # The goal is to diagnose if the dU_dxdx function is getting retraced.
    dim_constants = {"t_star": 1e4, "m_star": 1e0, "l_star": 1e1}

    statOD_dir = os.path.dirname(StatOD.__file__)
    df_file = f"{statOD_dir}/../Data/Dataframes/eros_filter_poly.data"

    dim_constants_pinn = dim_constants.copy()
    dim_constants_pinn["l_star"] *= 1e3
    model = pinnGravityModel(
        df_file,
        learning_rate=1e-4,
        dim_constants=dim_constants_pinn,
    )

    # Initialize a list of position testing data in a numpy array
    positions = np.random.uniform(-1e3, 1e3, size=(100, 3))

    # loop through each position as a (1,3) array and compute the dU_dxdx
    for position in positions:

        # print the amount of time it takes to run generate_dadx for each position
        start_time = time.time()
        model.generate_dadx(position)
        print("Time to run generate_dadx: ", time.time() - start_time)

        # print the amount of time it takes to run compute_acceleration for each position
        start_time = time.time()
        model.compute_acceleration(position)
        print("Time to run generate_dadx: ", time.time() - start_time)


if __name__ == "__main__":
    main()
