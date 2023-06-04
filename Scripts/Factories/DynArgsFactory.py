import numpy as np

from StatOD.constants import ErosParams, SunParams


class DynArgsFactory:
    def __init__(self):
        pass

    def get_HF_args(self, model):
        sunParams = SunParams()

        SRP = sunParams.radiant_flux  # W of the sun
        area_2_mass = 0.01  # m^2/kg
        sun_pos_P = np.array([-2.180987e11, 0.0, 0.0]) / 1e3  # 1.4579 AU from Eros
        # sun_pos_P = np.array([-2.180987e9, 0.0, 0.0]) / 1e3  # Test to see STM work
        Cr = 0.1
        eros_pos_P = np.zeros((6,))

        f_args = np.hstack(
            (
                model,
                eros_pos_P,
                sun_pos_P,
                area_2_mass,
                SRP,
                sunParams.mu_sun,
                Cr,
                sunParams.c,
                0.0,
                ErosParams().omega,
            ),
        )

        return f_args

    def get_gravity_only_args(self, model, t_vec):
        eros_pos = np.zeros((6,))
        f_args = np.hstack((model, eros_pos, 0.0, ErosParams().omega))
        f_args = np.full((len(t_vec), len(f_args)), f_args)
        f_args[:, -2] = t_vec
        f_args[:, -1] = ErosParams().omega
