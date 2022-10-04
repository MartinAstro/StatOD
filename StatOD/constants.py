import numpy as np

###########
## Earth ##
###########

class EarthParams:
    def __init__(self):
        self.mu = 398600.4415 # km^3/s^2
        self.R = 6378.1363 # km
        self.J2 = 0.001082626925638815 # [unitless]
        self.J3 = -1.61*10**-6 # [unitless]
        self.omega =  7.2921158553e-5  # 2*np.pi/(24*60*60) # Rotation [rad/s]

        # From EGM96
        self.C20 = -0.484165371736E-03
        self.S20 = 0.000000000000E+00

        self.C21 = -0.186987635955E-09
        self.S21 = 0.119528012031E-08

        self.C22 = 0.243914352398E-05
        self.S22 = -0.140016683654E-05

class ErosParams:
    def __init__(self):
        G = 6.67430 * 10 ** -11
        self.mu = G * 6.687 * 10 ** 15 / (1E3)**3
        self.R = 16.000 # km
        self.X_BE_E = np.array([1.995636E7,0,0,0,0,0]) # [km, km/s] Assume Eros is 0.1334 AU from Earth
        self.omega = 3.318*1E-4 # rad/s