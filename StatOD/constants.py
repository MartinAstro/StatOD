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