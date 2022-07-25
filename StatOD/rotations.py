import numpy as np
def ECI_2_RCI(x):
    
    # Inertial Frame
    r = x[0:3]
    r_dot = x[3:6]
    h = np.cross(r, r_dot)

    # RIC Frame (radial, in track, cross track)
    o_r = r / np.linalg.norm(r)
    o_h = h / np.linalg.norm(h)
    o_theta = np.cross(o_h, o_r)

    ON = np.vstack([o_r.T, o_theta.T, o_h.T])
    return ON


def no_rotation(x):
    return np.eye(3)