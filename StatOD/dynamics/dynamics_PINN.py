import numpy as np
#################
# PINN Dynamics #
#################

def f_PINN(x, args):
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:].astype(float)
    x_pos_km = (X_sc_ECI[0:3] - X_body_ECI[0:3])
    x_vel_km = (X_sc_ECI[3:6] - X_body_ECI[3:6])

    # gravity model requires meters so convert km -> m
    x_pos_m = x_pos_km*1E3
    x_acc_m = model.compute_acceleration(x_pos_m).reshape((-1,))

    #convert acceleration to km/s^2
    x_acc_km = x_acc_m/1E3

    return np.hstack((x_vel_km, x_acc_km))

def dfdx_PINN(x, f, args):
    # f argument is needed to make interface standard 
    X_sc_ECI = x
    model = args[0]
    X_body_ECI = args[1:].astype(float)

    x_pos_km = X_sc_ECI[0:3] - X_body_ECI[0:3]
    x_pos_m = x_pos_km*1E3

    dfdx_acc_m = model.generate_dadx(x_pos_m).reshape((3,3)) #[(m/s^2) / m] = [1/s^2]

    dfdx_vel = np.eye(3)
    zero_3x3 = np.zeros((3,3))
    dfdx = np.block([[zero_3x3, dfdx_vel],[dfdx_acc_m, zero_3x3]])
    return dfdx