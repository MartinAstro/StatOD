from sympy import * 
import numpy as np
 
##########################
# Scenario State Dynamics #
##########################
def f_scenario_1_J2(x, args):
    x, y, z, vx, vy, vz, mu, J2, Cd, R_0x, R_0y, R_0z, R_1x, R_1y, R_1z, R_2x, R_2y, R_2z = x
    area, mass, rho_0, r_0, H, omega, R = args
    
    def P(arg, l):
        if l == 0:
            return 1
        elif l == 1:
            return arg
        else:
            return ((2*l - 1)*arg*P(arg, l-1) - (l-1)*P(arg, l-2))/l

    r_mag = sqrt(x**2 + y**2 + z**2)
    P2 = P(z/r_mag, 2)

    U0 = -mu/r_mag
    U2 = (mu/r_mag)*(R/r_mag)**2*P2*J2
    
    def differentiate(U, arg):
        u_arg = diff(U, arg)
        return simplify(u_arg)

    U0_x, U0_y, U0_z = differentiate(U0, x), differentiate(U0, y), differentiate(U0, z)
    U2_x, U2_y, U2_z = differentiate(U2, x), differentiate(U2, y), differentiate(U2, z)

    rho = rho_0*exp((-(r_mag - r_0)/H))

    # Velocity of atmosphere 

    # r_atmos_dot_ECI = r_atmos_prime_ECI(?) + omega_ECEF/ECI x r_atmos_ECI
    # r_atmos_dot_ECI = 0_ECI + (...)
    v_atmos_x = -omega*y
    v_atmos_y = omega*x
    v_atmos_z = 0.0
    
    v_rel_x = vx - v_atmos_x
    v_rel_y = vy - v_atmos_y
    v_rel_z = vz - v_atmos_z

    v_rel_mag = sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
    F_drag = 1.0/2.0*rho*v_rel_mag*Cd*area/mass

    a_x = -(U0_x + U2_x) - F_drag*v_rel_x
    a_y = -(U0_y + U2_y) - F_drag*v_rel_y
    a_z = -(U0_z + U2_z) - F_drag*v_rel_z

    mu_dot = 0.0
    J2_dot = 0.0
    Cd_dot = 0.0

    R_0x_d, R_0y_d, R_0z_d = -omega*R_0y, omega*R_0x, 0.0
    R_1x_d, R_1y_d, R_1z_d = -omega*R_1y, omega*R_1x, 0.0
    R_2x_d, R_2y_d, R_2z_d = -omega*R_2y, omega*R_2x, 0.0


    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    mu_dot, J2_dot, Cd_dot, 
                    R_0x_d, R_0y_d, R_0z_d, 
                    R_1x_d, R_1y_d, R_1z_d, 
                    R_2x_d, R_2y_d, R_2z_d]).tolist() 

def f_scenario_2(x, args):
    x_sc_E, y_sc_E, z_sc_E, vx, vy, vz, Cr = x
    x_E_sun, y_E_sun, z_E_sun, A2M, flux, mu, mu_sun, AU, c = args
    
    # from earth to sun (E/S)
    x_sun_E = -x_E_sun
    y_sun_E = -y_E_sun
    z_sun_E = -z_E_sun

    # from sun to sc (B/S)
    x_sc_sun = x_sc_E + x_E_sun
    y_sc_sun = y_sc_E + y_E_sun
    z_sc_sun = z_sc_E + z_E_sun

    # from sun to sc (S/B)
    x_sun_sc = -x_sc_sun
    y_sun_sc = -y_sc_sun
    z_sun_sc = -z_sc_sun

    r_sc_E_mag   = sqrt(x_sc_E**2   + y_sc_E**2   + z_sc_E**2)
    r_E_sun_mag  = sqrt(x_sun_E**2  + y_sun_E**2  + z_sun_E**2)
    r_sc_sun_mag = sqrt(x_sc_sun**2 + y_sc_sun**2 + z_sc_sun**2)

    # Gravity

    # 2BP
    a_x_grav_E = -mu*x_sc_E/r_sc_E_mag**3
    a_y_grav_E = -mu*y_sc_E/r_sc_E_mag**3
    a_z_grav_E = -mu*z_sc_E/r_sc_E_mag**3

    # 3BP Perturbation
    a_x_grav_sun = mu_sun*(x_sun_sc/r_sc_sun_mag**3 - x_sun_E/r_E_sun_mag**3)
    a_y_grav_sun = mu_sun*(y_sun_sc/r_sc_sun_mag**3 - y_sun_E/r_E_sun_mag**3)
    a_z_grav_sun = mu_sun*(z_sun_sc/r_sc_sun_mag**3 - z_sun_E/r_E_sun_mag**3)

    # SRP

    # P = flux/c
    # scale = (AU/r_sc_sun_mag)**2

    # a_x_SRP = -Cr*P*scale*A2M*(x_sun_sc/r_sc_sun_mag)
    # a_y_SRP = -Cr*P*scale*A2M*(y_sun_sc/r_sc_sun_mag)
    # a_z_SRP = -Cr*P*scale*A2M*(z_sun_sc/r_sc_sun_mag)

    P = flux/c
    a_x_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(x_sun_sc/r_sc_sun_mag)
    a_y_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(y_sun_sc/r_sc_sun_mag)
    a_z_SRP = -Cr*P/r_sc_sun_mag**2*A2M*(z_sun_sc/r_sc_sun_mag)
   
    a_x = a_x_grav_E + a_x_grav_sun + a_x_SRP
    a_y = a_y_grav_E + a_y_grav_sun + a_y_SRP
    a_z = a_z_grav_E + a_z_grav_sun + a_z_SRP


    Cr_dot = 0.0

    # return np.array([x_sc_E, y_sc_E, z_sc_E,
    #                 x_E_sun, y_E_sun, z_E_sun,
    #                 vx, vy, vz, 
    #                 x_sc_sun, y_sc_sun, z_sc_sun, 
    #                 a_x_grav_E, a_y_grav_E, a_z_grav_E, 
    #                 a_x_grav_sun, a_y_grav_sun, a_z_grav_sun, 
    #                 a_x_SRP, a_y_SRP, a_z_SRP, 
    #                 Cr_dot, 
    #                 ]) 

    return np.array([vx, vy, vz, 
                    a_x, a_y, a_z, 
                    Cr_dot, 
                    ]).tolist() 

