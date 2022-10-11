import numpy as np
from sympy import *
########################
# UKF Example Dynamics #
########################
def f_spring(x, args):
    x, v_x = x
    k, eta = args
    a_x = -k*x
    return np.array([v_x, a_x]).tolist() 

def f_spring_duffing(x, args):
    x, v_x = x
    k, eta = args
    a_x = -k*x - eta*x**3
    return np.array([v_x, a_x]).tolist() 

