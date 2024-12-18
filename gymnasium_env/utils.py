import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp         


def magnitude_to_components(vector_maginitude, angle):
    x_component = np.cos(angle)*vector_maginitude
    y_component = np.sin(angle)*vector_maginitude
    return x_component, y_component

def lift_coefficient():
    C_L = 1.0
    return C_L

def drag_coefficient():
    C_D = 0.3
    return C_D

def lift(C_L, rho, S, v_a, alpha):
    L = (C_L*rho*S*np.linalg.norm(v_a)**2)*np.sin(alpha)
    return L

def drag(C_D, rho, S, v_a, alpha):
    D = (C_D*rho*S*np.linalg.norm(v_a)**2)*np.cos(alpha)
    return D
    

def angle_of_atack(v_b, v_a):
    alpha = np.arccos(-(np.dot(v_a,v_b))/(np.linalg.norm(v_a)*np.linalg.norm(v_b)))
    return alpha 

def build_initial_conditions(initial_state, initial_mag_v_b, theta_b):
    x_0,y_0 = initial_state
    v_x_0, v_y_0 = magnitude_to_components(initial_mag_v_b, theta_b)

    return np.array([x_0, y_0, v_x_0, v_y_0])

def system_ode(t, z, theta_b, theta_w, mag_w_v_t, fixed_param):
    
    m_b, S, rho = fixed_param
    
    x, y, v_x, v_y = z

    v_t_x, v_t_y = magnitude_to_components(mag_w_v_t, theta_w)
    v_t = np.array([v_t_x, v_t_y])
    
    v_b = np.array([v_x, v_y])
    v_a = v_t - v_b
    
    alpha = angle_of_atack(v_b, v_a)
    
    C_L, C_D = lift_coefficient(), drag_coefficient()
    
    L = lift(C_L, rho, S, v_a, alpha)
    D = drag(C_D, rho, S, v_a, alpha)
    
    F = (L - D)/m_b
    
    F_x, F_y = magnitude_to_components(F, theta_b)      
    
    # Define the ODE system
    dx_dt = v_x
    dy_dt = v_y
    dv_x_dt = F_x
    dv_y_dt = F_y
    
    return np.array([dx_dt, dy_dt, dv_x_dt, dv_y_dt])

def rkf45_step(f, t, z, h, args=()):


    k1 = h * f(t, z, *args)
    k2 = h * f(t + (1/4) * h, z + (1/4) * k1, *args)
    k3 = h * f(t + (3/8) * h, z + (3/32) * k1 + (9/32) * k2, *args)
    k4 = h * f(t + (12/13) * h, z + (1932/2197) * k1 - (7200/2197) * k2 + (7296/2197) * k3, *args)
    k5 = h * f(t + h, z + (439/216) * k1 - 8 * k2 + (3680/513) * k3 - (845/4104) * k4, *args)
    k6 = h * f(t + (1/2) * h, z - (8/27) * k1 + 2 * k2 - (3544/2565) * k3 + (1859/4104) * k4 - (11/40) * k5, *args)

    z_next = z + (16/135) * k1 + (6656/12825) * k3 + (28561/56430) * k4 - (9/50) * k5 + (2/55) * k6


    return z_next

def terminal_state_check(boat_pos, ternminal_state = (0,0), epislon = 1):
    x,y = boat_pos
    a,b = ternminal_state
    if (x-a)**2 + (y-b)**2 <= epislon**2:
        return True
    
    return False