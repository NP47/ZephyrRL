########## libraries
import numpy as np
import math
import scipy.io
import pickle
from sympy import symbols
import sys
import os

##Physics constants
g = 9.81 #m/s^2
step_time = 1 # minute per step.


##########################SPEED CALCULATION

def calculate_lift_force(deg_sail, WV, deg_boat):
    deg = deg_sail - deg_boat
    # Constants
    A = 100 # sail area big sailboat
    rho = 1.225  # Air density in kg/m^3 at sea
    rad = math.radians(deg)
    CL0 = 0.3  # aprox lift coefficient at zero angle of attack for a sail
    dCL_dalpha = 2 * math.pi  # aprox lift curve slope for thin airfoils

    # lift coefficient
    CL = CL0 + dCL_dalpha * rad
    # lift force
    lift_force = CL * 0.5 * rho * WV**2 * A

    return lift_force
    

#def knots_to_mps(knots):
#    return knots * 0.514444

def get_speed(in_v, sail, wind, angle):
    rho_H2O = 1025  # density of seawater kg/m^3
    Cd = 0.8  # Drag coefficient, from reference ship.
    A_boat = 5.11*2.43  # m^2 simplified from Hallberg-Rassy reference sailboat
    F_drag = 0.5 * rho_H2O * in_v**2 * Cd * A_boat
    acceleration = calculate_lift_force(sail, wind, angle)-F_drag
    speed = in_v + acceleration*step_time*60 #m/s
    return speed


##########################MANEOUVER

##Public Ship characteristics example maneouvrability standard ship
# To extract the data from the pickle file
with open('ship.pkl', "rb") as f:
    # Load the data from the file
    loaded_data = pickle.load(f)

# Assign the values to variables with the same names
L, B, T, CB = loaded_data["L"], loaded_data["B"], loaded_data["T"], loaded_data["CB"]

######################################################HYDRODYNAMIC FORCES#########################################

def maneuver(Ui, u, v, r, delta, shallow):
    # Load the lever position to speed mapping from the pickle file

    # Coefficients from Ship Geometry.
    with open('ShipGeometry.pkl', 'rb') as f:
        data = pickle.load(f)

    m = data['m']
    xG = data['xG']
    Iz = data['Iz']

    fn = Ui / math.sqrt(g * L)
    
    Ut = math.sqrt((u**2 + v**2)) # ship speed
    du = (u - Ui) / Ut
    u = u / Ut
    v = v / Ut
    r = r * L / Ut
    delta = math.radians(delta)
   
    # Shallow/Deep water coefficients------LEGACY from thesis----kept in case we want to add it later
    if shallow == 1:
        with open('Ns.pkl', 'rb') as f:
            N = pickle.load(f)
        with open('Xs.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('Ys.pkl', 'rb') as f:
            Y = pickle.load(f)
    else:
        with open('N.pkl', 'rb') as f:
            N = pickle.load(f)
        with open('X.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('Y.pkl', 'rb') as f:
            Y = pickle.load(f)

    ud, vd, rd = symbols('ud vd rd')

    ###### Mass matrix
    # Specify the shape of the array
    shapem = (3, 3)

    # Create a 3x3 array of zeros
    Mass = np.zeros(shapem)

    # Fill in the values
    Mass[0, 0] = m - X['ud'] - X['udu2'] * du**2
    Mass[1, 1] = m - Y['vd'] - Y['vdv2'] * v**2
    Mass[1, 2] = m * xG - Y['rd'] - Y['rdr2'] * r**2
    Mass[2, 1] = m * xG - N['vd'] - N['vdv2'] * v**2
    Mass[2, 2] = Iz - N['rd'] - N['rdr2'] * r**2
    
    Xext_Damp = (m * v * r + m * xG * (r ** 2) + X["u"] * du + X["u2"] * (du ** 2) + X["u3"] * (du ** 3) + X["v"] * v + X["v2"] * (v ** 2) + X["v3"] * (v ** 3) + X["r"] * r + X["r2"] * (r ** 2) + X["r3"] * (r ** 3) + X["dt"] * delta + X["dt2"] * (delta ** 2) + X["dt4"] * (delta ** 4) + X["vr"] * v * r + X["rdt"] * r * delta + X["vu"] * v * du + X["vdt"] * v * delta + X["ru"] * r * du + X["v2u"] * (v ** 2) * du + X["dt2u"] * (delta ** 2) * du + X["r2u"] * (r ** 2) * du + X["vu2"] * v * (du ** 2) + X["dtu2"] * delta * (du ** 2) + X["v2dt"] * (v ** 2) * delta + X["r2dt"] * (r ** 2) * delta + X["dt3u"] * (delta ** 3) * du + X["dt3u"] * (delta ** 3) * du + X["v3u"] * (v ** 3) * du + X["r3u"] * (r ** 3) * du)
             
    Yext_Damp =  Y["o"] - m * u * r + Y["u"] * du + Y["v"] * v + Y["v3"] * (v ** 3) + Y["r"] * r + Y["r2"] * (r ** 2) + Y["r3"] * (r ** 3) + Y["dt"] * delta + Y["dt2"] * (delta ** 2) + Y["dt3"] * (delta ** 3) + Y["dt4"] * (delta ** 4) + Y["dt5"] * (delta ** 5) + Y["vr2"] * (r ** 2) * v + Y["vdt2"] * (delta ** 2) * v + Y["ru"] * r * du + Y["dtu"] * delta * du + Y["ru2"] * r * (du ** 2) + Y["rv2"] * r * (v ** 2) + Y["rdt2"] * r * (delta ** 2) + Y["dtv2"] * delta * (v ** 2) + Y["dtr2"] * delta * (r ** 2) + Y["dtu2"] * delta * (du ** 2) + Y["dt3u"] * (delta ** 3) * du + Y["v3r"] * (v ** 3) * r + Y["r3u"] * (r ** 3) * du + Y["vv"] * v * abs(v) + Y["rr"] * r * abs(r) + Y["dtdt"] * delta * abs(delta)
    
    Next_Damp = N["o"] - m * xG * u * r + N["u"] * du + N["v"] * v + N["v2"] * (v ** 2) + N["v3"] * (v ** 3) + N["r"] * r + N["r2"] * (r ** 2) + N["r3"] * (r ** 3) + N["dt"] * delta + N["dt2"] * (delta ** 2) + N["dt3"] * (delta ** 3) + N["dt4"] * (delta ** 4) + N["dt5"] * (delta ** 5) + N["vu"] * v * du + N["ru"] * r * du + N["dtu"] * delta * du + N["vr"] * v * r + N["vr2"] * v * (r ** 2) + N["vdt2"] * v * (delta ** 2) + N["ru2"] * r * (du ** 2) + N["rv2"] * r * (v ** 2) + N["rdt2"] * r * (delta ** 2) + N["dtv2"] * delta * (v ** 2) + N["dtr2"] * delta * (r ** 2) + N["dtu2"] * delta * (du ** 2) + N["dt2u"] * (delta ** 2) * du + N["dt3u"] * (delta ** 3) * du + N["v3u"] * (v ** 3) * du + N["r3u"] * (r ** 3) * du + N["vv"] * v * abs(v) + N["rr"] * r * abs(r) + N["dtdt"] * delta * abs(delta)

    shapef = (3, 1)
    Fext = np.zeros(shapef)
    Fext[0, 0] = Xext_Damp
    Fext[1, 0] = Yext_Damp
    Fext[2, 0] = Next_Damp
    
    result = np.linalg.solve(Mass, Fext)
    
    ud = result[0] * (Ui**2) / L
    vd = result[1] * (Ui**2) / L
    rd = result[2] * (Ui**2) / (L**2)
    
    return ud, vd, rd

############################################################TRAJECTORY SECTION CALCULATION


def trajectory(velocity, rudder, depth):

    # shallow water
    shallow = depth
      
    ####################################################################################
    ####Operating characteristics

    # Rudder Data
    delta = -35
    delta_dot = 2

    time_total = step_time*60
    time_step = 0.5
    
    ###############################################################################################
    time = np.transpose(np.arange(0, time_total+time_step, time_step)) # creating time matrix for test

    # Creating 0 Matrix
    delta_rudder = np.zeros((len(time), 1))
    xo = np.zeros((len(time), 1))
    yo = np.zeros((len(time), 1))
    u = np.zeros((len(time), 1)) #axial velocity
    v = np.zeros((len(time), 1)) #side velocity
    r = np.zeros((len(time), 1)) #yaw velocity
    psi = np.zeros((len(time), 1))
    ud = np.zeros((len(time)-1, 1))
    vd = np.zeros((len(time)-1, 1))
    rd = np.zeros((len(time)-1, 1))

    delta_rudder = rudder
    Ui = velocity
    u[0] = Ui
      
    for i in range(len(time)-1):
     # Initial Check for rudder degree to not exceed 35 degree
        if abs(delta_rudder) > abs(delta):
            delta_rudder[i] = rudder
        if delta_rudder[i] < 0:
            delta_dot_app = -delta_dot
        elif delta_rudder[i] > 0:
            delta_dot_app = delta_dot
        else:
            delta_dot_app = 0
            
        delta_rudder[i+1] = delta_rudder[i] + time_step*delta_dot_app # rudder angle = time*(rudder acceleration)
        udi, vdi, rdi = maneuver(Ui, u[i], v[i], r[i], delta_rudder[i], shallow) # reading data from function
        ud[i,0] = udi # reading this value (acceleration : axial) from function
        vd[i,0] = vdi # reading this value (acceleration : side) from function
        rd[i,0] = rdi # reading this value (acceleration : yaw) from function

        # velocity = acc * time.
        u[i+1,0] = u[i,0] + time_step * udi # axial
        v[i+1,0] = v[i,0] + time_step * vdi # side
        r[i+1,0] = r[i,0] + time_step * rdi # yaw

        # transformation from body-fixed coordinates to earth-fixed coordinates with point reference 0
        psi[i+1,0] = psi[i,0] + time_step * r[i,0]
        xd = u[i]*np.cos(psi[i]) - v[i] * np.sin(psi[i])
        yd = u[i]*np.sin(psi[i]) + v[i] * np.cos(psi[i])
        xo[i+1,0] = xo[i,0] + time_step * xd
        yo[i+1,0] = yo[i,0] + time_step * yd
            
        # Find the final displacement, in metres, in reference to the initial point 0,0
        final_xo = xo[i, 0]
        final_yo = yo[i, 0]
        final_yaw = math.degrees(psi[i,0]) #radians to degrees, yaw
        
        # Average velocity in each coordinate for the step in m/s
        v_x = final_xo / time_total
        v_y = final_yo / time_total
            
    return final_xo, final_yo, v_x, v_y, final_yaw
