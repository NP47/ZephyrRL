########## libraries
import numpy as np
import math
import scipy.io
import pickle
from sympy import symbols



##########################SPEED CALCULATION

def calculate_lift_force(deg, V):
    # Constants
    A = 100 # sail area big sailboat
    rho = 1.225  # Air density in kg/m^3 at sea
    rad = math.radians(deg)
    CL0 = 0.3  # aprox lift coefficient at zero angle of attack for a sail
    dCL_dalpha = 2 * math.pi  # aprox lift curve slope for thin airfoils

    # Calculate lift coefficient
    CL = CL0 + dCL_dalpha * alpha_rad

    # Calculate lift force
    L = CL * 0.5 * rho * V**2 * A

    return L
    

def knots_to_mps(knots):
    return knots * 0.514444

def get_speed(sail, wind):
    knots = calculate_lift_force(sail, wind)/3620052 #constant approximation simplification from Hallber-Rassy hulls and sails
    mps = knots_to_mps(knots)
    return mps


##########################MANEOUVER

########### functions
#from calculate_speed import get_speed #####calculate_speed.py

##Physics constants
g = 9.81 #m/s^2

##Public Ship characteristics Eco Series
# To extract the data from the pickle file
with open('ship.pkl', "rb") as f:
    # Load the data from the file
    loaded_data = pickle.load(f)

# Assign the values to variables with the same names
L, B, T, CB = loaded_data["L"], loaded_data["B"], loaded_data["T"], loaded_data["CB"]

######################################################HYDRODYNAMIC FORCES#########################################

def maneuver(u, v, r, delta, shallow, sail, wind):
    # Load the lever position to speed mapping from the pickle file

    # Coefficients from Ship Geometry.
    with open('ShipGeometry.pkl', 'rb') as f:
        data = pickle.load(f)

    m = data['m']
    xG = data['xG']
    Iz = data['Iz']

    Ui = get_speed(sail, wind)

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


def trajectory(rudder, sail, wind)

    ##Physics constants
    g = 9.81 #m/s^2
    
    ##Time Step ## 1 minute for ease of calculation
    time = 1

    ##Public Ship characteristics example maneouvrability standard ship
    # To extract the data from the pickle file
    with open('ship.pkl', "rb") as f:
        # Load the data from the file
        loaded_data = pickle.load(f)

    # Assign the values to variables with the same names
    L, B, T, CB = loaded_data["L"], loaded_data["B"], loaded_data["T"], loaded_data["CB"]

    #Coefficients from Ship Geometry.
    with open('ShipGeometry.pkl', 'rb') as f:
        data = pickle.load(f)

    m = data['m']
    xG = data['xG']
    Iz = data['Iz']

    time_data = []
    rudder_data = []
    sail_data = []
    wind_data = []
    
    shallow_water = []
    
    # ship not in shallow water
    shallow_water.append(0)
    
    # Append the data to the separate arrays
    time_data.append(time)
    rudder_data.append(rudder)
    sail_data.append(sail)
    wind_data.append(wind)
    
    num_data_sets = len(time_data)
    total_time = sum(time_data)
    
    ####################################################################################
    ####Operating characteristics

    # Rudder Data
    delta_dot = 2

    time_total = total_time*60
    time_step = 0.5
    
    ###############################################################################################
    time = np.transpose(np.arange(0, time_total+time_step, time_step)) # creating time matrix for test

    # Creating 0 Matrix
    delta_rudder = np.zeros((len(time), 1))
    xo = np.zeros((len(time), 1))
    yo = np.zeros((len(time), 1))
    u = np.zeros((len(time), 1))
    v = np.zeros((len(time), 1))
    r = np.zeros((len(time), 1))
    psi = np.zeros((len(time), 1))
    ud = np.zeros((len(time)-1, 1))
    vd = np.zeros((len(time)-1, 1))
    rd = np.zeros((len(time)-1, 1))

    k=0
    for j in range(num_data_sets):
        time_data_s=time_data[j]*60
        time_part = np.transpose(np.arange(0, time_data_s+time_step, time_step)) # creating time matrix for test
        delta = rudder_data[j]
        print(rudder_data[j])
        print(delta)
        sail = sail_data[j]
        wind = wind_data[j]
        Ui = get_speed(sail_data[j], wind_data[j])
        shallow = shallow_water[j]
        u[k] = Ui
        delta_rudder[k] = delta
        
        for i in range(len(time_part)-1):
         # Initial Check for rudder degree
            if abs(delta_rudder[k]) > abs(delta):
                delta_rudder[k] = delta
            if delta_rudder[k] < 0:
                delta_dot_app = -delta_dot
            elif delta_rudder[k] > 0:
                delta_dot_app = delta_dot
            else:
                delta_dot_app = 0
                
            delta_rudder[k+1] = delta_rudder[k] + time_step*delta_dot_app # basically, rudder angle is time*(rudder acceleration)
            udi, vdi, rdi = maneuver(u[k], v[k], r[k], delta_rudder[k], shallow, sail, wind) # reading data from function
            ud[k,0] = udi # reading this value (acceleration) from function
            vd[k,0] = vdi # reading this value (acceleration) from function
            rd[k,0] = rdi # reading this value (acceleration) from function

            u[k+1,0] = u[k,0] + time_step * udi # velocity = acc * time. 
            v[k+1,0] = v[k,0] + time_step * vdi
            r[k+1,0] = r[k,0] + time_step * rdi

            psi[k+1,0] = psi[k,0] + time_step * r[k,0]
            xd = u[k]*np.cos(psi[k]) - v[k] * np.sin(psi[k])
            yd = u[k]*np.sin(psi[k]) + v[k] * np.cos(psi[k])
            xo[k+1,0] = xo[k,0] + time_step * xd
            yo[k+1,0] = yo[k,0] + time_step * yd
            k += 1
            
            # Find the final points
            final_xo = xo[-1, 0]
            final_yo = yo[-1, 0]
            
            
    return final_xo, final_yo

##################################################################################################################################################

class Sailboat(object):
    def _init_(self, x, y, angle_turn_limit, speed_limits, actions_scale_1, action_scale_2):
        #initialize the starting conditions of the boat
        
        #trajectory properties + other basic pyshicl properties needed f
        self.x = x
        self.y = y
        
        self.speed = 0
        self.angle = 0
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]
        
        
        
        #limitis 
        
        #self.speed_limits = speed_limits
        #self.angle_turn_limit = angle_turn_limit
        
        
        #action scales 
        #self.actions_scale_1 = actions_scale_1
        #self.action_scale_2 = action_scale_2


    def step(self, rudder, sail, wind):
        #essentially applying the actions, the ruder or sail angle change, an dhow that affects the position

        xt, _ = trajectory(rudder, sail, wind)
        _, yt = trajectory(rudder, sail, wind)

        self.speed = get_speed(sail, wind)
        #self.angle = (self.angle + steer) % (2 * math.pi)
        self.x += xt
        self.y += yt