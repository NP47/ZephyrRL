import math
import random
import time

import numpy as np
from scipy.integrate import solve_ivp 

import pygame


screen_width = 800
screen_height = 800

def rkf45_step(f, t, z, h, args=()):


    k1 = h * f(t, z, *args)
    k2 = h * f(t + (1/4) * h, z + (1/4) * k1, *args)
    k3 = h * f(t + (3/8) * h, z + (3/32) * k1 + (9/32) * k2, *args)
    k4 = h * f(t + (12/13) * h, z + (1932/2197) * k1 - (7200/2197) * k2 + (7296/2197) * k3, *args)
    k5 = h * f(t + h, z + (439/216) * k1 - 8 * k2 + (3680/513) * k3 - (845/4104) * k4, *args)
    k6 = h * f(t + (1/2) * h, z - (8/27) * k1 + 2 * k2 - (3544/2565) * k3 + (1859/4104) * k4 - (11/40) * k5, *args)

    z_next = z + (16/135) * k1 + (6656/12825) * k3 + (28561/56430) * k4 - (9/50) * k5 + (2/55) * k6


    return z_next


def rotate(x, y, angle):
    new_x = math.cos(angle) * x - math.sin(angle) * y
    new_y = math.sin(angle) * x + math.cos(angle) * y
    return new_x, new_y


def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def is_in_bounds(x, y):
    return 0 <= x <= screen_width and 0 <= y <= screen_height


scale_factor = distance((0, 0), (screen_width, screen_height))

def magnitude_to_components(vector_maginitude, angle):
    x_component = np.cos(angle)*vector_maginitude
    y_component = np.sin(angle)*vector_maginitude
    return x_component, y_component

def lift_coefficient():
    C_L = 1.0
    return C_L

def drag_coefficient():
    C_D = 0.6
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


def system_ode(t, z, theta_b, theta_s, theta_w, mag_w_v_t, fixed_param):
    
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

class BasicSailboat(object):
    def __init__(self, x=0, y=0, theta_boat=0, v_x = 0.0001, v_y = 0.0001):
        self.x, self.y = x, y
        self.v_x, self.v_y = v_x, v_y
        self.speed = np.sqrt(v_x**2 + v_y**2)
        self.theta_boat = theta_boat
    
        #sail params
        self.theta_sail = 0 - np.pi/2
    
        #Wind parameters
        mag_w_v_t = 4 
        theta_w_t = np.pi 

        #boat parameters
        m_b = 1000
        S = 5 #sail size
        
        rho_a = 1.225 #air density 

        #fixed parameters
        self.fixed_param = np.array([m_b, S, rho_a])
            
        self.t_step = 0.5
        self.t = 0
        self.speed = 0
        self.angle = 0
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]
        

        #self.rudder_angle = 180
        self.length = 20
        self.width = 50

        # Sail geometry: define the four corners of the rectangular sail
        self.sail_height = self.length * 1.1  # Sail height proportional to boat length
        self.sail_width = self.width * 0.1    # Sail width proportional to boat width

        # Sail geometry: define the four corners of the rectangular sail
        self.rudder_height = self.length * 1  # Sail height proportional to boat length
        self.rudder_width = self.width * 0.2    # Sail width proportional to boat width

    def step(self, delta_theta_b, wind_info):
        self.t+=self.t_step
        
        delta_theta_b = (max(-1, min(1, delta_theta_b[0])) * 0.8e-1)% (2 * np.pi)
        
        self.theta_wind, self.wind_speed = wind_info
        
        
        args=(self.theta_boat, self.theta_sail, self.theta_wind, self.wind_speed, self.fixed_param)
        self.theta_boat += delta_theta_b
        args = (self.theta_boat,) + args[1:]
        
        z = [self.x, self.y, self.v_x, self.v_y]
            
    
        self.x, self.y, self.v_x, self.v_y = rkf45_step(system_ode, self.t_step, z, self.t_step, args)

        self.speed = np.sqrt(self.v_x**2 + self.v_y**2)
        
  
        
    def draw_boat(self, surface):
        self.points = [
        (0, -self.length / 2),           # Front tip (pointy end)
        (self.width / 2, 0),             # Right side curve
        (0, self.length / 2),            # Back end (rounded)
        (-self.width / 2, 0)             # Left side curve
        ]
        
        points = [rotate(x, y, self.theta_boat) for x, y in self.points]
        points = list([(x + self.x, y + self.y) for x, y in points])
        pygame.draw.polygon(surface, (225, 237, 233), points) 

    def draw_sail(self, surface):
        # Sail base center (aligned with the middle of the boat)
        sail_base_x, sail_base_y = self.x, self.y


        # Define the four corners of the rectangle
        sail_points = [
            (-self.sail_width / 2, 0),  (self.sail_width / 2, 0),               # Bottom-right corner
            (self.sail_width / 2, -self.sail_height),    # Top-right corner
            (-self.sail_width / 2, -self.sail_height)    # Top-left corner
        ]

        # Rotate the sail based on self.sail_angle + self.angle
        rotated_sail_points = [
            rotate(x, y, self.theta_sail + self.theta_boat) for x, y in sail_points
        ]

        # Translate the rotated sail to the sail base position
        sail_points_translated = [
            (x + sail_base_x, y + sail_base_y) for x, y in rotated_sail_points
        ]

        # Draw the sail as a rectangle
        pygame.draw.polygon(surface, (255, 0, 255), sail_points_translated)  # White sail



    def draw(self, surface):
        self.draw_boat(surface)
        self.draw_sail(surface)
        #self.draw_rudder(surface)
        
        
    def pos(self):
        return self.x, self.y


class BasicSailboatEnv(object):
    def __init__(self):
        self.sailboat = None
        self.target = None
        
        self.steps = 0
        self.surf = None
        self.done = True
       
        
        

    def reset(self):
        self.sailboat = BasicSailboat(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1), np.random.uniform(0, np.pi*2))#BasicSailboat(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1), np.pi)
        #self.target = random.randint(0, screen_width - 1), random.randint(0, screen_height - 1) random one
        
        #top_left = (screen_width - 1)/7, (screen_width - 1)/7
        #top_right = screen_width - (screen_width - 1)/7,  (screen_width - 1)/7
        #bottom_left =  (screen_width - 1)/7, screen_width - (screen_width - 1)/7
        #bottom_right = screen_width -  (screen_width - 1)/7, screen_width - (screen_width - 1)/7
        
        #possible_locatin = [top_left, top_right, bottom_left, bottom_right]
        #self.target = possible_locatin[np.random.randint(0,4)]
        
        self.target = (screen_width - 1)/2, (screen_width - 1)/2
        
        self.done = False
        self.steps = 0
        
        self.wind_speed = 10 #np.pi + 0.1*np.random.uniform(-np.pi,np.pi)
        self.theta_wind = np.pi #10 + np.random.randint(-10,10)
        
        
        return self.__state__()

    def draw_surf(self):
        self.surf.fill((6, 66, 115))
        pygame.draw.circle(self.surf, (194, 178, 128), self.target, 30)
        pygame.draw.circle(self.surf, (150, 90, 62), (self.target[0] + 10, self.target[1] - 5) , 5)
    
    def draw_wind_arrow(self):
        arrow_x = self.surf.get_width() * 0.9
        arrow_y = self.surf.get_height() * 0.1



        
        # Arrow dimensions
        arrow_length = 48
        arrow_tip_x = arrow_x + arrow_length * math.cos(self.theta_wind)
        arrow_tip_y = arrow_y + arrow_length * math.sin(self.theta_wind)
        arrow_base_x = arrow_x
        arrow_base_y = arrow_y

        # Draw the arrow line
        pygame.draw.line(self.surf, (255, 0, 0), (arrow_base_x, arrow_base_y), (arrow_tip_x, arrow_tip_y), 3)

        # Arrowhead points
        arrowhead_length = 10
        arrowhead_angle = math.pi / 6  # 30 degrees
        left_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(self.theta_wind - arrowhead_angle)
        left_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(self.theta_wind - arrowhead_angle)
        right_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(self.theta_wind + arrowhead_angle)
        right_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(self.theta_wind + arrowhead_angle)

         # Draw the arrowhead
        pygame.draw.polygon(self.surf, (255, 0, 0), [(arrow_tip_x, arrow_tip_y),
                                                (left_arrowhead_x, left_arrowhead_y),
                                                (right_arrowhead_x, right_arrowhead_y)])

        # Display wind speed next to the arrow
        font = pygame.font.Font(None, 24)
        wind_speed_text = font.render(f"{self.wind_speed} m/s", True, (255, 255, 255))
        text_offset_x = -15  # Offset to position text near the arrow
        text_offset_y = 60
        self.surf.blit(wind_speed_text, (arrow_x + text_offset_x, arrow_y + text_offset_y))

        pygame.draw.circle(self.surf, (0, 0, 0), (arrow_x, arrow_y), 50, width=2)

    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.draw_surf()
        self.sailboat.draw(self.surf)
        self.draw_wind_arrow()
        pygame.display.flip()


    def step(self, action):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        delta_theta_boat = action
        prev_dist = distance(self.sailboat.pos(), self.target)
        
        self.sailboat.step(delta_theta_boat, (self.theta_wind, self.theta_wind))
        dist = distance(self.sailboat.pos(), self.target)
    
        
        r = (prev_dist - dist)/scale_factor - 0.001

        if not is_in_bounds(self.sailboat.x, self.sailboat.y) or self.steps > 1000:
            r -= 1
            self.done = True

        if dist < 30:
            r += 1
            self.done = True
        return self.__state__(), r, self.done, None

    def __state__(self):
  
       
        distance_to_target = distance(self.sailboat.pos(), self.target)
        distance_to_target /= scale_factor
    

        return np.array([self.sailboat.x / screen_width, self.sailboat.y / screen_height, self.sailboat.v_x, self.sailboat.v_y,
                           self.sailboat.speed ,distance_to_target, self.theta_wind, self.wind_speed])



if __name__ == "__main__":
    env = BasicSailboatEnv()

    while True:
        state = env.reset()
        done = False
        while not done:
            delta_theta_boat = 0
            state, r, done, _ = env.step(delta_theta_boat)
            print(r, state[0], state[1],state[4])
            env.draw()
            time.sleep(1/60)