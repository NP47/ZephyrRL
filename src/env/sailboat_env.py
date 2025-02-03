import sys
import os

# Add the project directory (parent of testing and algos) to sys.path
sys.path.append(os.path.abspath(".."))

import random
import time

import numpy as np
import pygame

from src.utils.physics_simulation import rkf45_step, system_ode
from src.utils.rander import draw_sail, draw_boat, draw_wind_arrow

screen_width = 800
screen_height = 800




def distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def is_in_bounds(x, y):
    return 0 <= x <= screen_width and 0 <= y <= screen_height


scale_factor = distance((0, 0), (screen_width, screen_height))


class Sailboat(object):
    def __init__(self, x=0, y=0, theta_boat=0, theta_sail = 0,  v_x = 0.0001, v_y = 0.0001):
        self.x, self.y = x, y
        self.v_x, self.v_y = v_x, v_y
        self.speed = np.sqrt(v_x**2 + v_y**2)
        
        self.theta_boat = theta_boat
    
        #sail params
        self.theta_sail = theta_sail - np.pi/2 # 
        
        
        m_b = 500      #boat weight 
        S = 5           #sail size
        rho_a = 1.225   #air density 

        #fixed parameters
        self.fixed_param = np.array([m_b, S, rho_a])
            
        self.t_step = 0.5
    
        
    
        

        


    def step(self, actions, wind_info):
        
        delta_theta_b = actions
        
        #print(delta_theta_b)
        delta_theta_b =  max(-np.pi/6, min(np.pi/6, delta_theta_b)) * 2e-1#(max(-np.pi/8, min(1, delta_theta_b)) * 0.8e-1)#% (2 * np.pi) #np.random.choice([-1, 1])*
        #print(-)
        
        
        
        delta_theta_b
        
        self.theta_wind, self.wind_speed = wind_info
        
        
        args=(self.theta_boat, self.theta_sail, self.theta_wind, self.wind_speed, self.fixed_param)
        self.theta_boat += delta_theta_b
        args = (self.theta_boat,) + args[1:]
        
        z = [self.x, self.y, self.v_x, self.v_y]
            
    
        self.x, self.y, self.v_x, self.v_y = rkf45_step(system_ode, z, self.t_step, args)

        self.speed = np.sqrt(self.v_x**2 + self.v_y**2)
        
  
    def draw(self, surface, lenght = 20, width = 50):
        

        draw_boat(self.x, self.y,self.theta_boat, width, lenght, surface)
        draw_sail(self.x, self.y,  self.theta_sail, self.theta_boat, width*0.1, lenght* 1.1 , surface)
        
        
    def pos(self):
        return self.x, self.y


class SailboatEnv(object):
    def __init__(self, wind_settings):
        
        self.wind_settings = wind_settings
        self.sailboat = None
        self.target = None
        
        self.steps = 0
        
        self.done = True
       
        
        self.surf = None

    def reset(self):
        self.sailboat = Sailboat(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1), np.random.uniform(0, np.pi*2))#BasicSailboat(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1), np.pi)
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
        
        if self.wind_settings['type'] == 'fixed':
            self.wind_speed = self.wind_settings['wind_speed'] #np.pi + 0.1*np.random.uniform(-np.pi,np.pi)
            self.theta_wind = self.wind_settings['theta_wind'] #10 + np.random.randint(-10,10)
        elif self.wind_settings['type'] == 'variable_per_epoch':
            self.wind_speed = np.random.uniform(5, 15)
            self.theta_wind = np.random.uniform(0, 2 * np.pi)
        
        
        return self.__state__()


    def step(self, action):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        delta_theta_boat = action[0]
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
        

    def draw_surf(self):
        self.surf.fill((6, 66, 115))
        pygame.draw.circle(self.surf, (194, 178, 128), self.target, 30)
        pygame.draw.circle(self.surf, (150, 90, 62), (self.target[0] + 10, self.target[1] - 5) , 5)
    

    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.draw_surf()
        self.sailboat.draw(self.surf)
        draw_wind_arrow(self.surf, self.theta_wind, self.wind_speed)
        pygame.display.flip()


if __name__ == "__main__":
    env = SailboatEnv()

    while True:
        state = env.reset()
        done = False
        while not done:
            delta_theta_boat = [0]
            state, r, done, _ = env.step(delta_theta_boat)
            print(r, state[0], state[1],state[4])
            env.draw()
            time.sleep(1/60)