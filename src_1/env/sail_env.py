"""
    A 2-player game.
    The blue car has to reach the Green target circle to win without leaving the screen.
    The red car has to tag the blue car to win, but is not allowed to enter the target circle (this costs points).
"""


import sys
import os

# Add the project directory (parent of testing and algos) to sys.path
sys.path.append(os.path.abspath(".."))



import math
import random
import numpy as np
import time
import pygame

screen_width = 800
screen_height = 800


def set_random_seed(seed):
    random.seed(seed)


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


def calc_angle_error_and_dist(car, pos):
    angle_to_target = math.atan2(car.y - pos[1], car.x - pos[0])
    distance_to_target = distance(car.pos(), pos)
    distance_to_target /= scale_factor
    angle_error = (car.angle - angle_to_target) % (2 * math.pi)
    angle_error -= math.pi
    angle_error /= math.pi
    return angle_error, distance_to_target


scale_factor = distance((0, 0), (screen_width, screen_height))


########## libraries
import numpy as np
import math
import scipy.io
import pickle
from sympy import symbols
import sys
import os

from env.ship import trajectory ###ship.py
from env.ship import get_speed ###ship.py



# Add the project directory (parent of testing and algos) to sys.path


##################################################################################################################################################

screen_width = 800
screen_height = 800


class Sailboat(object):
    def __init__(self, x, y):
        #initialize the starting conditions of the boat
        
        #trajectory properties + other basic pyshicl properties needed f
        self.x = x
        self.y = y
        
        self.speed = 0
        self.angle = 0
        self.rudder = 0
        self.sail = 0
        
        
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]
        
        self.length = 20
        self.width = 50

        # Sail geometry: define the four corners of the rectangular sail
        self.sail_height = self.length * 1.1  # Sail height proportional to boat length
        self.sail_width = self.width * 0.1    # Sail width proportional to boat width

        # Sail geometry: define the four corners of the rectangular sail
        self.rudder_height = self.length * 1  # Sail height proportional to boat length
        self.rudder_width = self.width * 0.2    # Sail width proportional to boat width


    def step(self, rudder, sail, wind, depth = 0):
        #essentially applying the actions, the ruder or sail angle change, an dhow that affects the position

        #depth = 0 #shallow water variable, 0=deep water // 1=shallow water

        wind_speed, wind_dir = wind
        #difference between the angle of the boat and the angle of the wind direction.
        wind_boat_angle = wind_dir - self.angle
        
        velocity = get_speed(self.speed, sail, wind_speed, wind_boat_angle)
        
        xt, yt, vx, vy, rotation = trajectory(velocity, rudder, depth)


        self.speed = np.sqrt(vx**2+vy**2)
     
        self.x += xt
        self.y += yt
        
        #angle which the boat is pointing in
        self.angle += rotation
        
        self.rudder = rudder
        self.sail = sail 
    
    def draw_boat(self, surface):
        self.points = [
        (0, -self.length / 2),           # Front tip (pointy end)
        (self.width / 2, 0),             # Right side curve
        (0, self.length / 2),            # Back end (rounded)
        (-self.width / 2, 0)             # Left side curve
        ]
        
        points = [rotate(x, y, self.angle) for x, y in self.points]
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
            rotate(x, y, self.sail + self.angle) for x, y in sail_points
        ]

        # Translate the rotated sail to the sail base position
        sail_points_translated = [
            (x + sail_base_x, y + sail_base_y) for x, y in rotated_sail_points
        ]

        # Draw the sail as a rectangle
        pygame.draw.polygon(surface, (255, 0, 255), sail_points_translated)  # White sail

    def draw_rudder(self, surface):
        rudder_base_x, rudder_base_y = self.x, self.y

        rudder_points = [
            (-self.rudder_width / 2, 0),  (self.rudder_width / 2, 0),   # Bottom-right corner
            (self.rudder_width / 2, -self.rudder_height),    # Top-right corner
            (-self.rudder_width / 2, -self.rudder_height)    # Top-left corner
        ]

        # Rotate the sail based on self.sail_angle + self.angle
        rotated_rudder_points = [
            rotate(x, y, - self.rudder + self.angle) for x, y in rudder_points
        ]

        # Translate the rotated sail to the sail base position
        rudder_points_translated = [
            (x + rudder_base_x, y + rudder_base_y) for x, y in rotated_rudder_points
        ]

        pygame.draw.polygon(surface, (120,20,20), rudder_points_translated)  # White sail

    def draw(self, surface):
        self.draw_boat(surface)
        self.draw_sail(surface)
        self.draw_rudder(surface)
       
    def pos(self):
        return self.x, self.y


class SailboatEnv(object):
    def __init__(self, speed_limits=(-2, 10), throttle_scale=0.2, steer_scale=5e-1, max_steps=1000, reversing_cost=0.0):
        self.sailboat = None
        self.target = None
        self.steps = 0
        
    
        self.wind = 5
        self.wind_dir = np.pi
        
        #limits 
        #self.speed_limits = speed_limits
        self.surf  = None
        
        #scalres
       # self.throttle_scale = throttle_scale
       # self.steer_scale = steer_scale
       # self.max_steps = max_steps
    

        #reward setting
      # self.reversing_cost = reversing_cost
        
        
        #wind 
        
        
        #rendering varaibles
        self.done = True

    def reset(self):
        #initialization of terminal state
        self.target = (screen_width - 1)/2, (screen_width - 1)/2



        #sailboat location initiazlization (change in rudder ange, change in sail angle)
        self.sailboat = Sailboat(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1))
        
        
        self.done = False
        self.steps = 0
        return self.__state__()


    def step(self, action):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        
        theta_r, theta_s = action
    

        
        prev_dist = distance(self.sailboat.pos(), self.target)
        
        self.sailboat.step(theta_r, theta_s, (self.wind, self.wind_dir))
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
        
        return np.array([self.sailboat.x/screen_width, self.sailboat.y/screen_height,
                        self.target[0]/screen_width, self.target[1]/screen_height])
        
        

    
    def draw(self):
        if self.surf is None:
            pygame.init()
            self.surf = pygame.display.set_mode((screen_width, screen_height))
        self.surf.fill((6,66,115))
        pygame.draw.circle(self.surf, (194, 178, 128), self.target, 30)
        pygame.draw.circle(self.surf, (150, 90, 62), (self.target[0] + 10, self.target[1] - 5) , 5)
        self.sailboat.draw(self.surf)
        pygame.display.flip()
        

        screen_center_x = self.surf.get_width() * 0.8
        screen_center_y = self.surf.get_height() * 0.2
        
        
        # Arrow dimensions
        arrow_length = self.wind * 4
        arrow_tip_x = screen_center_x + arrow_length * math.cos(self.wind_dir)
        arrow_tip_y = screen_center_y + arrow_length * math.sin(self.wind_dir)
        arrow_base_x = screen_center_x - arrow_length * 0.5 * math.cos(self.wind_dir)
        arrow_base_y = screen_center_y - arrow_length * 0.5 * math.sin(self.wind_dir)

        # Draw the arrow line
        pygame.draw.line(self.surf, (255, 0, 0), (arrow_base_x, arrow_base_y), (arrow_tip_x, arrow_tip_y), 3)

        # Arrowhead points
        arrowhead_length = self.wind
        arrowhead_angle = math.pi / 6  # 30 degrees
        left_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(self.wind_dir - arrowhead_angle)
        left_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(self.wind_dir - arrowhead_angle)
        right_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(self.wind_dir + arrowhead_angle)
        right_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(self.wind_dir + arrowhead_angle)

         # Draw the arrowhead
        pygame.draw.polygon(self.surf, (255, 0, 0), [(arrow_tip_x, arrow_tip_y),
                                                (left_arrowhead_x, left_arrowhead_y),
                                                (right_arrowhead_x, right_arrowhead_y)])

        # Display wind speed next to the arrow
        font = pygame.font.Font(None, 24)
        wind_speed_text = font.render(f"{self.wind} m/s", True, (255, 255, 255))
        text_offset_x = 20  # Offset to position text near the arrow
        text_offset_y = -10
        self.surf.blit(wind_speed_text, (arrow_tip_x + text_offset_x, arrow_tip_y + text_offset_y))
        pygame.display.flip()




    
        
        


if __name__ == "__main__":
    env = SailboatEnv(False)
    

    while True:
        state = env.reset()
        done = False
        while not done:
            
            delta_theta_r = 0
            
            
            state, r, done, _ = env.step((delta_theta_r, np.random.uniform()))
            print(r, state)
            env.draw()
            time.sleep(1/60)