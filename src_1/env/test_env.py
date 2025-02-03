import math
import random
import time

import numpy as np

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


scale_factor = distance((0, 0), (screen_width, screen_height))


class Car(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.points = [(8, 5), (8, -5), (-8, -5), (-8, 5), (8, 5)]
        
        
        
        self.sail_angle = 0
        self.rudder_angle = 180
        self.length = 20
        self.width = 50

        # Sail geometry: define the four corners of the rectangular sail
        self.sail_height = self.length * 1.1  # Sail height proportional to boat length
        self.sail_width = self.width * 0.1    # Sail width proportional to boat width

        # Sail geometry: define the four corners of the rectangular sail
        self.rudder_height = self.length * 1  # Sail height proportional to boat length
        self.rudder_width = self.width * 0.2    # Sail width proportional to boat width

    def step(self, throttle, steer):
        throttle = max(-1, min(1, throttle)) * 0.2
        steer = max(-1, min(1, steer)) * 5e-1

        self.speed += throttle
        # Clip speed
        self.speed = max(-2, min(10, self.speed))
        self.angle = (self.angle + steer) % (2 * math.pi)
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)

  
        
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
            rotate(x, y, self.sail_angle + self.angle) for x, y in sail_points
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
            rotate(x, y, self.rudder_angle + self.angle) for x, y in rudder_points
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


class CarEnv(object):
    def __init__(self, use_easy_state):
        self.car = None
        self.target = None
        self.steps = 0
        self.surf = None
        self.done = True
        self.use_easy_state = use_easy_state
        
        

    def reset(self):
        self.car = Car(random.randint(0, screen_width - 1), random.randint(0, screen_height - 1))
        #self.target = random.randint(0, screen_width - 1), random.randint(0, screen_height - 1)
        self.target = (screen_width - 1)/2, (screen_width - 1)/2
        self.done = False
        self.steps = 0
        
        self.wind_dir = np.pi + 0.1*np.random.uniform(-np.pi,np.pi)
        self.wind_speed = 10 + np.random.randint(-10,10)
        
        
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
        arrow_tip_x = arrow_x + arrow_length * math.cos(self.wind_dir)
        arrow_tip_y = arrow_y + arrow_length * math.sin(self.wind_dir)
        arrow_base_x = arrow_x
        arrow_base_y = arrow_y

        # Draw the arrow line
        pygame.draw.line(self.surf, (255, 0, 0), (arrow_base_x, arrow_base_y), (arrow_tip_x, arrow_tip_y), 3)

        # Arrowhead points
        arrowhead_length = 10
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
        self.car.draw(self.surf)
        self.draw_wind_arrow()
        pygame.display.flip()


    def step(self, action):
        self.steps += 1
        if self.done:
            raise RuntimeWarning("Calling step on environment that is currently in the 'done' state!")
        thrust, steer = action
        prev_dist = distance(self.car.pos(), self.target)
        self.car.step(thrust, steer)
        dist = distance(self.car.pos(), self.target)
        
        r = (prev_dist - dist)/scale_factor - 0.001

        if not is_in_bounds(self.car.x, self.car.y) or self.steps > 1000:
            r -= 1
            self.done = True

        if dist < 30:
            r += 1
            self.done = True
        return self.__state__(), r, self.done, None

    def __state__(self):
        if not self.use_easy_state:
            return np.array([self.car.x/screen_width, self.car.y/screen_height, self.car.angle/(2*math.pi),
                             self.car.speed*0.1, self.target[0]/screen_width, self.target[1]/screen_height])
        else:
            angle_to_target = math.atan2(self.car.y - self.target[1], self.car.x - self.target[0])
            distance_to_target = distance(self.car.pos(), self.target)
            distance_to_target /= scale_factor
            angle_error = (self.car.angle - angle_to_target)%(2*math.pi)
            angle_error -= math.pi
            angle_error /= math.pi

            return np.array([self.car.x / screen_width, self.car.y / screen_height, angle_error,
                             distance_to_target, self.car.angle/(2*math.pi), self.car.speed * 0.1])



if __name__ == "__main__":
    env = CarEnv(False)
    target_speed = 8

    while True:
        state = env.reset()
        done = False
        while not done:
            speed = state[3]
            throttle = target_speed - speed
            state, r, done, _ = env.step((throttle, 2 * random.random() - 1))
            print(r, speed)
            env.draw()
            time.sleep(1/60)