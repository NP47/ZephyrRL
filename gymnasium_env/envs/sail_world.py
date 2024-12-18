
import math
from typing import Optional, Union

import numpy as np

from ..utils import *
import gymnasium as gym
from gymnasium import spaces





def distance(pos_1, pos_2):
    x_1, y_1 = pos_1
    x_2, y_2 = pos_2
    return ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5


class SailBoatEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
   
        #sailboat
        self.mass_b = 100
        self.S = 5
        
        
        
        
        #wind
        self.true_wind_magnitude = 4 
        self.theta_true_wind = np.pi 
        
        #enviroment 
        self.rho_a = 1.225

        
        #minima and maxima of actiona an observation spaces
        self.max_action = np.pi/4   #self.dtheta_b = 0.01 #discrete arrproach turining rate 
        self.min_action = -np.pi/4
        
        
        self.max_x_pos = 100.0
        self.max_y_pos = 100.0
        self.max_x_vel = np.inf
        self.max_y_vel = np.inf
        
      
       
        self.h = 0.1  # seconds between state updates (so the step size)
        self.dynamics_integrator = "rkf45"


        #some parameters grouped together
        self.fixed_param = np.array([self.mass_b, self.S, self.rho_a])

        #variables which might be changed later on to be in the state and dynamic (due to seed dependance of noise)
        self.terminal_state = np.array([0,0])
        self.epsilon = 10.0

        self.t_max = 40

        #bounds of the programm, sould be inplemented in my opnion, makes everything more efficient
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        
        
        self.low_state = np.array(
            [-self.max_x_pos, -self.max_y_pos, -self.max_x_vel, -self.max_y_vel], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x_pos, self.max_y_pos, self.max_x_vel, self.max_y_vel], dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)
        #self.action_space = spaces.Box(
        #    low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        #)
        
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.state: np.ndarray | None = None

    def is_in_bounds(self, x, y):
        return -self.max_x_pos <= x <= self.max_x_pos and -self.max_y_pos <= y <= self.max_y_pos

    def step(self, action):
        
        in_bound = True
           
        z = self.state #so the state is the particular ode
        
        self.theta_b += (1-action)*0.05
        #self.theta_b = action[0]

        x_prev, y_prev, _, _  = z 
        dist_to_terminal_prev = distance((x_prev, y_prev), self.terminal_state)
        
        #compute the new step
        z_next = rkf45_step(system_ode, self.t, z, self.h, args = (self.theta_b, self.theta_true_wind, self.true_wind_magnitude, self.fixed_param))
        self.t = self.t + self.h
        
        
        x, y, v_x, v_y = z_next
        dist_to_terminal = distance((x, y), self.terminal_state)
        #build the current state at the step
        self.state = np.array([x, y, v_x, v_y]) #hmm maybe reduce observation space to only the postion?

        #conditions for termination
        terminated = terminal_state_check([x,y], self.terminal_state, self.epsilon)
        truncated = bool(self.t >= self.t_max) or not self.is_in_bounds(x, y)

        #rewards
        scale_factor = 10 #distance((0, 0), (2*self.max_x_pos, 2*self.max_y_pos))

        reward = (dist_to_terminal_prev - dist_to_terminal)/scale_factor - 0.001

        #rewards based on termination
        if terminated:
            reward += 100.0
        
        if truncated:
            reward -= 1
            
        if not self.is_in_bounds(x, y):
            reward -= 1
            
            
    
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        super().reset(seed=seed) #we get a new seed
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        
        self.t = 0
        # Boat initial conditions/parameters, these will be initialized as the state should do this randomly

        self.theta_b = np.random.uniform(0, 2*np.pi)
        x_0, y_0, v_x_0, v_y_0 = np.random.uniform(-66, 66), np.random.uniform(-66, 66), np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)
        
        #here also define new wind values and initial conditions? but id doesnt just have ot be that, we can apply  more things 
        self.state = np.array([x_0, y_0, v_x_0, v_y_0]) #define the initial state
        
    
        return np.array(self.state, dtype=np.float32), {}
    
    