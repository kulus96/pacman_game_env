
import gym
from gym import Env
from gym import spaces
from numpy.core.fromnumeric import shape
import numpy as np
import cv2
import sys
sys.path.append(r'gym_pacman/envs/Pacman_Game') 
sys.path.append(r'gym_pacman/envs') 
from run import *
from constants import PACMAN
import pygame

#from mss import mss

keyboard_keys = ["up","down","left","right"]

MAX_REWARD = 100000
N_DISCRETE_ACTIONS = 4

BOUNDING_BOX = {'top': 170 , 'left': 100, 'width': 448, 'height': 500}
NUMBER_OF_CHANNELS = 1
REWARD_RANGE = (-100,100)

class PacmanEnv(Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PacmanEnv, self).__init__()

        self.game = GameController()
        self.game.startGame()
        #self.sct = mss()
        self.saved_score = self.game.score
        self.saved_lives = self.game.lives
        self.done = 0

        self.reward_range = REWARD_RANGE

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        shape = (BOUNDING_BOX['height'],BOUNDING_BOX['width'])
        
        self.observation_space = spaces.Box(low = 0, high = 255, shape = shape, dtype = np.uint8)
            

    def _next_observation(self):
        view = pygame.surfarray.array3d(self.game.screen)
        #  convert from (width, height, channel) to (height, width, channel)
        view = view.transpose([1, 0, 2])
        view = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        return view[45:545, :]

    def _take_action(self, action):
        self.game.AI_direction = action        
        self.game.update()
            
    def step(self, action):
        if(self.game.pause.paused):
            self.game.pause.flip()
        # Execute one time step within the environment
        self._take_action(action)

        reward = self._get_reward()

        obs = self._next_observation()

        return obs, reward, self.done, {}

    def reset(self):
        self.game.restartGame()

        return self._next_observation()

    def render(self, mode='human', close=False):
        self.game.rendering = 0

    def _get_reward(self):
        self.done = 0
        if self.game.events_AI == 0: # nothing or wall 
            return -1
        elif self.game.events_AI == 1: # Pellets
            return 5
        elif self.game.events_AI == 2: # super pellets
            return 10
        elif self.game.events_AI == 3: # ghost kill
            return 20
        elif self.game.events_AI == 4: # pacman dead
            return -10
        elif self.game.events_AI == 5: # gameover
            self.done = 1
            return -20
        elif self.game.events_AI == 6: #  won
            self.done = 1
            return 20
        elif self.game.events_AI == 7: #  fruit (not in game)
            return 5
        else:
            return 0
