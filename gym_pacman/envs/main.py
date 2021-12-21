import warnings

from numpy.lib.npyio import save
from pacmanenv import PacmanEnv
import numpy as np
from mss import mss
import cv2
from numpy.core.fromnumeric import shape
import numpy as np
from pygame.constants import HIDDEN

from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

import os
import pygame
from pygame.locals import *

import copy 
import time

UP = 1
DOWN = -1 
LEFT = 2
RIGHT = -2
test = []
test1 = [0,1,2,3]
def input_key(obs):
    #time.sleep(0.5)
    #return  np.random.choice(test1)
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
            if event.type == KEYDOWN:
                key_pressed = pygame.key.get_pressed()
                if key_pressed[K_UP]:
                    return 0#UP
                if key_pressed[K_DOWN]:
                    return 1#DOWN
                if key_pressed[K_LEFT]:
                    return 2#LEFT
                if key_pressed[K_RIGHT]:
                    return 3#R  IGHT 


def generate_expert_traj(model, save_path=None, env=None, n_timesteps=0,
                         n_episodes=100, image_folder='recorded_images'):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to record images.
    :return: (dict) the generated expert trajectories.
    """
    # Sanity check
    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8
    if record_images and save_path is None:
        warnings.warn("Observations are images but no save path was specified, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    image_ext = 'jpg'
    if record_images:
        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None
    all_episode_rewards = []
    all_episodes_number_steps = []
    all_episode_scores = []
    games_won = []
    episode_score = 0
    number_of_steps = 0
    episode_reward_sum = 0
    while ep_idx < n_episodes:
        obs_ = obs
        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx, image_ext))
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            if obs_.shape[-1] == 3:
                obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, obs_)
            image_path_save = os.path.join('/content/recorded_images',"{}.{}".format(idx, image_ext))
            observations.append(image_path_save)
        else:
            observations.append(obs_)

        action = model(obs)

        obs, reward, done, _ = env.step(copy.copy(action))
        if env.game.score > 0:
            episode_score = env.game.score
        
        number_of_steps = number_of_steps + 1

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        episode_reward_sum += reward
        idx += 1
        if done:
            print(ep_idx,reward_sum)
            episode_returns[ep_idx] = reward_sum
            ep_idx += 1
            all_episode_rewards.append(episode_reward_sum)
            all_episode_scores.append(episode_score)
            all_episodes_number_steps.append(number_of_steps)
            games_won.append(env.game.events_AI)
            print("mean reward",np.mean(all_episode_rewards),"std reward",np.std(all_episode_rewards),"mean step",np.mean(all_episodes_number_steps),"mean score",np.mean(all_episode_scores),"std score",np.std(all_episode_scores),"games won",  games_won.count(6) )
            number_of_steps = 0
            episode_reward_sum = 0
            

    observations = np.array(observations)

    actions = np.array(actions).reshape((-1, 1))
    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])
 
    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict 


if __name__ == "__main__":
    env = PacmanEnv()
    i = 0
    bounding_box = {'top': 170 , 'left': 100, 'width': 448, 'height': 576-16*4}
    TOP = 170
    LEFT = 100
    WIDTH = 448
    HEIGHT = 576-16*4
    sct = mss()
    counter = 1 
    path = "/home/lpe/Desktop/Project_in_Artificial_Intelligence_PAC-MAN/pacman_game_env/gym_pacman/envs"
    generate_expert_traj(input_key,save_path=path,env=env,n_timesteps=0,n_episodes=50,image_folder='recorded_images2')


        
                                                        


