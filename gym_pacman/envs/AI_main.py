
import gym
import random
from gym_pacman.envs.pacmanenv import PacmanEnv
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

BOUNDING_BOX = {'top': 170 , 'left': 100, 'width': 448, 'height': 576-16*4}

env = PacmanEnv() # gym.make('pacman-v0')
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("trpo_pacman")

del model # remove to demonstrate saving and loading

model = TRPO.load("trpo_pacman")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# def build_model(height, width, channels, actions):
#     model = Sequential()
#     model.add(Convolution2D(32, (8,8), strides=(4,4),padding='same', activation='relu', input_shape=(channels, height, width)))
#     model.add(Convolution2D(64, (4,4), strides=(2,2),padding='same', activation='relu'))
#     model.add(Convolution2D(64, (3,3),padding='same', activation='relu'))
#     model.add(Flatten()) 
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model


# env = PacmanEnv() # gym.make('pacman-v0')
# model = build_model(BOUNDING_BOX['height'],BOUNDING_BOX['width'],1,4)
# model.summary()



# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, 
#                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn


# dqn = build_agent(model, 4)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
# scores = dqn.test(env, nb_episodes=100, visualize=False)
# # episodes = 10
# # for episode in range(1, episodes+1):
# #     state = env.reset()
# #     done = False
# #     score = 0 
    
# #     while not done:
# #         #env.render()
# #         action = random.choice([0,1,2,3])
# #         n_state, reward, done, info = env.step(action)
# #         if done:
# #             print("DONE")
# #         score+=reward
# #     print('Episode:{} Score:{}'.format(episode, score))
# # env.close()
