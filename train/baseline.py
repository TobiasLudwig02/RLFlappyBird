import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import pandas as pd
import os
from classes import CustomFlappyBirdEnv_std

# Register env
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv_std',
    max_episode_steps=10000000,
)

# Environment setup
env = gym.make("CustomFlappyBird-v0", render_mode="human", use_lidar=False)

# Heuristic function
def heuristic(obs):
    bird_y = obs[9]
    threshold = 0.4
    return 1 if bird_y > threshold else 0

# Reset the environment
obs, _ = env.reset()

while 1:
    action = heuristic(obs)
    obs, reward, terminated, _, _ = env.step(action)
    if terminated:
        obs, _ = env.reset()
        ep_rew_sum = 0

# Close the environment
env.close()

