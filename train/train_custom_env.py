import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np
import pandas as pd
from classes import CustomFlappyBirdEnv, SaveEpisodeRewardCallback

# Log-Directory für den Callback festlegen
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=10000000,
)
# Environment erstellen und mit VecMonitor wrappen
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

# PPO Modell definieren
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=3e-4, 
    n_steps=256, 
    batch_size=64, 
    n_epochs=10, 
    gamma=0.99, 
    ent_coef=0.01,
    verbose=1,
    device='cuda'
)

# Callback erstellen
callback = SaveEpisodeRewardCallback(log_dir=log_dir)

# Modell trainieren mit Callback
model.learn(total_timesteps=300000, callback=callback)

# Modell speichern
# model.save("models/model_10Mio")

