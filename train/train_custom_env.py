import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecMonitor
import numpy as np
from classes import CustomFlappyBirdEnv, RewardLoggerCallback

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

# Callback instanziieren
reward_logging_callback = RewardLoggerCallback()

# Modell trainieren mit Callback
model.learn(total_timesteps=10000, callback=reward_logging_callback)

# Modell speichern
# model.save("models/model_10Mio")

