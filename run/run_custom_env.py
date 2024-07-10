import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys

# Add classes.py to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

# Import classes
from classes import CustomFlappyBirdEnv_std

# Register env
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv_std',
    max_episode_steps=100000,
)

# Create env
env = gym.make("CustomFlappyBird-v0", render_mode='human', use_lidar=False)

# Load model
# model = PPO.load("models/A2C_MLP_rew100_2Mio")
# model = PPO.load("models/DQN_MLP_rew100_2Mio")
# model = PPO.load("models/PPO_MLP_rew100_2Mio")
model = PPO.load("models/PPO_MLP_std_2Mio")

# Run model
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, info = env.step(action)
    env.render()
    if dones:
        obs, _ = env.reset()

