from stable_baselines3.common.callbacks import BaseCallback
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=False, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)
        self.pipe_gap = 150  # Beispiel: Änderung des Abstands zwischen den Rohren
        self.gravity = 0.5  # Beispiel: Änderung der Schwerkraft

    def _get_reward(self):
        reward = 0.1  # +0.1 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -1.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -0.5 # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 1.0  # +1.0 - successfully passing a pipe
        return reward

class SaveEpisodeRewardCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose=1):
        super(SaveEpisodeRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        self.current_rewards.append(rewards)

        # Check if any environment is done
        dones = self.locals['dones']
        if any(dones):
            # Calculate the total reward for this episode
            episode_reward = sum(np.sum(reward) for reward in self.current_rewards)
            self.episode_rewards.append({
                'timesteps': self.num_timesteps,
                'reward': episode_reward
            })
            self.current_rewards = []  # Reset for the next episode

        return True

    def on_training_end(self) -> None:
        df = pd.DataFrame(self.episode_rewards)
        df.to_csv(os.path.join(self.log_dir, 'episode_rewards.csv'), index=False)