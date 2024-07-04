from stable_baselines3.common.callbacks import BaseCallback
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
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
            reward = -10.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -0.5 # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 100  # +1.0 - successfully passing a pipe
        return reward
