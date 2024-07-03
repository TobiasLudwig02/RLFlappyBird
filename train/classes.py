from stable_baselines3.common.callbacks import BaseCallback
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import matplotlib.pyplot as plt
import numpy as np

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

# Custom Callback to log ep_rew_mean
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.ep_rew_mean = []

    def _on_step(self) -> bool:
        # Use the buffer of episode info to gather rewards
        if len(self.model.ep_info_buffer) > 0:
            # Calculate mean reward for this episode
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            self.ep_rew_mean.append(mean_reward)
        return True

    def plot_rewards(self):
        plt.scatter(range(len(self.ep_rew_mean)), self.ep_rew_mean)
        plt.xlabel('Iteration')
        plt.ylabel('Episode Reward Mean')
        plt.title('Episode Reward Mean per Iteration')
        plt.savefig("test.png")
        plt.show()