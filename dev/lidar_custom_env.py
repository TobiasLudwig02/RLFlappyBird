import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=True, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)
        self.pipe_gap = 150  # Example: Change the gap between pipes
        self.gravity = 0.5  # Example: Change gravity
        # Further customizations as needed

    def _get_reward(self):
        reward = 0.4  # +0.4 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -20.0  # -20.0 - dying
        elif self.player['y'] <= 0:
            reward = -10  # -10 - touching the top of the screen
        elif self.pipe_passed:
            reward = 100.0  # +100.0 - successfully passing a pipe
        return reward

# Register the custom environment
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=100000,
)

# Create the environment
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': True})

# Define the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("models/ppo_flappybird_lidar_v2")

# Load the model (if needed)
# model = PPO.load("models/ppo_flappybird_custom")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()
