import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import pandas as pd
import os
from classes import CustomFlappyBirdEnv

gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=10000000,
)

# Environment setup
env = gym.make("CustomFlappyBird-v0", render_mode="rgb_array", use_lidar=False)
# Heuristic function
def heuristic(obs):
    bird_y = obs[9]
    threshold = 0.4
    return 1 if bird_y > threshold else 0

# Variables for tracking progress
total_timesteps = 2000000
current_total_timesteps = 0
ep_rew_sum = 0
episode_rewards = []
episode_lengths = []

# Reset the environment
obs, _ = env.reset()

while current_total_timesteps < total_timesteps:
    action = heuristic(obs)
    obs, reward, terminated, _, _ = env.step(action)
    ep_rew_sum += reward
    current_total_timesteps += 1

    if terminated:
        episode_rewards.append(ep_rew_sum)
        episode_lengths.append(current_total_timesteps)
        obs, _ = env.reset()
        ep_rew_sum = 0

        # Log progress every 100 episodes
        if len(episode_rewards) % 100 == 0:
            mean_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {len(episode_rewards)}, Total Timesteps: {current_total_timesteps}, Mean Reward (last 100 episodes): {mean_reward}")

# Close the environment
env.close()

# Calculate cumulative timesteps and mean rewards
cumulative_timesteps = np.cumsum(episode_lengths)
mean_rewards = [np.mean(episode_rewards[:i+1]) for i in range(len(episode_rewards))]

# Prepare the results in a DataFrame
results_df = pd.DataFrame({
    'total_timesteps': cumulative_timesteps,
    'ep_rew_mean': mean_rewards
})

# Log the results to a file
log_dir = "./logs/baseline_rew100_2Mio"
os.makedirs(log_dir, exist_ok=True)
baseline_log_file = os.path.join(log_dir, "progress.csv")
results_df.to_csv(baseline_log_file, index=False)

print(f"Results saved to {baseline_log_file}")