import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import os
from classes import CustomFlappyBirdEnv_rew100

# Log-dict and folder name for callback 
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
custom_log_file = os.path.join(log_dir, "A2C_MLP_rew100_2Mio")

# Configure the logger to save data to a specific folder
new_logger = configure(custom_log_file, ["stdout", "csv"])

# Register env
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv_rew100',
    max_episode_steps=10000000,
)

# Create env
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

# Define A2C model
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_rms_prop=True,
    verbose=1,
    device='cuda'
)

# Attach the new logger to the model
model.set_logger(new_logger)

# Train model
model.learn(total_timesteps=2000000)

# Save model
model.save("models/A2C_MLP_rew100_2Mio")

