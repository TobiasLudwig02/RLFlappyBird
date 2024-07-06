import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import os
from classes import CustomFlappyBirdEnv_rew100

# Log-Directory und Dateiname für den Callback festlegen
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
custom_log_file = os.path.join(log_dir, "rew100_2Mio")

# Configure the logger to save data to a specific folder
new_logger = configure(custom_log_file, ["stdout", "csv"])

gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv_rew100',
    max_episode_steps=10000000,
)

# Environment erstellen und mit VecMonitor wrappen
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

# PPO Modell definieren
model = DQN(
    "MlpPolicy",  # Policy type
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    verbose=1,
    device='cuda'
)

# Attach the new logger to the model
model.set_logger(new_logger)

# Modell trainieren mit Callback
model.learn(total_timesteps=2000000)

# Modell speichern
model.save("models/DQN_MLP_rew100_2Mio")

