import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import matplotlib.pyplot as plt
import numpy as np
import os

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=False, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)
        self.pipe_gap = 150  # Beispiel: Änderung des Abstands zwischen den Rohren
        self.gravity = 0.5  # Beispiel: Änderung der Schwerkraft

    def _get_reward(self):
        reward = 0.4  # +0.1 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -20.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -10  # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 100.0  # +1.0 - successfully passing a pipe
        return reward

# Registrieren Sie die benutzerdefinierte Umgebung
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=100000,
)

# Environment erstellen
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

# PPO Modell definieren
# Modell laden
model = PPO.load("models/ppo_flappybird_custom_v2", env=env)

# Listen zum Speichern der Belohnungen und der Schritte
reward_list = []
timesteps_list = []

# Callback-Funktion zur Überwachung des Trainingsfortschritts
def callback(_locals, _globals):
    global reward_list, timesteps_list

    # Überwachen der Belohnungen und der Schritte
    if len(_locals['self'].ep_info_buffer) > 0:
        reward_list.append(_locals['self'].ep_info_buffer[0]['r'])
        timesteps_list.append(_locals['self'].num_timesteps)
    
    # Plot und speichern der Grafik alle 100000 Schritte
    if _locals['self'].num_timesteps % 100000 == 0:
        plot_and_save_progress()

    return True

def plot_and_save_progress():
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps_list, reward_list)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Training Progress')

    # Erstellen Sie das Verzeichnis, falls es nicht existiert
    os.makedirs('img', exist_ok=True)

    # Save the plot as a PNG file
    plt.savefig('img/training_progress.png')

    # Save the plot as a JPG file (if needed)
    plt.savefig('img/training_progress.jpg')

    plt.close()

# Modell trainieren mit Callback
model.learn(total_timesteps=1000000, callback=callback)

# Modell speichern
model.save("models/ppo_flappybird_custom_v2")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Plotting the final training progress
plot_and_save_progress()

# Teste den trainierten Agenten
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()
