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
        self.passed_pipes = 0  # Zählen der durchflogenen Säulen

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if self.pipe_passed:
            self.passed_pipes += 1  # Erhöhen des Zählers bei erfolgreichem Durchfliegen einer Säule
        info['passed_pipes'] = self.passed_pipes
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.passed_pipes = 0  # Zurücksetzen des Zählers beim Zurücksetzen der Umgebung
        return super().reset(**kwargs)

# Registrieren Sie die benutzerdefinierte Umgebung
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=1000,
)

# Environment erstellen
env = make_vec_env("CustomFlappyBird-v0", n_envs=1, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

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
    verbose=1
)

# Listen zum Speichern der Belohnungen und der Schritte
reward_list = []
timesteps_list = []
pipes_list = []

# Callback-Funktion zur Überwachung des Trainingsfortschritts
def callback(_locals, _globals):
    global reward_list, timesteps_list, pipes_list

    # Überwachen der Belohnungen, Schritte und der durchflogenen Säulen
    if len(_locals['self'].ep_info_buffer) > 0:
        reward_list.append(_locals['self'].ep_info_buffer[0]['r'])
        timesteps_list.append(_locals['self'].num_timesteps)
        pipes_list.append(_locals['self'].ep_info_buffer[0]['passed_pipes'])
    
    # Plot und speichern der Grafik alle 100000 Schritte
    if _locals['self'].num_timesteps % 1000 == 0:
        plot_and_save_progress()

    return True

def plot_and_save_progress():
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps_list, reward_list, label='Rewards')
    plt.plot(timesteps_list, pipes_list, label='Passed Pipes')
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()

    # Erstellen Sie das Verzeichnis, falls es nicht existiert
    os.makedirs('img', exist_ok=True)

    # Save the plot as a PNG file
    plt.savefig('img/training_progress.png')

    # Save the plot as a JPG file (if needed)
    plt.savefig('img/training_progress.jpg')

    plt.close()

# Modell trainieren mit Callback
model.learn(total_timesteps=1000, callback=callback)

# Modell speichern
model.save("models/ppo_flappybird_custom_v2")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Metriken während des Tests sammeln
total_pipes = 0
total_episodes = 10

for _ in range(total_episodes):
    obs = env.reset()
    episode_pipes = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_pipes = info[0]['passed_pipes']  # Da n_envs=1
        if dones:
            break
    total_pipes += episode_pipes

mean_pipes = total_pipes / total_episodes
print(f"Mean pipes passed per episode: {mean_pipes}")

# Plotting the final training progress
plot_and_save_progress()

env.close()
