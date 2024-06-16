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

gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=10000,
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

# Callback-Funktion zur Überwachung des Trainingsfortschritts
def callback(_locals, _globals):
    global reward_list, timesteps_list

    # Überwachen der Belohnungen und der Schritte
    if len(_locals['self'].ep_info_buffer) > 0:
        reward_list.append(_locals['self'].ep_info_buffer[0]['r'])
        timesteps_list.append(_locals['self'].num_timesteps)
    
    return True

# Modell trainieren mit Callback
model.learn(total_timesteps=10000, callback=callback)

# Modell speichern
model.save("models/ppo_flappybird_custom_v2")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Metriken während des Tests sammeln
def test_model(env, model, num_episodes=10):
    episode_rewards = []
    pipes_passed = []
    lifespans = []
    collisions = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        pipes = 0
        lifespan = 0
        collision = 0
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            total_reward += rewards[0]
            lifespan += 1
            if 'pipe_passed' in info[0] and info[0]['pipe_passed']:
                pipes += 1
            if rewards[0] < 0:  # Annahme: negative Belohnung bedeutet Kollision
                collision += 1
            if dones[0]:
                done = True

        episode_rewards.append(total_reward)
        pipes_passed.append(pipes)
        lifespans.append(lifespan)
        collisions.append(collision)

    return episode_rewards, pipes_passed, lifespans, collisions

# Modell testen
episode_rewards, pipes_passed, lifespans, collisions = test_model(env, model)

print(f"Durchschnittliche Belohnung pro Episode: {np.mean(episode_rewards)}")
print(f"Durchschnittliche Anzahl durchflogener Säulen pro Episode: {np.mean(pipes_passed)}")
print(f"Durchschnittliche Lebensdauer pro Episode: {np.mean(lifespans)}")
print(f"Maximale Lebensdauer pro Episode: {np.max(lifespans)}")
print(f"Mindestlebensdauer pro Episode: {np.min(lifespans)}")
print(f"Varianz der Belohnungen pro Episode: {np.var(episode_rewards)}")
print(f"Durchschnittliche Anzahl von Kollisionen pro Episode: {np.mean(collisions)}")

# Ausgabe der detaillierten Ergebnisse
for i, (reward, pipes, lifespan, collision) in enumerate(zip(episode_rewards, pipes_passed, lifespans, collisions)):
    print(f"Episode {i+1}: Belohnung = {reward}, Durchflogene Säulen = {pipes}, Lebensdauer = {lifespan}, Kollisionen = {collision}")

# Graphen erstellen und speichern
def plot_metrics(timesteps, rewards, pipes, lifespans, collisions):
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(timesteps, rewards, label='Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(timesteps, pipes, label='Average Pipes Passed per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Pipes Passed')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(timesteps, lifespans, label='Average Lifespan per Episode', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Lifespan')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Erstellen Sie das Verzeichnis, falls es nicht existiert
    os.makedirs('img', exist_ok=True)

    # Save the plot as a PNG file
    plt.savefig('img/testing_metrics.png')

    # Save the plot as a JPG file (if needed)
    plt.savefig('img/testing_metrics.jpg')

    plt.close()

    # Kollisionen plotten
    plt.figure(figsize=(7, 5))
    plt.plot(timesteps, collisions, label='Average Collisions per Episode', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Collisions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig('img/collisions_metrics.png')

    # Save the plot as a JPG file (if needed)
    plt.savefig('img/collisions_metrics.jpg')
    
    plt.close()

# Episodenindizes erstellen
episodes = list(range(1, len(episode_rewards) + 1))

# Graphen plotten und speichern
plot_metrics(episodes, episode_rewards, pipes_passed, lifespans, collisions)

env.close()
