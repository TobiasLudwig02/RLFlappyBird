import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import os
import sys
# Fügen Sie den Pfad zum 'train'-Verzeichnis hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

# Importieren Sie die Klasse
from classes import CustomFlappyBirdEnv_std

# Registrieren Sie die benutzerdefinierte Umgebung
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv_std',
    max_episode_steps=100000,
)

# Environment erstellen
env = gym.make("CustomFlappyBird-v0", render_mode='human', use_lidar=False)

# Modell laden
# model = PPO.load("models/A2C_MLP_rew100_2Mio")
# model = PPO.load("models/DQN_MLP_rew100_2Mio")
# model = PPO.load("models/PPO_MLP_rew100_2Mio")
model = PPO.load("models/PPO_MLP_std_2Mio")


# Teste den trainierten Agenten
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, info = env.step(action)
    env.render()
    if dones:
        obs, _ = env.reset()

