import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Environment erstellen
env = gym.make("FlappyBird-v0", render_mode='human', use_lidar=False)

# Modell laden
model = PPO.load("models\ppo_flappybird")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Teste den trainierten Agenten
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, info = env.step(action)
    env.render()
    if dones:
        obs, _ = env.reset()

env.close()
