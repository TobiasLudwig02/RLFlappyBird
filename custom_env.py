import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=False, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)
        # Hier können Sie die gewünschten Hyperparameter einstellen
        self.pipe_gap = 150  # Beispiel: Änderung des Abstands zwischen den Rohren
        self.gravity = 0.5  # Beispiel: Änderung der Schwerkraft
        # Weitere Anpassungen nach Bedarf

    def _get_reward(self):
        reward = 0.2  # +0.1 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -10.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -5  # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 10.0  # +1.0 - successfully passing a pipe
        return reward

# Registrieren Sie die benutzerdefinierte Umgebung
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=1000,
)

# Environment erstellen
env = make_vec_env("CustomFlappyBird-v0", n_envs=4, env_kwargs={'render_mode': 'rgb_array', 'use_lidar': False})

# PPO Modell definieren
model = PPO("MlpPolicy", env, verbose=1)

# Modell trainieren
model.learn(total_timesteps=1000000)

# Modell speichern
model.save("ppo_flappybird_custom")

# Modell laden (falls erforderlich)
# model = PPO.load("ppo_flappybird_custom")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Teste den trainierten Agenten
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break

env.close()
