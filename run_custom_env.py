import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=False, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)

    def _get_reward(self):
        reward = 0.1  # +0.1 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -1.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -0.5  # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 1.0  # +1.0 - successfully passing a pipe
        return reward

# Registrieren Sie die benutzerdefinierte Umgebung
gym.envs.registration.register(
    id='CustomFlappyBird-v0',
    entry_point='__main__:CustomFlappyBirdEnv',
    max_episode_steps=100000,
)

# Environment erstellen
env = gym.make("CustomFlappyBird-v0", render_mode='human', use_lidar=False)

# Modell laden
model = PPO.load("ppo_flappybird_custom_v2")

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
