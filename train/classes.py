from stable_baselines3.common.callbacks import BaseCallback
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
import matplotlib.pyplot as plt

class CustomFlappyBirdEnv(FlappyBirdEnv):
    def __init__(self, render_mode=None, use_lidar=False, **kwargs):
        super(CustomFlappyBirdEnv, self).__init__(render_mode=render_mode, use_lidar=use_lidar, **kwargs)
        self.pipe_gap = 150  # Beispiel: Änderung des Abstands zwischen den Rohren
        self.gravity = 0.5  # Beispiel: Änderung der Schwerkraft
        self.pipes_passed = 0

    def _get_reward(self):
        reward = 0.4  # +0.1 - every frame it stays alive
        if self.player['y'] + self.player['h'] >= self.screen_height:
            reward = -20.0  # -1.0 - dying
        elif self.player['y'] <= 0:
            reward = -10  # -0.5 - touch the top of the screen
        elif self.pipe_passed:
            reward = 100.0  # +1.0 - successfully passing a pipe
            self.pipes_passed += 1
        return reward

    def reset(self, **kwargs):
        self.pipes_passed = 0
        return super().reset(**kwargs)

class PipeCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PipeCounterCallback, self).__init__(verbose)
        self.pipes_passed_per_iteration = []

    def _on_step(self) -> bool:
        # Anzahl der durchflogenen Pipes pro Step hinzufügen
        pipes_passed = self.training_env.get_attr('pipes_passed', 0)
        if isinstance(pipes_passed, list):
            pipes_passed = pipes_passed[0]
        self.pipes_passed_per_iteration.append(pipes_passed)
        return True

    def _on_training_end(self) -> None:
        # Ausgabe der Ergebnisse nach dem Training
        plt.plot(self.pipes_passed_per_iteration)
        plt.xlabel('Iteration')
        plt.ylabel('Pipes Passed')
        plt.title('Pipes Passed Per Iteration')
        plt.savefig('pipes_passed_per_iteration.png')  # Speichern des Diagramms in einer Datei
        plt.close()