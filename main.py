import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
# Definieren einer einfachen Heuristik
def heuristic(obs):
    # Extrahiere die relevante Information aus der Beobachtung
    bird_y = obs[9]
    # Definiere eine Schwellenhöhe
    threshold = 0.4
    
    # Wenn der Vogel unterhalb der Schwelle ist, flap (Aktion 1), sonst keine Aktion (Aktion 0)
    if bird_y > threshold:
        return 1
    else:
        return 0

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = heuristic(obs)

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    print(obs[9])
    # Checking if the player is still alive
    if terminated:
        break

env.close()