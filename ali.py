import gymnasium as gym
import flappy_bird_gymnasium

# Erstellen der Flappy Bird-Umgebung
env = gym.make("FlappyBird-v0", render_mode="human")

# Zurücksetzen der Umgebung
obs, _ = env.reset()

# Definieren einer einfachen Heuristik
def heuristic(obs):
    # Extrahiere die relevante Information aus der Beobachtung
    bird_y = obs[0]
    # Definiere eine Schwellenhöhe
    threshold = 1
    # Wenn der Vogel unterhalb der Schwelle ist, flap (Aktion 1), sonst keine Aktion (Aktion 0)
    if bird_y < threshold:
        return 1
    else:
        return 0

# Hauptschleife für das Spiel
while True:dsffsfvds
    # Aktion basierend auf Heuristik auswählen
    action = heuristic(obs)
    
    # Ausführen der Aktion und Empfangen der neuen Beobachtung
    obs, reward, terminated, _, info = env.step(action)
    
    # Überprüfen, ob das Spiel beendet ist
    if terminated:
        break

# Schließen der Umgebung
env.close()
