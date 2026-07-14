# Workshop 04 · Reinforcement Learning (Q-Learning, SARSA & DQN)

In diesem Workshop bringst du einem Agenten bei, sich selbstständig durch eine kleine GridWorld zu navigieren – nur über Belohnung und Bestrafung, ganz ohne Labels. Du implementierst tabellarisches Q-Learning und SARSA und wagst als Bonus den Sprung zu Deep RL mit einem Deep-Q-Network (DQN) auf CartPole.

## Lernziele
- Den Unterschied zwischen überwachtem Lernen (Labels) und Reinforcement Learning (Reward-Signal) verstehen.
- Die zentralen RL-Begriffe kennen: State, Action, Reward, Policy, Episode, Discount-Faktor.
- Den Trade-off Exploration vs. Exploitation über epsilon-greedy nachvollziehen und selbst umsetzen.
- Das TD-Update für Q-Learning (off-policy) und SARSA (on-policy) implementieren und vergleichen.
- Den Einfluss der Hyperparameter (α, γ, ε-Decay) auf das Lernverhalten beobachten.
- Den Übergang von tabellarischem RL zu Deep RL (DQN) verstehen.

## Struktur
| Pfad | Inhalt |
|------|--------|
| [`workshop.md`](workshop.md) | Ausführliche Anleitung & Erklärung |
| [`aufgabe/`](aufgabe/) | Starter-Code mit TODO-Lücken |
| [`loesung/`](loesung/) | Musterlösung |
| [`requirements.txt`](requirements.txt) | Abhängigkeiten |

## Setup
```bash
pip install -r requirements.txt
```

## Ausführen
Führe den Code im Ordner `aufgabe/` aus (sobald die TODOs implementiert sind) oder in `loesung/` für die fertige Musterlösung:

```bash
# Standard-Training mit Q-Learning
python aufgabe/workshop_reinforcement_learning.py --train

# Mit SARSA statt Q-Learning
python aufgabe/workshop_reinforcement_learning.py --train --algo sarsa

# Training + finale Bewertung + Policy als ASCII anzeigen
python aufgabe/workshop_reinforcement_learning.py --train --evaluate --render

# Mehr Episoden
python aufgabe/workshop_reinforcement_learning.py --train --episodes 1000

# Bonus: Deep-Q-Network auf CartPole (benötigt gymnasium + tensorflow)
python aufgabe/workshop_reinforcement_learning.py --dqn

# Musterlösung
python loesung/workshop_reinforcement_learning.py --train --evaluate --render
```

[← Zur Workshop-Übersicht](../../README.md)
