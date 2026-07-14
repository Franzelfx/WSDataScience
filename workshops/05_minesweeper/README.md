# Workshop 05 · Reinforcement Learning auf Minesweeper (Deep-Q-Network)

Ein RL-Agent lernt per Deep-Q-Network (faltendes Netz) Minesweeper zu spielen; das Ergebnis lässt sich in einem interaktiven Viewer Zug für Zug per „Weiter"-Knopf nachvollziehen.

## Lernziele
- Ein nicht-triviales Spiel als RL-Problem modellieren (State, Action, Reward)
- Verstehen, warum ein CNN für ein 2D-Spielfeld die natürliche Wahl ist
- Action-Masking und Reward Shaping praktisch anwenden
- Einen DQN mit Experience Replay und Target-Netzwerk trainieren
- Das Verhalten des Agenten interaktiv validieren und kritisch bewerten

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
```bash
cd loesung        # oder cd aufgabe, sobald die drei TODOs gefüllt sind

# Trainieren (6x6, 6 Minen) – Modell wird gespeichert
python workshop_minesweeper.py --train

# Trainieren und danach direkt interaktiv ansehen
python workshop_minesweeper.py --train --play

# Gespeichertes Modell laden und Züge per Weiter-Knopf durchschalten
python workshop_minesweeper.py --play

# Mehr Episoden / andere Feldgröße
python workshop_minesweeper.py --train --episodes 3000 --rows 8 --cols 8 --mines 10
```

[← Zur Workshop-Übersicht](../../README.md)
