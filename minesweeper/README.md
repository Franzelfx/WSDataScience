# Workshop: Reinforcement Learning auf Minesweeper (DQN)

## Überblick

Ein **RL-Agent** lernt, **Minesweeper** zu spielen – trainiert mit einem
**Deep-Q-Network (CNN)**. Zur Validierung kannst du dem Agenten **interaktiv**
beim Spielen zusehen und mit einem **„Weiter"-Knopf** Zug für Zug durch eine
Partie schalten.

Die ausführliche Anleitung mit allen Aufgaben steht in **[workshop.md](workshop.md)**.

---

## Lernziele

- Ein Spiel als **RL-Problem** modellieren (State, Action, Reward)
- Verstehen, warum ein **CNN** für ein 2D-Spielfeld die natürliche Wahl ist
- **Action-Masking** und **Reward Shaping** anwenden
- Einen DQN mit **Experience Replay** und **Target-Netzwerk** trainieren
- Agentenverhalten **interaktiv validieren**

---

## Voraussetzungen

- **Python 3.10+**

```bash
pip install -r requirements.txt        # numpy, matplotlib, tensorflow
```

> Der interaktive Viewer (`--play`) benötigt ein **GUI-Backend** (z. B. `MacOSX`/`TkAgg`).
> Training und Plotten funktionieren auch headless.

---

## Dateien

| Datei                                                    | Beschreibung                                |
|----------------------------------------------------------|---------------------------------------------|
| [`workshop.md`](workshop.md)                             | Vollständige Workshop-Anleitung             |
| [`workshop_minesweeper.py`](workshop_minesweeper.py)     | Umgebung + DQN-Agent + Training + Viewer    |
| [`requirements.txt`](requirements.txt)                   | Abhängigkeiten                              |

---

## Ausführen

```bash
cd minesweeper

# Trainieren (Modell wird gespeichert)
python workshop_minesweeper.py --train

# Trainieren UND direkt interaktiv ansehen
python workshop_minesweeper.py --train --play

# Gespeichertes Modell Zug für Zug validieren
python workshop_minesweeper.py --play

# Kleineres Feld (konvergiert schneller, ideal zum Ausprobieren)
python workshop_minesweeper.py --train --play --rows 5 --cols 5 --mines 4
```

Im Viewer steuern die Buttons **Weiter ▶**, **◀ Zurück** und **Neues Spiel ⟳**
durch die Partie. Die zuletzt gewählte Zelle ist gelb (bzw. rot bei einer Mine) umrandet.

---

## Ausgabe-Dateien

```
saved_model/minesweeper_dqn.keras   ← trainiertes Modell
plots/training_minesweeper.png      ← Lernkurve + Siegrate
```

> Hinweis: `saved_model/` und `plots/` werden über die `.gitignore` des Repos nicht eingecheckt.

---

## Weiterführende Workshops

[Reinforcement Learning (Grundlagen)](../reinforcement_learning/) ·
[Explorative Datenanalyse](../explorative_datenanalyse/) ·
[Bildklassifikation](../iamge_classification/) ·
[Autoencoder](../autoencoder/)
