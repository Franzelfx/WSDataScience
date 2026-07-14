# Workshops Data Science

Sammlung praxisnaher Hands-on-Workshops rund um Data Science und Machine Learning
(Deutsch). Jeder Workshop ist gleich aufgebaut: eine **Aufgabe** zum Selbermachen,
eine vollständige **Lösung** und eine **Anleitung/Erklärung** (`workshop.md`).

---

## Übersicht

Empfohlene Reihenfolge – die Workshops bauen didaktisch aufeinander auf:

| # | Workshop | Thema | Methode |
|---|----------|-------|---------|
| 01 | [Explorative Datenanalyse](workshops/01_explorative_datenanalyse/) | Daten verstehen & visualisieren | pandas, matplotlib, seaborn |
| 02 | [Bildklassifikation](workshops/02_bildklassifikation/) | Bilder klassifizieren | CNN (TensorFlow/Keras) |
| 03 | [Autoencoder](workshops/03_autoencoder/) | Unüberwachtes Lernen / Anomalien | Autoencoder (TensorFlow/Keras) |
| 04 | [Reinforcement Learning](workshops/04_reinforcement_learning/) | Lernen durch Belohnung | Q-Learning, SARSA, DQN |
| 05 | [RL auf Minesweeper](workshops/05_minesweeper/) | RL angewandt (interaktiv) | Deep-Q-Network (CNN) |

---

## Aufbau jedes Workshops

Alle Workshops folgen demselben Schema:

```
workshops/NN_thema/
├── README.md          ← Kurzübersicht, Lernziele, Setup, Ausführen
├── workshop.md        ← Ausführliche Anleitung & Erklärung der Konzepte
├── aufgabe/           ← Starter mit Lücken (# TODO) – hier arbeitest du
├── loesung/           ← Vollständige Musterlösung
├── requirements.txt   ← Abhängigkeiten
└── data/              ← Beispieldaten (nur wo nötig)
```

**Empfohlener Ablauf:** `README.md` lesen → `workshop.md` durcharbeiten →
in `aufgabe/` die `# TODO`-Lücken füllen → mit `loesung/` vergleichen.

---

## Schnellstart

Jeder Workshop ist eigenständig. Beispiel Reinforcement Learning:

```bash
cd workshops/04_reinforcement_learning
pip install -r requirements.txt

# Eigene Umsetzung (nach dem Füllen der TODOs)
python aufgabe/workshop_reinforcement_learning.py --train --evaluate --render

# Oder direkt die Musterlösung ausführen
python loesung/workshop_reinforcement_learning.py --train --evaluate --render
```

Die jeweilige `workshop.md` enthält Lernziele, Aufgaben und alle Parameter.

---

## Hinweise

- **Python 3.10+** empfohlen. Details je Workshop in der `requirements.txt`.
- Generierte Ausgaben (`plots/`, `saved_model/`, `*.keras`, `venv/`) sind
  über die `.gitignore` bewusst nicht eingecheckt.
