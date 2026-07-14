# Workshop: Reinforcement Learning auf Minesweeper (Deep-Q-Network)

## Überblick

In diesem Workshop trainierst du einen **RL-Agenten**, der **Minesweeper** spielt –
und schaust ihm anschließend **interaktiv beim Spielen zu**: Über einen
**„Weiter"-Knopf** schaltest du Zug für Zug durch eine Partie und siehst genau,
welche Zelle der Agent als Nächstes aufdeckt.

Du durchläufst den kompletten Deep-RL-Workflow:

1. **Umgebung** – ein selbstgebautes Minesweeper mit Gym-ähnlicher API
2. **Agent** – ein **Deep-Q-Network (DQN)** als faltendes neuronales Netz (CNN)
3. **Training** – Lernen durch Experience Replay & Target-Netzwerk
4. **Validierung** – **interaktiver Viewer** mit Buttons: *Weiter*, *Zurück*, *Neues Spiel*

> Dieser Workshop baut auf dem [Reinforcement-Learning-Workshop](../05_reinforcement_learning/)
> auf. Dort lernst du die Grundlagen (Q-Learning, SARSA, einfaches DQN auf CartPole) –
> hier wendest du **Deep RL** auf ein deutlich anspruchsvolleres Problem an.

> **Aufbau des Ordners:** Der Starter-Code mit Lücken liegt in
> [`aufgabe/workshop_minesweeper.py`](aufgabe/workshop_minesweeper.py), die
> vollständige Musterlösung in
> [`loesung/workshop_minesweeper.py`](loesung/workshop_minesweeper.py).
> Fülle zuerst die drei **Programmier-Aufgaben** (siehe unten) in der
> `aufgabe/`-Version aus – erst danach lässt sich trainieren.

---

## Lernziele

- Ein nicht-triviales Spiel als **RL-Problem** modellieren (State, Action, Reward)
- Verstehen, warum ein **CNN** für ein 2D-Spielfeld die natürliche Wahl ist
- **Action-Masking** kennenlernen (nur verdeckte Zellen sind gültige Aktionen)
- **Reward Shaping** anwenden (sichere Zelle, Sieg, Mine)
- Einen DQN mit **Experience Replay** und **Target-Netzwerk** trainieren
- Das Verhalten eines Agenten **interaktiv validieren** und kritisch bewerten

---

## Voraussetzungen

- **Python 3.10+**

```bash
pip install -r requirements.txt        # numpy, matplotlib, tensorflow
```

> **Wichtig für den interaktiven Viewer:** Es wird ein **GUI-fähiges Matplotlib-Backend**
> benötigt (z. B. `MacOSX` oder `TkAgg`). Auf einem reinen Server (ohne Display)
> funktioniert das Training und Plotten, der `--play`-Viewer aber nicht.

---

## Das Spiel als RL-Problem

| Begriff      | Umsetzung im Minesweeper                                                        |
|--------------|---------------------------------------------------------------------------------|
| **State**    | Spielfeld-Sicht: verdeckte Zellen = `-1`, aufgedeckte Zellen = Anzahl Nachbarminen (0–8) |
| **Action**   | Eine von `ROWS × COLS` Zellen aufdecken (nur verdeckte sind gültig → *Action-Masking*) |
| **Reward**   | sichere Zelle `+0.3`, Sieg `+1.0`, Mine `−1.0`                                   |
| **Episode**  | Endet bei einer getroffenen Mine (Niederlage) oder wenn alle sicheren Zellen offen sind (Sieg) |

**State-Kodierung fürs CNN:** Jede Zelle wird als **One-Hot über 10 Kanäle** dargestellt
(Kanal 0 = verdeckt, Kanäle 1–9 = Zahlen 0–8). Das Netz ist **voll-faltend** und gibt
**pro Zelle einen Q-Wert** aus – also „Wie gut ist es, hier aufzudecken?".

---

## Ausführen

### Trainieren

```bash
cd loesung        # fertige Musterlösung  (oder: cd aufgabe, sobald die TODOs gefüllt sind)

# Standard-Training (6x6, 6 Minen) – Modell wird gespeichert
python workshop_minesweeper.py --train

# Trainieren UND danach direkt interaktiv ansehen
python workshop_minesweeper.py --train --play

# Mehr Episoden / andere Feldgröße
python workshop_minesweeper.py --train --episodes 3000 --rows 8 --cols 8 --mines 10
```

### Interaktiv validieren (Weiter-Knopf)

```bash
# Gespeichertes Modell laden und Spiele Zug für Zug durchschalten
python workshop_minesweeper.py --play
```

Im Fenster:

- **Weiter ▶** – nächster Zug des Agenten
- **◀ Zurück** – einen Zug zurück
- **Neues Spiel ⟳** – frische Partie mit demselben Modell

Die zuletzt gewählte Zelle ist **gelb umrandet**; bei einer getroffenen Mine wird
sie **rot** markiert und alle Minen aufgedeckt (✷).

---

## Alle CLI-Parameter

| Parameter     | Beschreibung                                  | Standard |
|---------------|-----------------------------------------------|----------|
| `--train`     | Agenten trainieren (speichert das Modell)     | –        |
| `--play`      | Interaktiver Viewer mit Weiter-Knopf          | –        |
| `--episodes`  | Anzahl Trainings-Episoden                     | 800      |
| `--rows`      | Höhe des Felds                                | 6        |
| `--cols`      | Breite des Felds                              | 6        |
| `--mines`     | Anzahl Minen                                  | 6        |

---

## Hyperparameter zum Experimentieren

Alle Parameter stehen im `CONFIG`-Block am Anfang des Skripts.

| Parameter        | Beschreibung                                | Werte zum Testen        |
|------------------|---------------------------------------------|-------------------------|
| `N_MINES`        | Schwierigkeit (Minendichte)                 | 4, 6, 10                |
| `R_SAFE`         | Belohnung pro sicherer Zelle                | 0.1, 0.3, 0.5           |
| `R_MINE`         | Strafe bei einer Mine                       | −1, −5, −10             |
| `GAMMA`          | Discount-Faktor                             | 0.85, 0.90, 0.95        |
| `LEARNING_RATE`  | Lernrate                                     | 1e-4, 5e-4, 1e-3        |
| `EPSILON_DECAY`  | Wie schnell die Exploration abnimmt         | 0.995, 0.997, 0.999     |
| `CONV_FILTERS`   | Filter pro Conv-Block                        | `[32,32]`, `[64,64,64]` |
| `EPISODES`       | Trainingsdauer                               | 800, 3000, 10000        |

---

## Programmier-Aufgaben (Lücken füllen)

In [`aufgabe/workshop_minesweeper.py`](aufgabe/workshop_minesweeper.py) sind die drei
didaktisch zentralen Stellen als `TODO` markiert. Fülle sie aus – jede blockiert
mit `raise NotImplementedError`, bis sie gelöst ist. Zum Abgleich dient
[`loesung/workshop_minesweeper.py`](loesung/workshop_minesweeper.py).

### Aufgabe 1: Zustands-Kodierung (`encode`)
Kodiere die Spielfeld-Sicht als **One-Hot über 10 Kanäle** (Kanal 0 = verdeckt,
Kanäle 1–9 = Zahlen 0–8). Das ist der Eingang des CNN.
- Warum One-Hot statt der rohen Zahlen −1…8 als ein einzelner Kanal?
- Welche Form hat der resultierende Tensor?

### Aufgabe 2: CNN-Architektur (`DQNAgent._build`)
Baue das **voll-faltende Q-Netz**: mehrere `Conv2D`-Blöcke mit `padding="same"`,
abgeschlossen durch eine `Conv2D(1, 1)` und `Flatten` → **ein Q-Wert pro Zelle**.
- Warum erhält `padding="same"` die Spielfeldgröße?
- Wieso liefert die `1×1`-Faltung genau `ROWS×COLS` Ausgänge?

### Aufgabe 3: DQN-Trainingsschritt (`DQNAgent.replay`)
Implementiere das **TD-Target** mit dem Target-Netz, das **Action-Masking** der
Folgezustände und den Gradientenschritt (`train_on_batch`).
- Warum wird der Zukunftsterm bei `done == 1` weggelassen?
- Welche Rolle spielt das separate **Target-Netz** für die Stabilität?

> Ist alles gefüllt, `python -m py_compile aufgabe/workshop_minesweeper.py` prüft die
> Syntax; danach `cd aufgabe && python workshop_minesweeper.py --train`.

---

## Experimente & Vertiefung

Diese Experimente laufen mit dem **fertigen** Code (deine gefüllte `aufgabe/` oder
die `loesung/`). Sie verändern nur Hyperparameter, keinen Code.

### Experiment 1: Baseline trainieren
Starte `--train` mit den Standardwerten und beobachte `plots/training_minesweeper.png`.
- Steigt die **Siegrate** über die Episoden?
- Ab wann „stagniert" das Lernen?

### Experiment 2: Agent beim Spielen zusehen
Führe `--play` aus und schalte mit **Weiter ▶** durch mehrere Partien.
- Deckt der Agent zuerst sinnvoll in der Fläche auf?
- Erkennt er, dass Zellen neben hohen Zahlen riskanter sind?
- Notiere eine Situation, in der er aus deiner Sicht einen **Fehler** macht.

### Experiment 3: Schwierigkeit variieren
Trainiere mit `--mines 4` und mit `--mines 10` (gleiche Feldgröße).
- Wie verändert sich die erreichbare Siegrate?
- Warum ist Minesweeper bei hoher Minendichte teils **nicht gewinnbar** (Raten nötig)?

### Experiment 4: Reward Shaping
Setze `R_MINE = -10` (im `CONFIG`) und trainiere erneut.
- Wird der Agent „vorsichtiger"?
- Vergleiche Siegrate und durchschnittliche Anzahl aufgedeckter Zellen.

### Experiment 5: Netzarchitektur
Reduziere `CONV_FILTERS` auf `[32, 32]` bzw. erhöhe auf `[128, 128, 128]`.
- Was passiert mit Trainingsdauer und Siegrate?
- Lohnt sich das größere Netz auf einem kleinen Feld?

### Bonus
- Erweitere den `step()`-Reward um einen **Bonus pro neu aufgedeckter Zelle** (`info["newly"]`).
- Implementiere **Double DQN** (Aktionsauswahl mit dem Online-Netz, Bewertung mit dem Target-Netz).
- Baue eine **First-Click-Safe-Regel** (der erste Klick trifft garantiert keine Mine).

---

## Ausgabe-Dateien

```
saved_model/
└── minesweeper_dqn.keras       ← trainiertes Modell

plots/
└── training_minesweeper.png    ← Lernkurve (Reward) + Siegrate
```

> Hinweis: `saved_model/` und `plots/` werden über die `.gitignore` des Repos nicht eingecheckt.

---

## Architektur (Überblick)

```
Spielfeld-Sicht (ROWS × COLS, Werte -1..8)
        │  One-Hot-Kodierung
        ▼
Input (ROWS × COLS × 10)
        │
   Conv2D(64,3) → ReLU
   Conv2D(64,3) → ReLU      (padding="same": Spielfeldgröße bleibt erhalten)
   Conv2D(64,3) → ReLU
   Conv2D(1,1)  → 1 Q-Wert pro Zelle
        │ Flatten
        ▼
Q-Werte (ROWS × COLS)  ──►  Action-Masking (nur verdeckte Zellen)  ──►  argmax = nächster Klick
```

---

## Tipps & Tricks

- **Training dauert.** Mit den Standardwerten lernt der Agent erste sinnvolle Strategien,
  erreicht aber keine Profi-Siegrate. Für bessere Ergebnisse `EPISODES` deutlich erhöhen.
- **Kleiner anfangen.** `--rows 5 --cols 5 --mines 4` konvergiert schneller und ist ideal zum Ausprobieren.
- **Viewer öffnet kein Fenster?** Prüfe das Matplotlib-Backend (`MacOSX`/`TkAgg`) – auf headless-Servern nicht verfügbar.
- **Reproduzierbarkeit:** `RANDOM_SEED` fixiert das Training; der `--play`-Viewer würfelt bewusst neue Spiele.

---

## Weiterführende Ideen

- **Convolutional DQN mit globalem Kontext** (z. B. zusätzliche Dense-Schicht)
- **Prioritized Experience Replay** für effizienteres Lernen
- **Curriculum Learning:** erst kleine Felder, dann größere
- **Policy-Gradient-Methoden** (PPO) statt wertbasiertem DQN

---

## Referenzen

- [DeepMind: Human-level control through deep RL (DQN-Paper)](https://www.nature.com/articles/nature14236)
- [Reinforcement-Learning-Grundlagen-Workshop](../05_reinforcement_learning/)
- [matplotlib Widgets (Button)](https://matplotlib.org/stable/api/widgets_api.html)
