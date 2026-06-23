# Workshop: Reinforcement Learning mit Python (Q-Learning, SARSA & DQN)

## Überblick

In diesem Workshop bringst du einem **Agenten** bei, sich selbstständig durch eine
kleine Welt (**GridWorld**) zu navigieren – ganz ohne vorgegebene Labels, nur über
**Belohnung (Reward)** und **Bestrafung**. Du durchläufst den kompletten
Reinforcement-Learning-Workflow:

1. **Umgebung verstehen** – eine selbstgebaute GridWorld (Start, Ziel, Fallen, Wände)
2. **Agent definieren** – tabellarisches **Q-Learning** (off-policy) oder **SARSA** (on-policy)
3. **Exploration vs. Exploitation** – epsilon-greedy mit Decay
4. **Trainieren** – Lernen durch Interaktion über viele Episoden
5. **Evaluieren** – die gelernte Policy ohne Exploration bewerten
6. **Visualisieren** – Lernkurve, Epsilon-Decay, Policy-Pfeile, State-Value-Heatmap
7. **Bonus** – Sprung zu **Deep RL**: ein **Deep-Q-Network (DQN)** auf Gymnasium CartPole

> Dieser Workshop erweitert die bestehende Data-Science-Workshop-Reihe
> (`explorative_datenanalyse`, `image_classification`, `autoencoder`) um das Thema
> **Reinforcement Learning** und nutzt denselben Aufbau (zentraler `CONFIG`-Block,
> CLI-Parameter, gespeicherte Plots).

---

## Lernziele

- Den Unterschied zwischen **überwachtem Lernen** (Labels) und **Reinforcement Learning** (Reward-Signal) verstehen
- Die zentralen RL-Begriffe kennen: **State, Action, Reward, Policy, Episode, Discount-Faktor**
- Den **Trade-off Exploration vs. Exploitation** (epsilon-greedy) nachvollziehen
- Den Unterschied zwischen **Q-Learning** (off-policy) und **SARSA** (on-policy) erfahren
- Den Einfluss von **Hyperparametern** (α, γ, ε-Decay) auf das Lernverhalten beobachten
- Den Übergang von **tabellarischem RL** zu **Deep RL (DQN)** verstehen

---

## Voraussetzungen

- **Python 3.10+** (empfohlen)
- Basis-Pakete (für GridWorld + Tabellen-Agent):

```bash
pip install numpy matplotlib
```

- Optional für den **DQN-Bonus**:

```bash
pip install gymnasium tensorflow
```

> **Hinweis:** Der Kern-Workshop (GridWorld + Q-Learning/SARSA) läuft komplett ohne
> Deep-Learning-Frameworks und damit auch auf schwächeren Rechnern flott.

---

## Die Umgebung: GridWorld

Der Agent startet in **S** und soll das **Ziel Z** erreichen. Dabei muss er
**Wände (█)** umgehen und **Fallen (X)** vermeiden:

```
| · | · | X | · | Z |
| · |███| · |███| · |
| · |███| · | · | · |
| S | · | · | · | · |
```

**Belohnungsstruktur (Reward):**

| Ereignis            | Reward | Episode endet? |
|---------------------|:------:|:--------------:|
| Schritt (normal)    |  −1    | nein           |
| Ziel erreicht       | +10    | ja             |
| In Falle gelaufen   | −10    | ja             |
| Max. Schritte (100) |  −1    | ja             |

Der Schritt-Reward von **−1** motiviert den Agenten, **möglichst kurze Wege** zu finden.

---

## Ausführen

### Agent trainieren (Standard: Q-Learning)

```bash
# Standard-Training mit Q-Learning
python workshop_reinforcement_learning.py --train

# Mit SARSA statt Q-Learning
python workshop_reinforcement_learning.py --train --algo sarsa

# Training + finale Bewertung + Policy als ASCII anzeigen
python workshop_reinforcement_learning.py --train --evaluate --render

# Mehr Episoden
python workshop_reinforcement_learning.py --train --episodes 1000
```

### Bonus: Deep-Q-Network auf CartPole

```bash
python workshop_reinforcement_learning.py --dqn
```

---

## Alle CLI-Parameter

| Parameter      | Beschreibung                                       | Standard     |
|----------------|----------------------------------------------------|--------------|
| `--train`      | Tabellarischen Agenten trainieren                  | –            |
| `--evaluate`   | Gelernte greedy-Policy am Ende bewerten            | –            |
| `--render`     | Gitter & gelernte Policy als ASCII ausgeben        | –            |
| `--algo`       | `q_learning` oder `sarsa`                           | `q_learning` |
| `--episodes`   | Anzahl Trainings-Episoden                          | 500          |
| `--dqn`        | Bonus: DQN auf CartPole (gymnasium + tensorflow)   | –            |

---

## Hyperparameter zum Experimentieren

Im `CONFIG`-Block am Anfang des Skripts findest du alle Parameter gesammelt.

### Lernen

| Parameter        | Beschreibung                                   | Werte zum Testen        |
|------------------|------------------------------------------------|-------------------------|
| `ALGO`           | Lernalgorithmus                                | `q_learning`, `sarsa`   |
| `ALPHA`          | Lernrate (α) – wie stark neue Erfahrung zählt  | 0.01, 0.1, 0.3, 0.5     |
| `GAMMA`          | Discount-Faktor (γ) – Gewicht der Zukunft      | 0.8, 0.9, 0.95, 0.99    |
| `EPSILON_DECAY`  | Wie schnell die Exploration abnimmt            | 0.99, 0.995, 0.999      |
| `EPSILON_MIN`    | Minimale Explorationsrate                      | 0.0, 0.05, 0.1          |
| `EPISODES`       | Anzahl Trainings-Episoden                      | 200, 500, 1000          |

### Umgebung

| Parameter      | Beschreibung                          | Werte zum Testen          |
|----------------|---------------------------------------|---------------------------|
| `GRID_ROWS/COLS` | Größe des Gitters                   | 4×5, 6×6, 8×8             |
| `OBSTACLES`    | Positionen der Wände                  | eigene Layouts            |
| `TRAPS`        | Positionen der Fallen                 | mehr/weniger Fallen       |
| `STEP_REWARD`  | Kosten pro Schritt                    | 0, −0.1, −1               |
| `TRAP_REWARD`  | Strafe in einer Falle                 | −5, −10, −50              |

---

## Aufgaben

### Aufgabe 1: Baseline trainieren
Starte das Skript mit Standard-Einstellungen (`--train --evaluate --render`) und notiere:
- Nach wie vielen Episoden steigt die Erfolgsrate(50) deutlich an?
- Welchen durchschnittlichen Reward erreicht die finale greedy-Policy?
- Sieht die in `plots/policy_values.png` gezeigte Policy sinnvoll aus (kürzester Weg um die Fallen)?

### Aufgabe 2: Exploration vs. Exploitation
Der Agent muss früh **explorieren** (Neues ausprobieren) und später **exploitieren** (Gelerntes nutzen).
- Setze `EPSILON_DECAY = 0.999` (langsamer Decay) – wie verändert sich die Lernkurve?
- Setze `EPSILON_MIN = 0.0` und `EPSILON_DECAY = 0.95` (schneller Decay) – wird der Agent zu früh „gierig"?
- Schau dir `plots/epsilon_decay.png` an und verknüpfe den Verlauf mit der Lernkurve.

### Aufgabe 3: Lernrate & Discount
- Vergleiche `ALPHA = 0.01` vs. `ALPHA = 0.5` – was passiert mit Stabilität und Tempo?
- Vergleiche `GAMMA = 0.8` vs. `GAMMA = 0.99` – plant der Agent kurzfristiger oder langfristiger?
- Beobachte jeweils die State-Value-Heatmap: Wie „weit" strahlt der Wert vom Ziel aus?

### Aufgabe 4: Q-Learning vs. SARSA
Trainiere denselben Aufbau einmal mit `--algo q_learning` und einmal mit `--algo sarsa`.
- Vergleiche die gelernten Policies in der Nähe der Falle.
- **Hypothese:** SARSA (on-policy) verhält sich oft „vorsichtiger" in der Nähe von Fallen, weil es das eigene Explorationsverhalten mitlernt. Bestätigt sich das?

### Aufgabe 5: Eigene Welt bauen
Verändere im `CONFIG`-Block `GRID_ROWS`, `GRID_COLS`, `OBSTACLES`, `TRAPS`, `START`, `GOAL`.
- Baue ein Labyrinth mit einer „Abkürzung", die an einer Falle vorbeiführt.
- Schafft es der Agent, den riskanten kurzen Weg gegen den sicheren langen Weg abzuwägen?

### Bonus: Deep RL mit DQN
Führe `--dqn` aus (benötigt `gymnasium` + `tensorflow`).
- Warum braucht CartPole ein **neuronales Netz** statt einer Q-Tabelle? (Stichwort: kontinuierlicher Zustandsraum)
- Was bewirken **Experience Replay** und das **Target-Netzwerk** für die Stabilität?
- Erreicht der Agent in `plots/dqn_cartpole.png` den Maximal-Reward von 500?

---

## Ausgabe-Dateien

Nach dem Training findest du folgende Diagramme:

```
plots/
├── learning_curve.png   ← Reward pro Episode + Episodenlänge
├── epsilon_decay.png    ← Verlauf der Explorationsrate ε
├── policy_values.png    ← State-Value-Heatmap mit Policy-Pfeilen
└── dqn_cartpole.png     ← (nur bei --dqn) DQN-Lernkurve
```

---

## Die RL-Schleife (Überblick)

```
        ┌─────────────────────────────────────────┐
        │                                         │
        │   Zustand sₜ                            │
        ▼                                         │
   ┌─────────┐   Aktion aₜ (epsilon-greedy)   ┌───────────┐
   │  AGENT  │ ──────────────────────────────▶ │  UMGEBUNG │
   │ (Q[s,a])│                                 │(GridWorld)│
   └─────────┘ ◀────────────────────────────── └───────────┘
        ▲        Reward rₜ, neuer Zustand sₜ₊₁       │
        │                                            │
        └──────────  TD-Update: Q(s,a) anpassen  ────┘
```

**Q-Learning (off-policy):**  `Q(s,a) ← Q(s,a) + α·[ r + γ·maxₐ' Q(s',a') − Q(s,a) ]`

**SARSA (on-policy):**         `Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s',a')      − Q(s,a) ]`

---

## Tipps & Tricks

- **Lernt der Agent nicht?** Mehr Episoden, höhere Lernrate oder langsameren ε-Decay testen.
- **Policy wirkt zufällig?** ε ist evtl. zu hoch geblieben – prüfe `EPSILON_MIN`/`EPSILON_DECAY`.
- **Agent läuft in Fallen?** `TRAP_REWARD` stärker negativ machen oder `GAMMA` erhöhen (mehr Weitsicht).
- **Reproduzierbarkeit:** `RANDOM_SEED` fixiert die Zufallswerte – für faire Vergleiche gleich lassen.
- **Heatmap lesen:** Helle Felder = hoher erwarteter Wert. Der „Wert-Gradient" sollte vom Ziel ausstrahlen.

---

## Weiterführende Ideen

- **Double DQN / Dueling DQN:** stabilere Deep-RL-Varianten
- **Policy-Gradient-Methoden:** REINFORCE, Actor-Critic (A2C/PPO)
- **Reward Shaping:** Zwischenbelohnungen zur Beschleunigung des Lernens
- **Stochastische Umgebung:** Aktionen mit Wahrscheinlichkeit „rutschen" lassen (slippery GridWorld)
- **Andere Gymnasium-Umgebungen:** MountainCar, LunarLander, Acrobot

---

## Referenzen

- [Sutton & Barto – Reinforcement Learning: An Introduction (Standardwerk, frei verfügbar)](http://incompleteideas.net/book/the-book.html)
- [Gymnasium Dokumentation](https://gymnasium.farama.org/)
- [Q-Learning (Wikipedia)](https://de.wikipedia.org/wiki/Q-Learning)
- [DeepMind: Human-level control through deep RL (DQN-Paper)](https://www.nature.com/articles/nature14236)
