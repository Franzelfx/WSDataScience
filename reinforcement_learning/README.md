# Workshop: Reinforcement Learning (Q-Learning, SARSA & DQN)

## Überblick

In diesem Workshop bringst du einem **Agenten** bei, sich selbstständig durch eine
kleine Welt (**GridWorld**) zu navigieren – ohne vorgegebene Labels, nur über
**Belohnung (Reward)** und **Bestrafung**. Anschließend folgt der Sprung zu **Deep RL**
mit einem **Deep-Q-Network (DQN)** auf Gymnasium CartPole.

Die ausführliche Anleitung mit allen Aufgaben steht in **[workshop.md](workshop.md)**.

---

## Lernziele

- Unterschied zwischen **überwachtem Lernen** (Labels) und **RL** (Reward-Signal) verstehen
- Zentrale Begriffe kennen: **State, Action, Reward, Policy, Episode, Discount-Faktor**
- Den Trade-off **Exploration vs. Exploitation** (epsilon-greedy) nachvollziehen
- Unterschied zwischen **Q-Learning** (off-policy) und **SARSA** (on-policy) erfahren
- Übergang von **tabellarischem RL** zu **Deep RL (DQN)** verstehen

---

## Voraussetzungen

- **Python 3.10+**

```bash
pip install -r requirements.txt        # numpy, matplotlib (Kern-Workshop)
pip install gymnasium tensorflow       # optional: DQN-Bonus
```

> Der Kern-Workshop (GridWorld + Q-Learning/SARSA) läuft komplett ohne
> Deep-Learning-Frameworks.

---

## Dateien

| Datei                                                                          | Beschreibung                              |
|--------------------------------------------------------------------------------|-------------------------------------------|
| [`workshop.md`](workshop.md)                                                   | Vollständige Workshop-Anleitung           |
| [`workshop_reinforcement_learning.py`](workshop_reinforcement_learning.py)     | Starter-Code mit zentralem `CONFIG`-Block |
| [`requirements.txt`](requirements.txt)                                         | Abhängigkeiten                            |

---

## Ausführen

```bash
cd reinforcement_learning

# Standard-Training (Q-Learning) + Bewertung + Policy anzeigen
python workshop_reinforcement_learning.py --train --evaluate --render

# SARSA statt Q-Learning
python workshop_reinforcement_learning.py --train --algo sarsa

# Bonus: Deep-Q-Network auf CartPole
python workshop_reinforcement_learning.py --dqn
```

---

## Ausgabe-Dateien

```
plots/
├── learning_curve.png   ← Reward pro Episode + Episodenlänge
├── epsilon_decay.png    ← Verlauf der Explorationsrate ε
├── policy_values.png    ← State-Value-Heatmap mit Policy-Pfeilen
└── dqn_cartpole.png     ← (nur bei --dqn) DQN-Lernkurve
```

> Hinweis: `plots/` wird über die `.gitignore` des Repos nicht eingecheckt.

---

## Weiterführende Workshops

[Explorative Datenanalyse](../explorative_datenanalyse/) ·
[Bildklassifikation](../iamge_classification/) ·
[Autoencoder](../autoencoder/)
