#!/usr/bin/env python3
"""
Workshop: Reinforcement Learning mit Python (GridWorld + Q-Learning / SARSA / DQN)
=================================================================================
STARTER-CODE MIT LÜCKEN (TODOs)
--------------------------------
Dies ist die Aufgaben-Version des Skripts. Das komplette Gerüst (CONFIG,
GridWorld-Umgebung, Trainings-Loop, Evaluation, Plotting, argparse) ist bereits
vorhanden. Deine Aufgabe ist es, die didaktisch zentralen Stellen selbst zu
implementieren – sie sind mit

    # ==========...
    # TODO (Aufgabe X): ...
    # ==========...
    raise NotImplementedError(...)

markiert. Die vollständige Musterlösung findest du unter:
    ../loesung/workshop_reinforcement_learning.py

Dieses Skript demonstriert den vollständigen Workflow eines
Reinforcement-Learning-Experiments:

  1. Umgebung (Environment) definieren  -> selbstgebaute GridWorld
  2. Agent definieren                   -> tabellarisches Q-Learning oder SARSA
  3. Exploration vs. Exploitation       -> epsilon-greedy mit Decay
  4. Training (Interaktion mit der Umwelt über Episoden)
  5. Evaluation der gelernten Policy (greedy)
  6. Visualisierung: Lernkurve, Policy, State-Value-Heatmap, Epsilon-Decay
  7. Bonus: Deep-Q-Network (DQN) auf Gymnasium CartPole (optional, mit TensorFlow)

Reinforcement Learning unterscheidet sich von überwachtem Lernen:
Es gibt keine vorgegebenen Labels, sondern nur ein *Reward-Signal*. Der Agent
muss durch Ausprobieren (Trial & Error) selbst herausfinden, welche Aktionen
langfristig die höchste kumulierte Belohnung bringen.

Aufruf:
  python workshop_reinforcement_learning.py --train
  python workshop_reinforcement_learning.py --train --algo sarsa
  python workshop_reinforcement_learning.py --train --evaluate --render
  python workshop_reinforcement_learning.py --dqn          # Bonus (benötigt gymnasium + tensorflow)
"""

import os
import sys
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")            # << Nicht-interaktives Backend (Server-tauglich)
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════╗
# ║               HYPERPARAMETER-KONFIGURATION               ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Hier kannst du alle wichtigen Parameter zentral         ║
# ║  anpassen und experimentieren.                           ║
# ╚══════════════════════════════════════════════════════════╝
CONFIG = {
    # ── Umgebung (GridWorld) ───────────────────────────────
    "GRID_ROWS": 4,                   # << Höhe des Gitters
    "GRID_COLS": 5,                   # << Breite des Gitters
    "START": (3, 0),                  # Startzelle (Zeile, Spalte)
    "GOAL": (0, 4),                   # Zielzelle (+Belohnung)
    "OBSTACLES": [(1, 1), (2, 1), (1, 3)],   # << Wände, die blockieren
    "TRAPS": [(0, 2)],                # << Fallen (negative Belohnung, Episodenende)
    "STEP_REWARD": -1.0,              # << Kosten pro Schritt (motiviert kurze Wege)
    "GOAL_REWARD": 10.0,              # << Belohnung beim Ziel
    "TRAP_REWARD": -10.0,             # << Strafe in einer Falle
    "MAX_STEPS": 100,                 # << Max. Schritte pro Episode

    # ── Agent / Lernen ─────────────────────────────────────
    "ALGO": "q_learning",             # << "q_learning" (off-policy) oder "sarsa" (on-policy)
    "ALPHA": 0.10,                    # << Lernrate (0.01–0.5)
    "GAMMA": 0.95,                    # << Discount-Faktor (0.8–0.99): Gewicht der Zukunft
    "EPSILON_START": 1.0,             # << Start-Explorationsrate
    "EPSILON_MIN": 0.05,              # << Minimale Explorationsrate
    "EPSILON_DECAY": 0.995,           # << Multiplikativer Decay pro Episode
    "EPISODES": 500,                  # << Anzahl Trainings-Episoden

    # ── Sonstiges ──────────────────────────────────────────
    "RANDOM_SEED": 42,
    "PLOT_DIR": "plots",              # Ordner für Diagramme
    "EVAL_EPISODES": 100,             # Episoden für die finale Evaluation
}

# Aktionen: Index -> (d_row, d_col) und Symbol für die Policy-Visualisierung
ACTIONS = {
    0: (-1, 0),   # hoch
    1: (1, 0),    # runter
    2: (0, -1),   # links
    3: (0, 1),    # rechts
}
ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


# ╔══════════════════════════════════════════════════════════╗
# ║                     HILFSFUNKTIONEN                      ║
# ╚══════════════════════════════════════════════════════════╝

def set_seeds(seed: int) -> None:
    """Reproduzierbare Ergebnisse durch fixierte Zufallsseeds."""
    np.random.seed(seed)


# ╔══════════════════════════════════════════════════════════╗
# ║                     GRIDWORLD-UMGEBUNG                   ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Eine kleine, selbstgebaute Umgebung. Der Agent startet  ║
# ║  in START und soll GOAL erreichen, ohne in eine TRAP zu  ║
# ║  laufen. Wände (OBSTACLES) blockieren Bewegungen.        ║
# ╚══════════════════════════════════════════════════════════╝
class GridWorld:
    """Diskrete GridWorld-Umgebung mit Gym-ähnlicher API (reset/step)."""

    def __init__(self, cfg: dict):
        self.rows = cfg["GRID_ROWS"]
        self.cols = cfg["GRID_COLS"]
        self.start = tuple(cfg["START"])
        self.goal = tuple(cfg["GOAL"])
        self.obstacles = {tuple(o) for o in cfg["OBSTACLES"]}
        self.traps = {tuple(t) for t in cfg["TRAPS"]}
        self.step_reward = cfg["STEP_REWARD"]
        self.goal_reward = cfg["GOAL_REWARD"]
        self.trap_reward = cfg["TRAP_REWARD"]
        self.max_steps = cfg["MAX_STEPS"]

        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTIONS)
        self._validate()
        self.reset()

    def _validate(self) -> None:
        for cell in [self.start, self.goal]:
            if not self._in_bounds(cell):
                raise ValueError(f"Zelle {cell} liegt außerhalb des Gitters.")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("START/GOAL dürfen nicht auf einem Hindernis liegen.")

    def _in_bounds(self, cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def state_to_index(self, cell) -> int:
        r, c = cell
        return r * self.cols + c

    def reset(self) -> int:
        """Setzt die Umgebung zurück und liefert den Start-Zustandsindex."""
        self.agent = self.start
        self.steps = 0
        return self.state_to_index(self.agent)

    def step(self, action: int):
        """
        Führt eine Aktion aus.
        Rückgabe: (next_state_index, reward, done, info)
        """
        self.steps += 1
        d_row, d_col = ACTIONS[action]
        nxt = (self.agent[0] + d_row, self.agent[1] + d_col)

        # Bewegung nur, wenn im Gitter und kein Hindernis -> sonst bleiben
        if self._in_bounds(nxt) and nxt not in self.obstacles:
            self.agent = nxt

        if self.agent == self.goal:
            return self.state_to_index(self.agent), self.goal_reward, True, {"result": "goal"}
        if self.agent in self.traps:
            return self.state_to_index(self.agent), self.trap_reward, True, {"result": "trap"}
        if self.steps >= self.max_steps:
            return self.state_to_index(self.agent), self.step_reward, True, {"result": "timeout"}

        return self.state_to_index(self.agent), self.step_reward, False, {"result": "move"}

    def render_ascii(self, policy: np.ndarray = None) -> str:
        """Erzeugt eine ASCII-Darstellung des Gitters (optional mit Policy-Pfeilen)."""
        lines = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.goal:
                    row_cells.append(" Z ")            # Ziel
                elif cell in self.traps:
                    row_cells.append(" X ")            # Falle
                elif cell in self.obstacles:
                    row_cells.append("███")            # Wand
                elif cell == self.start and policy is None:
                    row_cells.append(" S ")            # Start
                elif policy is not None:
                    a = policy[self.state_to_index(cell)]
                    row_cells.append(f" {ACTION_ARROWS[a]} ")
                else:
                    row_cells.append(" · ")
            lines.append("|" + "|".join(row_cells) + "|")
        return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════╗
# ║              TABELLARISCHER RL-AGENT (Q/SARSA)          ║
# ╚══════════════════════════════════════════════════════════╝
class TabularAgent:
    """
    Tabellarischer Agent mit Q-Tabelle Q[state, action].

    - Q-Learning (off-policy):  Q(s,a) += α [ r + γ·maxₐ' Q(s',a') − Q(s,a) ]
    - SARSA      (on-policy):   Q(s,a) += α [ r + γ·Q(s',a')        − Q(s,a) ]
    """

    def __init__(self, n_states: int, n_actions: int, cfg: dict):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = cfg["ALPHA"]
        self.gamma = cfg["GAMMA"]
        self.algo = cfg["ALGO"]
        self.epsilon = cfg["EPSILON_START"]
        self.epsilon_min = cfg["EPSILON_MIN"]
        self.epsilon_decay = cfg["EPSILON_DECAY"]
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy-Aktionswahl (greedy=True -> reine Exploitation)."""
        # ============================================================
        # TODO (Aufgabe 1): Epsilon-greedy-Aktionswahl implementieren
        # Ziel: Balance zwischen Exploration und Exploitation.
        #   - Mit Wahrscheinlichkeit ε (nur wenn greedy=False): eine
        #     ZUFÄLLIGE Aktion wählen (Exploration).
        #     Tipp: np.random.rand() < self.epsilon  und
        #           np.random.randint(self.n_actions)
        #   - Sonst: die BESTE Aktion laut Q-Tabelle wählen (Exploitation),
        #     also argmax über self.Q[state]. Bei Gleichstand mehrerer
        #     Aktionen zufällig unter den Besten wählen, um Bias zu vermeiden.
        #     Tipp: np.flatnonzero(q_row == q_row.max()) und np.random.choice(...)
        #   Rückgabe: ein int (Aktionsindex 0..n_actions-1).
        # ============================================================
        raise NotImplementedError("TODO (Aufgabe 1): epsilon-greedy-Aktionswahl implementieren")

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> None:
        """Ein TD-Update-Schritt – abhängig vom gewählten Algorithmus."""
        # ============================================================
        # TODO (Aufgabe 2): TD-Update für Q-Learning UND SARSA implementieren
        # Berechne zuerst das TD-Target, dann aktualisiere self.Q[s, a].
        #
        #   1) done == True   -> td_target = r   (kein Folgezustand mehr)
        #   2) SARSA (on-policy):
        #        td_target = r + γ · Q(s', a')      (a' = tatsächlich gewählte Folgeaktion)
        #   3) Q-Learning (off-policy):
        #        td_target = r + γ · maxₐ' Q(s', a')  (beste Folgeaktion)
        #        Tipp: np.max(self.Q[s_next])
        #
        #   Danach das Update (für beide Algorithmen identisch):
        #        Q(s,a) ← Q(s,a) + α · [ td_target − Q(s,a) ]
        #
        #   Hinweis: self.algo ist "sarsa" oder "q_learning".
        # ============================================================
        raise NotImplementedError("TODO (Aufgabe 2): Q-Learning- und SARSA-Update implementieren")

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        """Liefert für jeden Zustand die beste Aktion (argmax über Q)."""
        return np.argmax(self.Q, axis=1)

    def state_values(self) -> np.ndarray:
        """V(s) = maxₐ Q(s,a) – nützlich für die Heatmap."""
        return np.max(self.Q, axis=1)


# ╔══════════════════════════════════════════════════════════╗
# ║                        TRAINING                          ║
# ╚══════════════════════════════════════════════════════════╝

def train(env: GridWorld, agent: TabularAgent, cfg: dict):
    """Trainiert den Agenten über mehrere Episoden und sammelt Statistiken."""
    episode_rewards = []
    episode_steps = []
    epsilons = []
    successes = deque(maxlen=50)

    print(f"[INFO] Starte Training: algo={agent.algo}, episodes={cfg['EPISODES']}")
    for ep in range(1, cfg["EPISODES"] + 1):
        s = env.reset()
        a = agent.select_action(s)
        total_reward = 0.0
        done = False

        while not done:
            s_next, r, done, info = env.step(a)
            # Für SARSA brauchen wir die NÄCHSTE Aktion bereits jetzt (on-policy)
            a_next = agent.select_action(s_next)
            agent.update(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next
            total_reward += r

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(env.steps)
        epsilons.append(agent.epsilon)
        successes.append(1 if info.get("result") == "goal" else 0)

        if ep % max(1, cfg["EPISODES"] // 10) == 0:
            print(f"  Episode {ep:4d} | Reward {total_reward:7.1f} | "
                  f"Schritte {env.steps:3d} | ε {agent.epsilon:.3f} | "
                  f"Erfolgsrate(50) {100*np.mean(successes):5.1f}%")

    stats = {
        "rewards": np.array(episode_rewards),
        "steps": np.array(episode_steps),
        "epsilons": np.array(epsilons),
    }
    return stats


def evaluate(env: GridWorld, agent: TabularAgent, episodes: int) -> dict:
    """Bewertet die gelernte greedy-Policy (ohne Exploration)."""
    rewards, steps, goals = [], [], 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(s, greedy=True)
            s, r, done, info = env.step(a)
            total += r
        rewards.append(total)
        steps.append(env.steps)
        goals += 1 if info.get("result") == "goal" else 0
    result = {
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": 100.0 * goals / episodes,
    }
    print(f"""\
[EVAL] greedy-Policy über {episodes} Episoden:
  Ø Reward       : {result['mean_reward']:.2f}
  Ø Schritte     : {result['mean_steps']:.2f}
  Erfolgsrate    : {result['success_rate']:.1f}%""")
    return result


# ╔══════════════════════════════════════════════════════════╗
# ║                    VISUALISIERUNG                        ║
# ╚══════════════════════════════════════════════════════════╝

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curve(stats: dict, cfg: dict) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    rewards = stats["rewards"]
    ma = moving_average(rewards, window=20)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(rewards, alpha=0.3, label="Reward/Episode")
    ax[0].plot(range(len(rewards) - len(ma), len(rewards)), ma,
               color="red", label="gleitender Ø (20)")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("kumulierter Reward")
    ax[0].set_title("Lernkurve")
    ax[0].legend()

    ax[1].plot(stats["steps"], alpha=0.4, color="green")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Schritte bis Episodenende")
    ax[1].set_title("Episodenlänge")

    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "learning_curve.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Lernkurve gespeichert: {out}")


def plot_epsilon(stats: dict, cfg: dict) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    fig = plt.figure(figsize=(6, 4))
    plt.plot(stats["epsilons"], color="purple")
    plt.xlabel("Episode")
    plt.ylabel("ε (Explorationsrate)")
    plt.title("Epsilon-Decay (Exploration → Exploitation)")
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "epsilon_decay.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Epsilon-Decay gespeichert: {out}")


def plot_value_and_policy(env: GridWorld, agent: TabularAgent, cfg: dict) -> None:
    """Heatmap der State-Values V(s) mit überlagerten Policy-Pfeilen."""
    _ensure_dir(cfg["PLOT_DIR"])
    values = agent.state_values().reshape(env.rows, env.cols)
    policy = agent.greedy_policy()

    # Hindernisse/Fallen für die Darstellung maskieren
    masked = np.array(values, dtype=float)
    for cell in env.obstacles:
        masked[cell] = np.nan

    fig, ax = plt.subplots(figsize=(1.6 * env.cols, 1.6 * env.rows))
    im = ax.imshow(masked, cmap="viridis")
    fig.colorbar(im, ax=ax, label="V(s) = maxₐ Q(s,a)")

    for r in range(env.rows):
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.obstacles:
                ax.text(c, r, "Wand", ha="center", va="center", color="white", fontsize=8)
            elif cell == env.goal:
                ax.text(c, r, "ZIEL", ha="center", va="center", color="white", fontweight="bold")
            elif cell in env.traps:
                ax.text(c, r, "FALLE", ha="center", va="center", color="red", fontweight="bold")
            else:
                a = policy[env.state_to_index(cell)]
                label = ACTION_ARROWS[a]
                if cell == env.start:
                    label = f"S {label}"
                ax.text(c, r, label, ha="center", va="center", color="white", fontsize=12)

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(f"Gelernte Policy & State-Values ({agent.algo})")
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "policy_values.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[PLOT] Policy/Value-Heatmap gespeichert: {out}")


# ╔══════════════════════════════════════════════════════════╗
# ║         BONUS: DEEP-Q-NETWORK (DQN) auf CartPole        ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Zeigt den Schritt von tabellarischem RL zu Deep RL.    ║
# ║  Benötigt zusätzlich: gymnasium, tensorflow             ║
# ╚══════════════════════════════════════════════════════════╝

def run_dqn(cfg: dict) -> None:
    """Trainiert einen einfachen DQN-Agenten auf CartPole-v1 (optional)."""
    try:
        import gymnasium as gym
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        print("[FEHLER] Für den DQN-Bonus werden 'gymnasium' und 'tensorflow' benötigt.")
        print("         Installation:  pip install gymnasium tensorflow")
        print(f"         (Import fehlgeschlagen: {exc})")
        return

    set_seeds(cfg["RANDOM_SEED"])
    tf.random.set_seed(cfg["RANDOM_SEED"])

    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    def build_q_network():
        # ============================================================
        # TODO (Aufgabe 3, Bonus): Q-Netzwerk aufbauen
        # Ein kleines Feedforward-Netz, das aus dem Zustand (n_states Werte)
        # einen Q-Wert PRO Aktion (n_actions Ausgänge) schätzt.
        #   - Eingang:  keras.Input(shape=(n_states,))
        #   - 2 versteckte Dense-Schichten mit je 64 Neuronen, Aktivierung "relu"
        #   - Ausgabeschicht: Dense(n_actions) mit Aktivierung "linear"
        #     (Q-Werte sind reelle Zahlen, keine Wahrscheinlichkeiten!)
        #   - Kompilieren mit optimizer=Adam(lr=1e-3), loss="mse"
        #   Tipp: keras.Sequential([...]) und model.compile(...)
        #   Rückgabe: das kompilierte Modell.
        # ============================================================
        raise NotImplementedError("TODO (Aufgabe 3, Bonus): Q-Netzwerk aufbauen")

    q_net = build_q_network()
    target_net = build_q_network()
    target_net.set_weights(q_net.get_weights())

    replay = deque(maxlen=10000)
    gamma = cfg["GAMMA"]
    epsilon = 1.0
    eps_min, eps_decay = 0.05, 0.995
    batch_size = 64
    episodes = 150
    rewards_hist = []

    print("[INFO] Starte DQN-Training auf CartPole-v1 …")
    for ep in range(1, episodes + 1):
        s, _ = env.reset(seed=cfg["RANDOM_SEED"] + ep)
        done = False
        total = 0.0
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(q_net.predict(s[None, :], verbose=0)[0]))

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            replay.append((s, a, r, s_next, done))
            s = s_next
            total += r

            if len(replay) >= batch_size:
                idx = np.random.choice(len(replay), batch_size, replace=False)
                batch = [replay[i] for i in idx]
                states = np.array([b[0] for b in batch])
                actions = np.array([b[1] for b in batch])
                rew = np.array([b[2] for b in batch])
                next_states = np.array([b[3] for b in batch])
                dones = np.array([b[4] for b in batch], dtype=float)

                q_next = target_net.predict(next_states, verbose=0)
                targets = q_net.predict(states, verbose=0)
                # ============================================================
                # TODO (Aufgabe 4, Bonus): DQN-TD-Target berechnen
                # Ersetze in der Target-Matrix NUR den Eintrag der tatsächlich
                # gewählten Aktion durch das Bellman-Target:
                #     target = r + γ · maxₐ' Q_target(s', a') · (1 − done)
                # Der Faktor (1 − done) sorgt dafür, dass am Episodenende kein
                # Zukunftswert addiert wird (dann ist das Target einfach r).
                #   - q_next     = Q-Werte des Ziel-Netzes für die Folgezustände
                #   - rew, dones = Rewards bzw. done-Flags des Mini-Batches
                #   Tipp: np.max(q_next, axis=1) liefert maxₐ' Q(s',a') je Sample.
                #   Zuweisung: targets[np.arange(batch_size), actions] = ...
                # ============================================================
                raise NotImplementedError("TODO (Aufgabe 4, Bonus): DQN-TD-Target berechnen")
                q_net.fit(states, targets, epochs=1, verbose=0)

        epsilon = max(eps_min, epsilon * eps_decay)
        rewards_hist.append(total)
        if ep % 10 == 0:
            target_net.set_weights(q_net.get_weights())
            print(f"  Episode {ep:3d} | Reward {total:6.1f} | "
                  f"Ø(10) {np.mean(rewards_hist[-10:]):6.1f} | ε {epsilon:.3f}")

    env.close()

    _ensure_dir(cfg["PLOT_DIR"])
    fig = plt.figure(figsize=(7, 4))
    plt.plot(rewards_hist, alpha=0.4, label="Reward/Episode")
    plt.plot(moving_average(np.array(rewards_hist), 10), color="red", label="gleitender Ø (10)")
    plt.xlabel("Episode")
    plt.ylabel("kumulierter Reward")
    plt.title("DQN auf CartPole-v1")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "dqn_cartpole.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[PLOT] DQN-Lernkurve gespeichert: {out}")


# ╔══════════════════════════════════════════════════════════╗
# ║                          MAIN                            ║
# ╚══════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reinforcement-Learning-Workshop (GridWorld + DQN)")
    p.add_argument("--train", action="store_true", help="Tabellarischen Agenten trainieren")
    p.add_argument("--evaluate", action="store_true", help="Gelernte Policy am Ende bewerten")
    p.add_argument("--render", action="store_true", help="Gitter & Policy als ASCII ausgeben")
    p.add_argument("--algo", choices=["q_learning", "sarsa"], default=None,
                   help="Lernalgorithmus (überschreibt CONFIG['ALGO'])")
    p.add_argument("--episodes", type=int, default=None, help="Anzahl Trainings-Episoden")
    p.add_argument("--dqn", action="store_true", help="Bonus: DQN auf CartPole (gymnasium+tensorflow)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)

    if args.algo is not None:
        cfg["ALGO"] = args.algo
    if args.episodes is not None:
        cfg["EPISODES"] = args.episodes

    if args.dqn:
        run_dqn(cfg)
        return

    if not args.train:
        print("Nichts zu tun. Nutze --train (optional --evaluate --render) oder --dqn.")
        print("Beispiel:  python workshop_reinforcement_learning.py --train --evaluate --render")
        return

    set_seeds(cfg["RANDOM_SEED"])
    env = GridWorld(cfg)

    print("[INFO] GridWorld-Layout (S=Start, Z=Ziel, X=Falle, █=Wand):")
    print(env.render_ascii())

    agent = TabularAgent(env.n_states, env.n_actions, cfg)
    stats = train(env, agent, cfg)

    plot_learning_curve(stats, cfg)
    plot_epsilon(stats, cfg)
    plot_value_and_policy(env, agent, cfg)

    if args.render:
        print("\n[INFO] Gelernte greedy-Policy:")
        print(env.render_ascii(policy=agent.greedy_policy()))

    if args.evaluate:
        evaluate(env, agent, cfg["EVAL_EPISODES"])


if __name__ == "__main__":
    main()
