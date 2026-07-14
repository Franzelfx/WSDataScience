#!/usr/bin/env python3
"""
Workshop: Reinforcement Learning auf Minesweeper (Deep-Q-Network) — STARTER
==========================================================================
Dies ist die **Aufgaben-Version** mit drei Lücken (TODOs). Fülle sie aus, bis das
Skript läuft. Die vollständige Musterlösung liegt unter `../loesung/`.

  * Aufgabe 1: One-Hot-Zustandskodierung des Spielfelds  ->  Funktion `encode()`
  * Aufgabe 2: CNN-Architektur des Q-Netzes              ->  `DQNAgent._build()`
  * Aufgabe 3: DQN-Trainingsschritt (TD-Target/Loss)     ->  `DQNAgent.replay()`

Alles andere (Spiellogik, Replay-Buffer, Viewer, Plots, main) ist bereits fertig.

Dieses Skript trainiert einen RL-Agenten, der **Minesweeper** spielt, und stellt
das Ergebnis **interaktiv** dar: Mit einem **„Weiter"-Knopf** kannst du die
einzelnen Spielzüge des Agenten Schritt für Schritt nachvollziehen.

Aufbau:
  1. Umgebung (Environment)  -> selbstgebautes Minesweeper (Gym-ähnliche API)
  2. Agent (DQN)             -> faltendes Q-Netzwerk (CNN) mit Action-Masking
  3. Training                -> Experience Replay + Target-Netzwerk
  4. Validierung             -> interaktiver Viewer (Weiter / Zurück / Neues Spiel)

Warum ein CNN? Das Spielfeld ist ein 2D-Gitter; benachbarte Zellen hängen
räumlich zusammen (eine „3" bedeutet 3 Minen in der Nachbarschaft). Faltungs-
schichten erkennen genau solche lokalen Muster – pro Zelle gibt das Netz einen
Q-Wert „Wie sicher/sinnvoll ist es, hier aufzudecken?" aus.

Aufruf:
  python workshop_minesweeper.py --train                 # trainieren + Modell speichern
  python workshop_minesweeper.py --train --play          # trainieren, dann interaktiv ansehen
  python workshop_minesweeper.py --play                  # gespeichertes Modell interaktiv ansehen
  python workshop_minesweeper.py --train --episodes 2000 # mehr Training
"""

import os
import sys
import argparse
from collections import deque

import numpy as np

# Backend-Wahl: Für den interaktiven Viewer (Weiter-Knopf) brauchen wir ein
# GUI-Backend. Nur fürs reine Training/Plotten genügt das headless-Backend "Agg".
import matplotlib
_INTERACTIVE = ("--play" in sys.argv) or ("--interactive" in sys.argv)
if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════╗
# ║               HYPERPARAMETER-KONFIGURATION               ║
# ╚══════════════════════════════════════════════════════════╝
CONFIG = {
    # ── Spielfeld ──────────────────────────────────────────
    "ROWS": 6,                        # << Höhe des Felds
    "COLS": 6,                        # << Breite des Felds
    "N_MINES": 6,                     # << Anzahl Minen (Schwierigkeit)

    # ── Belohnung (Reward Shaping) ─────────────────────────
    "R_SAFE": 0.3,                    # << sichere Zelle aufgedeckt
    "R_WIN": 1.0,                     # << Spiel gewonnen
    "R_MINE": -1.0,                   # << Mine getroffen (Spielende)

    # ── DQN / Training ─────────────────────────────────────
    "GAMMA": 0.90,                    # << Discount-Faktor
    "LEARNING_RATE": 1e-3,            # << Lernrate des Optimizers
    "EPSILON_START": 1.0,             # << Start-Exploration
    "EPSILON_MIN": 0.05,              # << minimale Exploration
    "EPSILON_DECAY": 0.997,           # << Decay pro Episode
    "EPISODES": 800,                  # << Anzahl Trainings-Episoden
    "BATCH_SIZE": 128,                # << Replay-Batchgröße
    "REPLAY_SIZE": 50000,             # << max. Größe des Replay-Buffers
    "WARMUP": 500,                    # << Transitions vor dem ersten Lernen
    "TARGET_UPDATE": 10,              # << Target-Netz alle N Episoden synchronisieren
    "CONV_FILTERS": [64, 64, 64],     # << Filter pro Conv-Block

    # ── Sonstiges ──────────────────────────────────────────
    "RANDOM_SEED": 42,
    "MODEL_PATH": "saved_model/minesweeper_dqn.keras",
    "PLOT_DIR": "plots",
}

# Zell-Kodierung in der „Sicht" (view) des Agenten
HIDDEN = -1     # noch verdeckt


# ╔══════════════════════════════════════════════════════════╗
# ║                  MINESWEEPER-UMGEBUNG                    ║
# ╚══════════════════════════════════════════════════════════╝
class Minesweeper:
    """
    Minimal-Minesweeper mit Gym-ähnlicher API.

    Aktionen: 0 … ROWS*COLS-1  ->  decke Zelle (action // COLS, action % COLS) auf.
    Beobachtung (view): 2D-Array; verdeckte Zellen = HIDDEN(-1),
                        aufgedeckte Zellen = Anzahl benachbarter Minen (0..8).
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.rows = cfg["ROWS"]
        self.cols = cfg["COLS"]
        self.n_mines = cfg["N_MINES"]
        self.r_safe = cfg["R_SAFE"]
        self.r_win = cfg["R_WIN"]
        self.r_mine = cfg["R_MINE"]
        self.n_cells = self.rows * self.cols
        self.rng = rng
        self.reset()

    def reset(self) -> np.ndarray:
        """Neues Spiel: Minen platzieren, alles verdecken."""
        self.mines = np.zeros((self.rows, self.cols), dtype=bool)
        idx = self.rng.choice(self.n_cells, size=self.n_mines, replace=False)
        for i in idx:
            self.mines[i // self.cols, i % self.cols] = True
        self.counts = self._compute_counts()
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.done = False
        self.result = None  # "win" | "loss" | None
        return self.get_view()

    def _neighbors(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def _compute_counts(self) -> np.ndarray:
        counts = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mines[r, c]:
                    continue
                counts[r, c] = sum(self.mines[nr, nc] for nr, nc in self._neighbors(r, c))
        return counts

    def get_view(self) -> np.ndarray:
        """Sicht des Agenten: HIDDEN für verdeckt, sonst Nachbarschaftszahl."""
        view = np.full((self.rows, self.cols), HIDDEN, dtype=np.int32)
        view[self.revealed] = self.counts[self.revealed]
        return view

    def valid_mask(self) -> np.ndarray:
        """Gültige Aktionen = noch verdeckte Zellen (flach, Länge n_cells)."""
        return (~self.revealed).reshape(-1)

    def _flood_reveal(self, r: int, c: int) -> int:
        """Deckt (r,c) auf; bei einer 0 rekursiv die Nachbarn. Liefert Anzahl neu aufgedeckter Zellen."""
        stack = [(r, c)]
        newly = 0
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr, cc]:
                continue
            self.revealed[cr, cc] = True
            newly += 1
            if self.counts[cr, cc] == 0:
                for nr, nc in self._neighbors(cr, cc):
                    if not self.revealed[nr, nc] and not self.mines[nr, nc]:
                        stack.append((nr, nc))
        return newly

    def step(self, action: int):
        """Führt eine Aktion aus. Rückgabe: (view, reward, done, info)."""
        if self.done:
            raise RuntimeError("Spiel ist beendet – bitte reset() aufrufen.")
        r, c = divmod(action, self.cols)

        if self.revealed[r, c]:
            # Ungültig (sollte dank Action-Masking nicht vorkommen)
            return self.get_view(), -1.0, False, {"event": "invalid", "cell": (r, c)}

        if self.mines[r, c]:
            self.revealed[r, c] = True
            self.done = True
            self.result = "loss"
            return self.get_view(), self.r_mine, True, {"event": "mine", "cell": (r, c)}

        newly = self._flood_reveal(r, c)
        revealed_total = int(self.revealed.sum())
        if revealed_total == self.n_cells - self.n_mines:
            self.done = True
            self.result = "win"
            return self.get_view(), self.r_win, True, {"event": "win", "cell": (r, c), "newly": newly}

        return self.get_view(), self.r_safe, False, {"event": "safe", "cell": (r, c), "newly": newly}


# ╔══════════════════════════════════════════════════════════╗
# ║                 ZUSTANDS-KODIERUNG (für CNN)            ║
# ╠══════════════════════════════════════════════════════════╣
# ║  One-Hot über 10 Kanäle: verdeckt + Zahlen 0..8         ║
# ╚══════════════════════════════════════════════════════════╝
N_CHANNELS = 10

def encode(view: np.ndarray) -> np.ndarray:
    rows, cols = view.shape
    enc = np.zeros((rows, cols, N_CHANNELS), dtype=np.float32)
    # ============================================================
    # TODO (Aufgabe 1): One-Hot-Kodierung des Spielfelds für das CNN.
    #   `enc` hat die Form (rows, cols, 10). Fülle die 10 Kanäle so:
    #     - Kanal 0        = 1.0, wo die Zelle verdeckt ist (view == HIDDEN)
    #     - Kanäle 1..9    = 1.0, wo die aufgedeckte Zahl 0..8 steht
    # Hinweis: `(view == HIDDEN).astype(np.float32)` liefert eine 0/1-Maske
    #          in Feldgröße. Für die Zahlen eignet sich eine Schleife
    #          `for k in range(9): enc[..., k + 1] = (view == k).astype(np.float32)`.
    # ============================================================
    raise NotImplementedError("TODO: One-Hot-Kodierung (10 Kanäle) implementieren")
    return enc


# ╔══════════════════════════════════════════════════════════╗
# ║                       DQN-AGENT                         ║
# ╚══════════════════════════════════════════════════════════╝
class DQNAgent:
    """Faltendes Q-Netzwerk: gibt pro Zelle einen Q-Wert aus (Output-Länge ROWS*COLS)."""

    def __init__(self, cfg: dict):
        from tensorflow import keras  # lazy import (nur wenn Agent gebraucht wird)
        self.keras = keras
        self.rows, self.cols = cfg["ROWS"], cfg["COLS"]
        self.n_cells = self.rows * self.cols
        self.gamma = cfg["GAMMA"]
        self.epsilon = cfg["EPSILON_START"]
        self.eps_min = cfg["EPSILON_MIN"]
        self.eps_decay = cfg["EPSILON_DECAY"]
        self.batch_size = cfg["BATCH_SIZE"]
        self.cfg = cfg
        self.memory = deque(maxlen=cfg["REPLAY_SIZE"])
        self.model = self._build()
        self.target = self._build()
        self.target.set_weights(self.model.get_weights())

    def _build(self):
        keras = self.keras
        from tensorflow.keras import layers
        inp = keras.Input(shape=(self.rows, self.cols, N_CHANNELS))
        # ============================================================
        # TODO (Aufgabe 2): Baue das voll-faltende Q-Netz (CNN) auf.
        #   Ziel: ein Q-Wert PRO Zelle -> Output-Länge ROWS*COLS.
        #   1) Für jede Filterzahl `f` in self.cfg["CONV_FILTERS"] einen
        #      Conv2D-Block: Conv2D(f, kernel_size=3, padding="same",
        #      activation="relu"). `padding="same"` erhält die Feldgröße.
        #   2) Abschluss: Conv2D(1, kernel_size=1, padding="same",
        #      activation="linear") -> genau 1 Kanal (Q-Wert je Zelle).
        #   3) layers.Flatten() -> Vektor der Länge ROWS*COLS.
        # Hinweis: Beginne mit `x = inp`, wende die Layer nacheinander
        #          auf `x` an und lege das Ergebnis in `out` ab. Layer
        #          werden funktional aufgerufen: `x = layers.Conv2D(...)(x)`.
        # ============================================================
        raise NotImplementedError("TODO: CNN-Architektur (Conv-Blöcke + 1x1-Conv + Flatten) bauen")
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(self.cfg["LEARNING_RATE"]), loss="mse")
        return model

    def act(self, view: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy mit Action-Masking (nur verdeckte Zellen wählbar)."""
        mask = (view.reshape(-1) == HIDDEN)
        valid = np.flatnonzero(mask)
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))
        q = self.model(encode(view)[None], training=False).numpy()[0]
        q_masked = np.where(mask, q, -1e9)
        return int(np.argmax(q_masked))

    def q_values(self, view: np.ndarray) -> np.ndarray:
        """Rohe Q-Werte (für die Heatmap im Viewer)."""
        return self.model(encode(view)[None], training=False).numpy()[0]

    def remember(self, view, action, reward, next_view, done):
        self.memory.append((view, action, reward, next_view, done))

    def replay(self):
        if len(self.memory) < max(self.batch_size, self.cfg["WARMUP"]):
            return
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in idx]

        states = np.stack([encode(b[0]) for b in batch])
        next_states = np.stack([encode(b[3]) for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # ============================================================
        # TODO (Aufgabe 3): Kern des DQN-Trainingsschritts (TD-Target).
        #   1) Q-Werte der Folgezustände mit dem TARGET-Netz schätzen:
        #      q_next = self.target(next_states, training=False).numpy()
        #   2) Ungültige Folgeaktionen (bereits aufgedeckte Zellen) maskieren –
        #      nur verdeckte Zellen zählen. Nutze `next_masks` (siehe Hinweis)
        #      und setze ungültige Q-Werte auf einen sehr kleinen Wert (-1e9),
        #      z. B. mit np.where(next_masks, q_next, -1e9). Dann:
        #      max_next = np.max(q_next, axis=1)
        #   3) TD-Target bilden und nur für die ausgeführte Aktion einsetzen:
        #        target = reward + GAMMA * max_next * (1 - done)
        #      Starte von targets = self.model(states, ...).numpy() und
        #      überschreibe targets[arange(batch), actions] = ...
        #   4) Einen Gradientenschritt machen:
        #      self.model.train_on_batch(states, targets)
        # Hinweis (Maske der Folgezustände):
        #   next_masks = np.stack([(b[3].reshape(-1) == HIDDEN) for b in batch])
        #   Bei done==1 fällt der Zukunftsterm weg -> nur der Reward zählt.
        # ============================================================
        raise NotImplementedError("TODO: DQN-Trainingsschritt (TD-Target + train_on_batch) implementieren")

    def sync_target(self):
        self.target.set_weights(self.model.get_weights())

    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = self.keras.models.load_model(path)
        self.target = self.keras.models.load_model(path)


# ╔══════════════════════════════════════════════════════════╗
# ║                        TRAINING                         ║
# ╚══════════════════════════════════════════════════════════╝
def train(cfg: dict) -> DQNAgent:
    rng = np.random.default_rng(cfg["RANDOM_SEED"])
    np.random.seed(cfg["RANDOM_SEED"])
    import tensorflow as tf
    tf.random.set_seed(cfg["RANDOM_SEED"])

    env = Minesweeper(cfg, rng)
    agent = DQNAgent(cfg)

    rewards_hist, wins_hist = [], []
    win_window = deque(maxlen=100)

    print(f"[INFO] Training: {cfg['ROWS']}x{cfg['COLS']} Feld, "
          f"{cfg['N_MINES']} Minen, {cfg['EPISODES']} Episoden")
    for ep in range(1, cfg["EPISODES"] + 1):
        view = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(view)
            next_view, reward, done, info = env.step(action)
            agent.remember(view, action, reward, next_view, done)
            agent.replay()
            view = next_view
            total += reward

        agent.decay()
        if ep % cfg["TARGET_UPDATE"] == 0:
            agent.sync_target()

        rewards_hist.append(total)
        win = 1 if env.result == "win" else 0
        win_window.append(win)
        wins_hist.append(win)

        if ep % max(1, cfg["EPISODES"] // 20) == 0:
            print(f"  Episode {ep:4d} | Reward {total:6.2f} | "
                  f"ε {agent.epsilon:.3f} | Siegrate(100) {100*np.mean(win_window):5.1f}%")

    _plot_training(rewards_hist, wins_hist, cfg)
    agent.save(cfg["MODEL_PATH"])
    print(f"[INFO] Modell gespeichert: {cfg['MODEL_PATH']}")
    return agent


def _moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _plot_training(rewards, wins, cfg):
    os.makedirs(cfg["PLOT_DIR"], exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(rewards, alpha=0.3, label="Reward/Episode")
    ax[0].plot(_moving_average(rewards), color="red", label="gleitender Ø")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("kumulierter Reward")
    ax[0].set_title("Lernkurve"); ax[0].legend()

    ax[1].plot(100 * _moving_average(wins, 100), color="green")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Siegrate (%) – gleitend(100)")
    ax[1].set_title("Siegrate")
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "training_minesweeper.png")
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] Trainingskurve gespeichert: {out}")


# ╔══════════════════════════════════════════════════════════╗
# ║          TRAJEKTORIE EINES SPIELS (für den Viewer)     ║
# ╚══════════════════════════════════════════════════════════╝
def rollout(env: Minesweeper, agent: DQNAgent) -> list:
    """
    Spielt EIN Spiel mit der greedy-Policy und zeichnet jeden Zug auf.
    Rückgabe: Liste von Frames (Zustand je Schritt) für die Visualisierung.
    """
    view = env.reset()
    frames = [{
        "view": view.copy(), "cell": None, "reward": 0.0,
        "step": 0, "done": False, "result": None, "q": None,
    }]
    done = False
    step = 0
    while not done:
        q = agent.q_values(view)
        action = agent.act(view, greedy=True)
        next_view, reward, done, info = env.step(action)
        step += 1
        frames.append({
            "view": next_view.copy(),
            "cell": info.get("cell"),
            "reward": reward,
            "step": step,
            "done": done,
            "result": env.result,
            "q": q.reshape(env.rows, env.cols).copy(),
        })
        view = next_view
    return frames


# ╔══════════════════════════════════════════════════════════╗
# ║          INTERAKTIVER VIEWER (Weiter-Knopf)            ║
# ╚══════════════════════════════════════════════════════════╝
def interactive_viewer(cfg: dict, agent: DQNAgent):
    """Zeigt Spiele des Agenten Zug für Zug – steuerbar per Buttons."""
    from matplotlib.widgets import Button
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng()
    env = Minesweeper(cfg, rng)
    state = {"frames": rollout(env, agent), "i": 0, "mines": env.mines.copy()}

    # Farben für die Zahlen 1..8 (klassisches Minesweeper-Schema)
    num_colors = {1: "#1976D2", 2: "#388E3C", 3: "#D32F2F", 4: "#7B1FA2",
                  5: "#FF8F00", 6: "#0097A7", 7: "#424242", 8: "#9E9E9E"}

    fig, ax = plt.subplots(figsize=(1.1 * cfg["COLS"] + 1, 1.1 * cfg["ROWS"] + 1.4))
    plt.subplots_adjust(bottom=0.18, top=0.90)

    def draw():
        frames = state["frames"]
        i = state["i"]
        f = frames[i]
        view = f["view"]
        ax.clear()
        ax.set_xlim(0, cfg["COLS"]); ax.set_ylim(0, cfg["ROWS"])
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

        lost = (f["done"] and f["result"] == "loss")
        for r in range(cfg["ROWS"]):
            for c in range(cfg["COLS"]):
                x, y = c, r
                v = view[r, c]
                if v == HIDDEN:
                    # verdeckt – am Spielende Minen aufdecken
                    if (lost or (f["done"] and f["result"] == "win")) and state["mines"][r, c]:
                        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#FFCDD2", edgecolor="white"))
                        ax.text(x + 0.5, y + 0.5, "✷", ha="center", va="center", fontsize=16, color="#B71C1C")
                    else:
                        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#BDBDBD", edgecolor="white"))
                else:
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#FAFAFA", edgecolor="#CCCCCC"))
                    if v > 0:
                        ax.text(x + 0.5, y + 0.5, str(int(v)), ha="center", va="center",
                                fontsize=14, fontweight="bold", color=num_colors.get(int(v), "black"))

        # zuletzt gewählte Zelle hervorheben
        if f["cell"] is not None:
            cr, cc = f["cell"]
            edge = "#D32F2F" if lost else "#FFB300"
            ax.add_patch(Rectangle((cc, cr), 1, 1, fill=False, edgecolor=edge, linewidth=3.5))

        n_total = len(frames) - 1
        title = f"Zug {f['step']}/{n_total}"
        if f["done"]:
            title += "  –  " + ("✅ GEWONNEN" if f["result"] == "win" else "💥 MINE GETROFFEN")
        if f["cell"] is not None:
            title += f"   |   Reward: {f['reward']:+.2f}"
        ax.set_title(title, fontsize=12)
        fig.canvas.draw_idle()

    def on_next(_):
        if state["i"] < len(state["frames"]) - 1:
            state["i"] += 1
            draw()

    def on_prev(_):
        if state["i"] > 0:
            state["i"] -= 1
            draw()

    def on_new(_):
        env2 = Minesweeper(cfg, np.random.default_rng())
        state["frames"] = rollout(env2, agent)
        state["mines"] = env2.mines.copy()
        state["i"] = 0
        draw()

    ax_prev = plt.axes([0.13, 0.04, 0.18, 0.08])
    ax_next = plt.axes([0.41, 0.04, 0.18, 0.08])
    ax_new = plt.axes([0.69, 0.04, 0.18, 0.08])
    b_prev = Button(ax_prev, "◀ Zurück")
    b_next = Button(ax_next, "Weiter ▶")
    b_new = Button(ax_new, "Neues Spiel ⟳")
    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_new.on_clicked(on_new)

    draw()
    print("[INFO] Interaktiver Viewer: 'Weiter ▶' schaltet zum nächsten Zug.")
    plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║                          MAIN                           ║
# ╚══════════════════════════════════════════════════════════╝
def parse_args():
    p = argparse.ArgumentParser(description="RL-Workshop: Minesweeper mit DQN")
    p.add_argument("--train", action="store_true", help="Agenten trainieren")
    p.add_argument("--play", action="store_true", help="Interaktiver Viewer (Weiter-Knopf)")
    p.add_argument("--episodes", type=int, default=None, help="Anzahl Trainings-Episoden")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--mines", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    if args.episodes is not None: cfg["EPISODES"] = args.episodes
    if args.rows is not None: cfg["ROWS"] = args.rows
    if args.cols is not None: cfg["COLS"] = args.cols
    if args.mines is not None: cfg["N_MINES"] = args.mines

    if not args.train and not args.play:
        print("Nichts zu tun. Beispiele:")
        print("  python workshop_minesweeper.py --train")
        print("  python workshop_minesweeper.py --train --play")
        print("  python workshop_minesweeper.py --play   (benötigt gespeichertes Modell)")
        return

    agent = None
    if args.train:
        agent = train(cfg)

    if args.play:
        if agent is None:
            if not os.path.exists(cfg["MODEL_PATH"]):
                print(f"[FEHLER] Kein Modell unter {cfg['MODEL_PATH']}. Zuerst mit --train trainieren.")
                return
            agent = DQNAgent(cfg)
            agent.load(cfg["MODEL_PATH"])
            print(f"[INFO] Modell geladen: {cfg['MODEL_PATH']}")
        interactive_viewer(cfg, agent)


if __name__ == "__main__":
    main()
