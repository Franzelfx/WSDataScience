#!/usr/bin/env python3
"""
Workshop: Bildklassifikation mit TensorFlow / Keras (CNN)
=========================================================
Dieses Skript demonstriert den vollständigen Workflow einer
Bildklassifikation mit einem Convolutional Neural Network (CNN):

  1. Daten laden (CIFAR-10 oder eigenes Bildverzeichnis)
  2. Datenaugmentation
  3. Train / Validation / Test Split
  4. Modellaufbau (CNN-Architektur)
  5. Training mit Callbacks
  6. Evaluation & Visualisierung
  7. Inferenz auf einzelnen Bildern
  8. Modell speichern & laden

Aufruf:
  python workshop_image_classification.py --train
  python workshop_image_classification.py --predict [BILDPFAD]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")            # << Nicht-interaktives Backend (Server-tauglich)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# ╔══════════════════════════════════════════════════════════╗
# ║               HYPERPARAMETER-KONFIGURATION              ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Hier kannst du alle wichtigen Parameter zentral        ║
# ║  anpassen und experimentieren.                          ║
# ╚══════════════════════════════════════════════════════════╝
CONFIG = {
    # ── Daten ──────────────────────────────────────────────
    "IMG_HEIGHT": 32,                 # Bildhöhe in Pixel
    "IMG_WIDTH": 32,                  # Bildbreite in Pixel
    "NUM_CHANNELS": 3,                # Farbkanäle (3 = RGB)
    "NUM_CLASSES": 10,                # Anzahl Klassen
    "VALIDATION_SPLIT": 0.2,          # Anteil Validierungsdaten
    "RANDOM_SEED": 42,

    # ── Augmentation ──────────────────────────────────────
    "AUG_FLIP_HORIZONTAL": True,      # << Horizontal spiegeln
    "AUG_ROTATION": 0.05,             # << Rotation (Anteil, z.B. 0.05 = ±18°)
    "AUG_ZOOM": 0.1,                  # << Zoom-Bereich (z.B. 0.1 = ±10 %)
    "AUG_CONTRAST": 0.1,              # << Kontrastjitter

    # ── Modellarchitektur ─────────────────────────────────
    "CONV_FILTERS": [32, 64, 128],    # << Filter pro Conv-Block
    "CONV_KERNEL": 3,                 # << Kernelgröße
    "DENSE_UNITS": [256],             # << Dense-Layer nach Flatten
    "DROPOUT": 0.4,                   # << Dropout-Rate (0.0–0.5)
    "L2": 1e-4,                       # << L2-Regularisierung
    "BATCH_NORM": True,               # << BatchNormalization nutzen?

    # ── Training ──────────────────────────────────────────
    "LEARNING_RATE": 1e-3,            # << 1e-4 bis 1e-2
    "BATCH_SIZE": 64,                 # << 32, 64, 128 …
    "EPOCHS": 30,                     # << Trainings-Epochen
    "PATIENCE": 7,                    # << Early-Stopping-Patience
    "REDUCE_LR_PATIENCE": 3,          # << LR-Reduktion nach N Epochen ohne Verbesserung
    "REDUCE_LR_FACTOR": 0.5,          # << LR-Multiplikator

    # ── Pfade ─────────────────────────────────────────────
    "MODEL_DIR": "saved_model",       # Ordner für gespeichertes Modell
    "PLOT_DIR": "plots",              # Ordner für Diagramme
}

# CIFAR-10 Klassennamen (Index = Label)
CIFAR10_CLASSES = [
    "Flugzeug", "Auto", "Vogel", "Katze", "Hirsch",
    "Hund", "Frosch", "Pferd", "Schiff", "LKW",
]


# ╔══════════════════════════════════════════════════════════╗
# ║                     HILFSFUNKTIONEN                     ║
# ╚══════════════════════════════════════════════════════════╝

def set_seeds(seed: int) -> None:
    """Reproduzierbare Ergebnisse durch fixierte Zufallsseeds."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"[SEEDS]  Random-Seed gesetzt: {seed}")


def print_section(title: str) -> None:
    """Gibt eine formatierte Abschnittsüberschrift aus."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def ensure_dir(path: str) -> None:
    """Erstellt ein Verzeichnis, falls es nicht existiert."""
    os.makedirs(path, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║              1. DATEN LADEN (DATA INGESTION)            ║
# ╚══════════════════════════════════════════════════════════╝

def load_cifar10() -> tuple:
    """
    Lädt CIFAR-10 über Keras und gibt Train-/Test-Daten zurück.
    Pixel werden auf [0, 1] normalisiert.
    """
    print_section("1. DATEN LADEN – CIFAR-10")

    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalisierung auf [0, 1]
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Labels von (N, 1) auf (N,) flatten
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    print(f"[DATEN]  Trainingsbilder gesamt : {X_train_full.shape}")
    print(f"[DATEN]  Testbilder             : {X_test.shape}")
    print(f"[DATEN]  Klassen                : {CONFIG['NUM_CLASSES']}")
    print(f"[DATEN]  Klassenbezeichnungen   : {CIFAR10_CLASSES}")
    print(f"[DATEN]  Pixelbereich           : [{X_train_full.min():.1f}, {X_train_full.max():.1f}]")

    return X_train_full, y_train_full, X_test, y_test, CIFAR10_CLASSES


def load_custom_dataset(data_dir: str) -> tuple:
    """
    Lädt Bilder aus einem Verzeichnis mit Unterordnern pro Klasse:
      data_dir/
        klasse_a/
          bild1.jpg
          bild2.png
        klasse_b/
          ...
    """
    print_section("1. DATEN LADEN – Eigener Datensatz")

    img_size = (CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"])

    full_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=None,           # Einzelne Bilder, kein Batching
        label_mode="int",
        shuffle=True,
        seed=CONFIG["RANDOM_SEED"],
    )

    class_names = full_ds.class_names
    CONFIG["NUM_CLASSES"] = len(class_names)

    # In NumPy-Arrays konvertieren
    images, labels = [], []
    for img, lbl in full_ds:
        images.append(img.numpy())
        labels.append(lbl.numpy())

    X_all = np.array(images, dtype="float32") / 255.0
    y_all = np.array(labels)

    # Train / Test Split
    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all,
        test_size=CONFIG["VALIDATION_SPLIT"],
        stratify=y_all,
        random_state=CONFIG["RANDOM_SEED"],
    )

    print(f"[DATEN]  Verzeichnis            : {data_dir}")
    print(f"[DATEN]  Klassen ({len(class_names)})          : {class_names}")
    print(f"[DATEN]  Trainingsbilder        : {X_train_full.shape}")
    print(f"[DATEN]  Testbilder             : {X_test.shape}")

    return X_train_full, y_train_full, X_test, y_test, class_names


# ╔══════════════════════════════════════════════════════════╗
# ║           2. DATENAUGMENTATION (AUGMENTATION)           ║
# ╚══════════════════════════════════════════════════════════╝

def build_augmentation_pipeline() -> keras.Sequential:
    """
    Erstellt eine Keras-Augmentation-Pipeline als Sequential-Layer.
    Diese wird nur beim Training angewendet (training=True).
    """
    print_section("2. DATENAUGMENTATION")

    aug_layers = []

    if CONFIG["AUG_FLIP_HORIZONTAL"]:
        aug_layers.append(layers.RandomFlip("horizontal"))
        print("[AUG]    RandomFlip(horizontal)")

    if CONFIG["AUG_ROTATION"] > 0:
        aug_layers.append(layers.RandomRotation(CONFIG["AUG_ROTATION"]))
        print(f"[AUG]    RandomRotation({CONFIG['AUG_ROTATION']})")

    if CONFIG["AUG_ZOOM"] > 0:
        aug_layers.append(layers.RandomZoom(CONFIG["AUG_ZOOM"]))
        print(f"[AUG]    RandomZoom({CONFIG['AUG_ZOOM']})")

    if CONFIG["AUG_CONTRAST"] > 0:
        aug_layers.append(layers.RandomContrast(CONFIG["AUG_CONTRAST"]))
        print(f"[AUG]    RandomContrast({CONFIG['AUG_CONTRAST']})")

    if not aug_layers:
        print("[AUG]    Keine Augmentation konfiguriert.")
        return None

    pipeline = keras.Sequential(aug_layers, name="augmentation")
    print(f"[AUG]    Pipeline: {len(aug_layers)} Stufen aktiv")
    return pipeline


# ╔══════════════════════════════════════════════════════════╗
# ║           3. TRAIN / VALIDATION / TEST SPLIT            ║
# ╚══════════════════════════════════════════════════════════╝

def split_train_val(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
) -> tuple:
    """
    Teilt die Trainingsdaten in Training und Validierung auf.
    """
    print_section("3. TRAIN / VALIDATION SPLIT")

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=CONFIG["VALIDATION_SPLIT"],
        stratify=y_train_full,
        random_state=CONFIG["RANDOM_SEED"],
    )

    print(f"[SPLIT]  Training    : {X_train.shape[0]:>6} Bilder  ({100*(1-CONFIG['VALIDATION_SPLIT']):.0f} %)")
    print(f"[SPLIT]  Validierung : {X_val.shape[0]:>6} Bilder  ({100*CONFIG['VALIDATION_SPLIT']:.0f} %)")

    # Klassenverteilung anzeigen
    unique, counts = np.unique(y_train, return_counts=True)
    print("[SPLIT]  Klassenverteilung (Training):")
    for cls_id, cnt in zip(unique, counts):
        print(f"         Klasse {cls_id:>2}: {cnt:>5} Bilder")

    return X_train, X_val, y_train, y_val


# ╔══════════════════════════════════════════════════════════╗
# ║            4. MODELL BAUEN (MODEL PREPARATION)          ║
# ╚══════════════════════════════════════════════════════════╝

def build_cnn_model(
    input_shape: tuple,
    num_classes: int,
    augmentation: keras.Sequential | None = None,
) -> keras.Model:
    """
    Baut ein CNN-Klassifikationsmodell mit konfigurierbarer Architektur.

    Architektur:
      [Augmentation] → Conv-Blöcke → Flatten → Dense → Softmax
    """
    print_section("4. MODELL BAUEN")

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # Augmentation (nur beim Training aktiv)
    if augmentation is not None:
        x = augmentation(x)

    # ── Convolutional Blöcke ──────────────────────────────
    for i, filters in enumerate(CONFIG["CONV_FILTERS"]):
        x = layers.Conv2D(
            filters,
            CONFIG["CONV_KERNEL"],
            padding="same",
            kernel_regularizer=regularizers.l2(CONFIG["L2"]),
            name=f"conv_{i+1}",
        )(x)

        if CONFIG["BATCH_NORM"]:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"pool_{i+1}")(x)

    # ── Übergang zu Dense ─────────────────────────────────
    x = layers.Flatten(name="flatten")(x)

    for j, units in enumerate(CONFIG["DENSE_UNITS"]):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(CONFIG["L2"]),
            name=f"dense_{j+1}",
        )(x)

        if CONFIG["BATCH_NORM"]:
            x = layers.BatchNormalization(name=f"bn_dense_{j+1}")(x)

        x = layers.Activation("relu", name=f"relu_dense_{j+1}")(x)
        x = layers.Dropout(CONFIG["DROPOUT"], name=f"dropout_{j+1}")(x)

    # ── Ausgabeschicht ────────────────────────────────────
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")

    # Kompilieren
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Zusammenfassung
    model.summary()

    total_params = model.count_params()
    print(f"\n[MODELL] Gesamtparameter    : {total_params:,}")
    print(f"[MODELL] Conv-Blöcke        : {len(CONFIG['CONV_FILTERS'])}")
    print(f"[MODELL] Dense-Layer        : {len(CONFIG['DENSE_UNITS'])}")
    print(f"[MODELL] Dropout            : {CONFIG['DROPOUT']}")
    print(f"[MODELL] BatchNorm          : {'Ja' if CONFIG['BATCH_NORM'] else 'Nein'}")
    print(f"[MODELL] L2-Regularisierung : {CONFIG['L2']}")

    return model


# ╔══════════════════════════════════════════════════════════╗
# ║                    5. TRAINING                          ║
# ╚══════════════════════════════════════════════════════════╝

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> keras.callbacks.History:
    """
    Trainiert das Modell mit Early Stopping und LR-Reduktion.
    """
    print_section("5. TRAINING")

    print(f"[TRAIN]  Epochen           : {CONFIG['EPOCHS']}")
    print(f"[TRAIN]  Batch-Größe       : {CONFIG['BATCH_SIZE']}")
    print(f"[TRAIN]  Lernrate          : {CONFIG['LEARNING_RATE']}")
    print(f"[TRAIN]  Early-Stop-Geduld : {CONFIG['PATIENCE']}")
    print(f"[TRAIN]  LR-Reduktion nach : {CONFIG['REDUCE_LR_PATIENCE']} Epochen")
    print()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["PATIENCE"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG["REDUCE_LR_FACTOR"],
            patience=CONFIG["REDUCE_LR_PATIENCE"],
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],
        callbacks=cb_list,
        verbose=1,
    )

    print(f"\n[TRAIN]  Training beendet nach {len(history.history['loss'])} Epochen")
    print(f"[TRAIN]  Bester Val-Loss   : {min(history.history['val_loss']):.4f}")
    print(f"[TRAIN]  Beste Val-Accuracy: {max(history.history['val_accuracy']):.4f}")

    return history


# ╔══════════════════════════════════════════════════════════╗
# ║          6. EVALUATION & VISUALISIERUNG                 ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    history: keras.callbacks.History | None = None,
) -> None:
    """
    Evaluiert das Modell auf den Testdaten und erstellt Diagramme.
    """
    print_section("6. EVALUATION & VISUALISIERUNG")

    # ── Test-Metriken ─────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[EVAL]   Test-Loss     : {test_loss:.4f}")
    print(f"[EVAL]   Test-Accuracy : {test_acc:.4f}  ({test_acc*100:.1f} %)")

    # ── Klassenweise Accuracy ─────────────────────────────
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n[EVAL]   Klassenweise Ergebnisse:")
    print(f"         {'Klasse':<15} {'Richtig':>8} {'Gesamt':>8} {'Accuracy':>10}")
    print("         " + "-" * 45)

    for cls_id in range(len(class_names)):
        mask = y_test == cls_id
        total = mask.sum()
        correct = (y_pred_classes[mask] == cls_id).sum()
        acc = correct / total if total > 0 else 0.0
        name = class_names[cls_id] if cls_id < len(class_names) else f"Klasse {cls_id}"
        print(f"         {name:<15} {correct:>8} {total:>8} {acc:>9.1%}")

    # ── Konfusionsmatrix (Text) ───────────────────────────
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred_classes)
    print("\n[EVAL]   Konfusionsmatrix:")
    header = "         " + "".join(f"{name[:5]:>6}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        name = class_names[i][:5] if i < len(class_names) else f"C{i}"
        print(f"  {name:>5} " + "".join(f"{v:>6}" for v in row))

    print("\n[EVAL]   Classification Report:")
    target_names = [str(c) for c in class_names]
    print(classification_report(y_test, y_pred_classes, target_names=target_names))

    # ── Diagramme ─────────────────────────────────────────
    ensure_dir(CONFIG["PLOT_DIR"])

    if history is not None:
        _plot_training_history(history)

    _plot_confusion_matrix(cm, class_names)
    _plot_sample_predictions(X_test, y_test, y_pred_classes, class_names)

    print(f"[EVAL]   Diagramme gespeichert in: {CONFIG['PLOT_DIR']}/")


def _plot_training_history(history: keras.callbacks.History) -> None:
    """Speichert Verlauf von Loss und Accuracy als Plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train-Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val-Loss", linewidth=2)
    ax1.set_title("Verlust (Loss) über Epochen")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train-Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Val-Accuracy", linewidth=2)
    ax2.set_title("Genauigkeit (Accuracy) über Epochen")
    ax2.set_xlabel("Epoche")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CONFIG["PLOT_DIR"], "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]   Training-History  → {path}")


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Speichert die Konfusionsmatrix als Heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Konfusionsmatrix")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Werte in Zellen schreiben
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    ax.set_ylabel("Wahres Label")
    ax.set_xlabel("Vorhergesagtes Label")
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOT_DIR"], "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]   Konfusionsmatrix → {path}")


def _plot_sample_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    n: int = 16,
) -> None:
    """Zeigt N zufällige Testbilder mit Vorhersage und wahrem Label."""
    indices = np.random.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        img = X_test[idx]
        true_label = class_names[y_test[idx]] if y_test[idx] < len(class_names) else str(y_test[idx])
        pred_label = class_names[y_pred[idx]] if y_pred[idx] < len(class_names) else str(y_pred[idx])
        correct = y_test[idx] == y_pred[idx]

        ax.imshow(img)
        color = "green" if correct else "red"
        ax.set_title(f"W: {true_label}\nV: {pred_label}", fontsize=9, color=color)
        ax.axis("off")

    # Übrige Achsen ausblenden
    for i in range(len(indices), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Beispiel-Vorhersagen (grün=richtig, rot=falsch)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOT_DIR"], "sample_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]   Beispielbilder   → {path}")


# ╔══════════════════════════════════════════════════════════╗
# ║             7. MODELL SPEICHERN & LADEN                 ║
# ╚══════════════════════════════════════════════════════════╝

def save_model(model: keras.Model, model_dir: str) -> None:
    """Speichert das gesamte Modell (Architektur + Gewichte + Optimizer)."""
    print_section("MODELL SPEICHERN")
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "cnn_classifier.keras")
    model.save(model_path)
    print(f"[SAVE]   Modell gespeichert: {model_path}")

    # Auch Klassennamen speichern
    classes_path = os.path.join(model_dir, "class_names.txt")
    # class_names werden über globale Variable übergeben
    print(f"[SAVE]   Modellverzeichnis: {model_dir}")


def load_saved_model(model_dir: str) -> keras.Model:
    """Lädt ein gespeichertes Modell."""
    model_path = os.path.join(model_dir, "cnn_classifier.keras")
    if not os.path.exists(model_path):
        print(f"[FEHLER] Kein Modell gefunden unter: {model_path}")
        print("         Bitte zuerst mit --train trainieren!")
        sys.exit(1)

    print(f"[LOAD]   Lade Modell aus: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"[LOAD]   Modell erfolgreich geladen.")
    return model


# ╔══════════════════════════════════════════════════════════╗
# ║               8. INFERENZ (PREDICTION)                  ║
# ╚══════════════════════════════════════════════════════════╝

def predict_single_image(
    model: keras.Model,
    image_path: str | None,
    class_names: list[str],
) -> None:
    """
    Führt eine Vorhersage auf einem einzelnen Bild durch.
    Falls kein Pfad angegeben → zufälliges CIFAR-10 Testbild.
    """
    print_section("8. INFERENZ / VORHERSAGE")

    if image_path and os.path.exists(image_path):
        # Externes Bild laden
        print(f"[PRED]   Lade Bild: {image_path}")
        img = keras.utils.load_img(
            image_path,
            target_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
        )
        img_array = keras.utils.img_to_array(img) / 255.0
        true_label = None
    else:
        # Zufälliges CIFAR-10-Testbild
        if image_path:
            print(f"[WARN]   Bild nicht gefunden: {image_path}")
        print("[PRED]   Verwende zufälliges CIFAR-10-Testbild …")
        (_, _), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_test = X_test.astype("float32") / 255.0
        y_test = y_test.flatten()

        idx = np.random.randint(0, len(X_test))
        img_array = X_test[idx]
        true_label = class_names[y_test[idx]] if y_test[idx] < len(class_names) else str(y_test[idx])

    # Vorhersage
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    predicted_name = class_names[predicted_class] if predicted_class < len(class_names) else str(predicted_class)

    print(f"\n[PRED]   ┌──────────────────────────────────────┐")
    print(f"[PRED]   │  Vorhersage: {predicted_name:<24}│")
    print(f"[PRED]   │  Konfidenz : {confidence:.1%}{' ' * (24 - len(f'{confidence:.1%}'))}│")
    if true_label:
        correct = "✓" if predicted_name == true_label else "✗"
        print(f"[PRED]   │  Wahrheit  : {true_label:<24}│")
        print(f"[PRED]   │  Korrekt   : {correct:<24}│")
    print(f"[PRED]   └──────────────────────────────────────┘")

    # Top-5 Wahrscheinlichkeiten
    top5_idx = np.argsort(predictions)[::-1][:5]
    print("\n[PRED]   Top-5 Klassen:")
    for rank, idx in enumerate(top5_idx, 1):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        bar = "█" * int(predictions[idx] * 30)
        print(f"         {rank}. {name:<12} {predictions[idx]:>6.1%} {bar}")

    # Bild speichern
    ensure_dir(CONFIG["PLOT_DIR"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array)
    title = f"Vorhersage: {predicted_name} ({confidence:.1%})"
    if true_label:
        title += f"\nWahrheit: {true_label}"
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOT_DIR"], "prediction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[PLOT]   Vorhersage-Bild → {path}")


# ╔══════════════════════════════════════════════════════════╗
# ║                       MAIN                              ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Workshop: Bildklassifikation mit CNN (TensorFlow/Keras)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python workshop_image_classification.py --train
  python workshop_image_classification.py --train --epochs 50 --batch-size 128
  python workshop_image_classification.py --train --data-dir ./mein_datensatz/
  python workshop_image_classification.py --predict
  python workshop_image_classification.py --predict --image mein_bild.jpg
        """,
    )

    # Modi
    parser.add_argument("--train", action="store_true", help="Modell trainieren")
    parser.add_argument("--predict", action="store_true", help="Vorhersage durchführen")

    # Daten
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Pfad zu eigenem Bild-Datensatz (Unterordner pro Klasse)")
    parser.add_argument("--image", type=str, default=None,
                        help="Pfad zu einem Bild für --predict")

    # Hyperparameter-Overrides
    parser.add_argument("--epochs", type=int, default=None, help="Anzahl Trainings-Epochen")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch-Größe")
    parser.add_argument("--learning-rate", type=float, default=None, help="Lernrate")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout-Rate")

    # Pfade
    parser.add_argument("--model-dir", type=str, default=None, help="Verzeichnis für Modell")

    args = parser.parse_args()

    # Keine Aktion angegeben
    if not args.train and not args.predict:
        parser.print_help()
        print("\n[FEHLER] Bitte --train und/oder --predict angeben!")
        sys.exit(1)

    # Hyperparameter-Overrides anwenden
    if args.epochs is not None:
        CONFIG["EPOCHS"] = args.epochs
    if args.batch_size is not None:
        CONFIG["BATCH_SIZE"] = args.batch_size
    if args.learning_rate is not None:
        CONFIG["LEARNING_RATE"] = args.learning_rate
    if args.dropout is not None:
        CONFIG["DROPOUT"] = args.dropout
    if args.model_dir is not None:
        CONFIG["MODEL_DIR"] = args.model_dir

    # ── Seeds setzen ──────────────────────────────────────
    set_seeds(CONFIG["RANDOM_SEED"])

    # TensorFlow Info
    print(f"[INFO]   TensorFlow Version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO]   GPUs verfügbar     : {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(f"         → {gpu.name}")

    # ══════════════════════════════════════════════════════
    #  TRAINING-MODUS
    # ══════════════════════════════════════════════════════
    if args.train:
        # 1. Daten laden
        if args.data_dir:
            X_train_full, y_train_full, X_test, y_test, class_names = load_custom_dataset(args.data_dir)
        else:
            X_train_full, y_train_full, X_test, y_test, class_names = load_cifar10()

        # 2. Augmentation
        augmentation = build_augmentation_pipeline()

        # 3. Train/Val Split
        X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)

        # 4. Modell bauen
        input_shape = (CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"], CONFIG["NUM_CHANNELS"])
        model = build_cnn_model(input_shape, CONFIG["NUM_CLASSES"], augmentation)

        # 5. Training
        history = train_model(model, X_train, y_train, X_val, y_val)

        # 6. Evaluation
        evaluate_model(model, X_test, y_test, class_names, history)

        # 7. Modell speichern
        save_model(model, CONFIG["MODEL_DIR"])

        # Klassennamen speichern
        classes_path = os.path.join(CONFIG["MODEL_DIR"], "class_names.txt")
        with open(classes_path, "w") as f:
            for name in class_names:
                f.write(name + "\n")
        print(f"[SAVE]   Klassennamen → {classes_path}")

    # ══════════════════════════════════════════════════════
    #  VORHERSAGE-MODUS
    # ══════════════════════════════════════════════════════
    if args.predict:
        # Modell laden
        model = load_saved_model(CONFIG["MODEL_DIR"])

        # Klassennamen laden
        classes_path = os.path.join(CONFIG["MODEL_DIR"], "class_names.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = CIFAR10_CLASSES

        predict_single_image(model, args.image, class_names)

    print_section("FERTIG")
    print("[DONE]   Workshop-Skript abgeschlossen. Viel Spass beim Experimentieren!")


if __name__ == "__main__":
    main()
