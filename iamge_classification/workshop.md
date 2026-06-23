# Workshop: Bildklassifikation mit CNN (TensorFlow / Keras)

## Überblick

In diesem Workshop trainierst du ein **Convolutional Neural Network (CNN)** zur **Bildklassifikation** mit TensorFlow/Keras. Dabei durchläufst du den kompletten Machine-Learning-Workflow:

1. **Daten laden** – CIFAR-10 oder eigener Datensatz
2. **Datenaugmentation** – Bilder on-the-fly verändern
3. **Train/Val/Test Split** – saubere Datentrennung
4. **Modell bauen** – CNN-Architektur mit Keras
5. **Trainieren** – mit Early Stopping & LR-Scheduling
6. **Evaluieren** – Accuracy, Konfusionsmatrix, Diagramme
7. **Vorhersage** – Inferenz auf einzelnen Bildern
8. **Modell speichern & laden** – für spätere Nutzung

---

## Lernziele

- Aufbau und Funktionsweise eines **CNN** verstehen (Conv2D, MaxPooling, Flatten, Dense)
- **Datenaugmentation** als Regularisierungstechnik kennenlernen
- Einfluss von **Hyperparametern** auf Trainingsverlauf und Modellqualität erfahren
- Ein Modell **evaluieren** und Ergebnisse interpretieren (Accuracy, Konfusionsmatrix)
- Den vollständigen Workflow **von Rohdaten bis zur Vorhersage** selbstständig durchführen

---

## Voraussetzungen

- **Python 3.10+** (empfohlen)
- Folgende Pakete installieren:

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

> **Hinweis:** TensorFlow installiert Keras automatisch mit. Auf Apple Silicon (M1/M2/M3) wird `tensorflow-macos` und `tensorflow-metal` empfohlen.

---

## Datensatz

### Standard: CIFAR-10

Das Skript nutzt standardmäßig **CIFAR-10** – ein beliebter Benchmark-Datensatz mit:

- **60.000 Farbbilder** (32×32 Pixel, RGB)
- **10 Klassen**: Flugzeug, Auto, Vogel, Katze, Hirsch, Hund, Frosch, Pferd, Schiff, LKW
- Automatischer Download beim ersten Start

### Eigener Datensatz

Du kannst auch einen **eigenen Bilddatensatz** verwenden. Strukturiere dein Verzeichnis so:

```
mein_datensatz/
├── katze/
│   ├── bild001.jpg
│   ├── bild002.png
│   └── ...
├── hund/
│   ├── bild001.jpg
│   └── ...
└── vogel/
    └── ...
```

Aufruf mit eigenem Datensatz:
```bash
python workshop_image_classification.py --train --data-dir ./mein_datensatz/
```

---

## Ausführen

### Modell trainieren

```bash
# Standard (CIFAR-10, Standardparameter)
python workshop_image_classification.py --train

# Mit angepassten Hyperparametern
python workshop_image_classification.py --train --epochs 50 --batch-size 128 --learning-rate 0.0005

# Mit eigenem Datensatz
python workshop_image_classification.py --train --data-dir ./mein_datensatz/
```

### Vorhersage durchführen

```bash
# Zufälliges CIFAR-10-Testbild vorhersagen
python workshop_image_classification.py --predict

# Eigenes Bild vorhersagen
python workshop_image_classification.py --predict --image ./mein_foto.jpg
```

### Training + Vorhersage kombiniert

```bash
python workshop_image_classification.py --train --predict
```

---

## Alle CLI-Parameter

| Parameter          | Beschreibung                                    | Standard       |
|--------------------|-------------------------------------------------|----------------|
| `--train`          | Modell trainieren                               | –              |
| `--predict`        | Vorhersage durchführen                          | –              |
| `--data-dir`       | Pfad zu eigenem Bilddatensatz                   | CIFAR-10       |
| `--image`          | Bildpfad für `--predict`                        | Zufälliges Bild|
| `--epochs`         | Anzahl Trainings-Epochen                        | 30             |
| `--batch-size`     | Batch-Größe                                     | 64             |
| `--learning-rate`  | Lernrate des Optimizers                         | 0.001          |
| `--dropout`        | Dropout-Rate                                    | 0.4            |
| `--model-dir`      | Verzeichnis für gespeichertes Modell            | `saved_model/` |

---

## Hyperparameter zum Experimentieren

Im `CONFIG`-Block am Anfang des Skripts findest du alle Hyperparameter gesammelt. Hier sind die wichtigsten zum Ausprobieren:

### Architektur

| Parameter       | Beschreibung                          | Werte zum Testen        |
|-----------------|---------------------------------------|-------------------------|
| `CONV_FILTERS`  | Filter pro Conv-Block                 | `[32,64,128]`, `[64,128,256]` |
| `DENSE_UNITS`   | Neuronen in Dense-Layern              | `[256]`, `[512,256]`    |
| `DROPOUT`       | Dropout-Rate                          | 0.0–0.5                 |
| `BATCH_NORM`    | BatchNormalization ein/aus            | `True` / `False`        |
| `L2`            | L2-Regularisierung                    | 0, 1e-5, 1e-4, 1e-3    |

### Training

| Parameter             | Beschreibung                    | Werte zum Testen      |
|-----------------------|---------------------------------|-----------------------|
| `LEARNING_RATE`       | Lernrate                        | 1e-4, 5e-4, 1e-3     |
| `BATCH_SIZE`          | Minibatch-Größe                 | 32, 64, 128, 256      |
| `EPOCHS`              | Max. Trainings-Epochen          | 10, 30, 50, 100       |
| `PATIENCE`            | Early-Stopping-Geduld           | 5, 7, 15              |
| `REDUCE_LR_PATIENCE`  | Epochen bis LR-Reduktion        | 2, 3, 5               |

### Augmentation

| Parameter              | Beschreibung              | Werte zum Testen    |
|------------------------|---------------------------|---------------------|
| `AUG_FLIP_HORIZONTAL`  | Horizontal spiegeln       | `True` / `False`    |
| `AUG_ROTATION`         | Rotationsbereich          | 0.0, 0.05, 0.1     |
| `AUG_ZOOM`             | Zoombereich               | 0.0, 0.1, 0.2      |
| `AUG_CONTRAST`         | Kontrastjitter            | 0.0, 0.1, 0.2      |

---

## Aufgaben

### Aufgabe 1: Baseline trainieren
Starte das Skript mit den Standard-Einstellungen und notiere:
- Wie viele Epochen werden trainiert?
- Welche Test-Accuracy wird erreicht?
- Welche Klassen werden gut/schlecht erkannt?

### Aufgabe 2: Architektur variieren
Verändere die CNN-Architektur und vergleiche:
- Was passiert mit weniger Filtern (`CONV_FILTERS = [16, 32]`)?
- Was passiert mit mehr Filtern (`CONV_FILTERS = [64, 128, 256]`)?
- Was passiert ohne BatchNorm (`BATCH_NORM = False`)?

### Aufgabe 3: Regularisierung untersuchen
Experimentiere mit Regularisierungstechniken:
- Setze `DROPOUT = 0.0` – wird das Modell overfitted?
- Erhöhe `L2 = 1e-3` – sinkt die Accuracy?
- Schalte Augmentation aus (`AUG_ROTATION = 0`, `AUG_FLIP_HORIZONTAL = False`, etc.)

### Aufgabe 4: Lernrate und Batch-Größe
- Vergleiche `LEARNING_RATE = 1e-2` vs. `1e-4`
- Teste `BATCH_SIZE = 32` vs. `256`
- Beobachte den Loss-Verlauf in den generierten Plots

### Aufgabe 5: Eigenes Bild vorhersagen
- Lade ein Bild aus dem Internet (z.B. ein Auto, eine Katze)
- Führe `--predict --image mein_bild.jpg` aus
- Ist die Vorhersage korrekt? Wie hoch ist die Konfidenz?

### Bonus: Eigener Datensatz
- Sammle Bilder aus 3–5 Kategorien (je ~50+ Bilder)
- Strukturiere sie in Unterordnern (siehe oben)
- Trainiere mit `--data-dir` und bewerte das Ergebnis

---

## Ausgabe-Dateien

Nach dem Training findest du folgende Dateien:

```
saved_model/
├── cnn_classifier.keras    ← Gespeichertes Modell
└── class_names.txt         ← Klassenbezeichnungen

plots/
├── training_history.png    ← Loss- und Accuracy-Verlauf
├── confusion_matrix.png   ← Konfusionsmatrix als Heatmap
├── sample_predictions.png  ← Beispielvorhersagen (16 Bilder)
└── prediction.png          ← Einzelvorhersage (bei --predict)
```

---

## CNN-Architektur (Überblick)

```
Input (32×32×3)
    │
    ├── Conv2D (32 Filter) → BatchNorm → ReLU → MaxPool
    ├── Conv2D (64 Filter) → BatchNorm → ReLU → MaxPool
    ├── Conv2D (128 Filter) → BatchNorm → ReLU → MaxPool
    │
    ├── Flatten
    ├── Dense (256) → BatchNorm → ReLU → Dropout (0.4)
    │
    └── Dense (10, Softmax) → Klassenwsk.
```

---

## Tipps & Tricks

- **Overfitting erkennen:** Wenn der Val-Loss steigt, während der Train-Loss sinkt → mehr Regularisierung (Dropout, Augmentation, L2)
- **Underfitting erkennen:** Train- und Val-Loss sind beide hoch → größeres Modell, mehr Epochen, höhere Lernrate
- **GPU nutzen:** Falls verfügbar, beschleunigt eine GPU das Training erheblich. TensorFlow erkennt GPUs automatisch
- **Plots analysieren:** Die generierten Diagramme helfen beim Verstehen, was das Modell gelernt hat
- **Geduld:** CIFAR-10 ist für CNNs kein triviales Problem – 70–80 % Accuracy mit einem kleinen CNN sind ein gutes Ergebnis!

---

## Weiterführende Ideen

- **Transfer Learning:** Nutze ein vortrainiertes Modell (z.B. MobileNetV2) als Feature-Extraktor
- **Learning-Rate-Scheduler:** Implementiere einen Cosine-Annealing-Scheduler
- **Data Augmentation erweitern:** z.B. CutOut, MixUp, CutMix
- **Modellarchitektur:** Experimentiere mit ResNet-ähnlichen Skip-Connections

---

## Referenzen

- [TensorFlow/Keras Dokumentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [CIFAR-10 Datensatz](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CNN Explained (Stanford CS231n)](https://cs231n.github.io/convolutional-networks/)
