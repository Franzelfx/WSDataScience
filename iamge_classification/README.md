# Workshop: Bildklassifikation mit CNN (TensorFlow / Keras)

## Überblick

In diesem Workshop trainierst du ein **Convolutional Neural Network (CNN)** zur
**Bildklassifikation** und durchläufst den kompletten Machine-Learning-Workflow –
von den Rohdaten bis zur Vorhersage auf einem einzelnen Bild. Als Datensatz dient
standardmäßig **CIFAR-10** (10 Klassen, 32×32-Farbbilder); alternativ lässt sich
ein eigener Bilddatensatz verwenden.

Die ausführliche Anleitung mit allen Aufgaben steht in **[workshop.md](workshop.md)**.

---

## Lernziele

- Aufbau und Funktionsweise eines **CNN** verstehen (Conv2D, MaxPooling, Flatten, Dense)
- **Datenaugmentation** als Regularisierungstechnik einsetzen
- Einfluss von **Hyperparametern** auf Trainingsverlauf und Modellqualität erfahren
- Ein Modell **evaluieren** (Accuracy, Konfusionsmatrix) und Ergebnisse interpretieren
- Den vollständigen Workflow **von Rohdaten bis zur Vorhersage** durchführen

---

## Voraussetzungen

- **Python 3.10+**

```bash
pip install tensorflow scikit-learn matplotlib numpy
```

> Auf Apple Silicon (M1/M2/M3) werden `tensorflow-macos` und `tensorflow-metal`
> empfohlen. CIFAR-10 wird beim ersten Start automatisch heruntergeladen.

---

## Dateien

| Datei                                                                  | Beschreibung                          |
|------------------------------------------------------------------------|---------------------------------------|
| [`workshop.md`](workshop.md)                                           | Vollständige Workshop-Anleitung       |
| [`workshop_image_classification.py`](workshop_image_classification.py) | Starter-Code mit zentralem `CONFIG`   |

---

## Ausführen

```bash
cd iamge_classification

# Modell trainieren (CIFAR-10, Standardparameter)
python workshop_image_classification.py --train

# Mit angepassten Hyperparametern
python workshop_image_classification.py --train --epochs 50 --batch-size 128 --learning-rate 0.0005

# Vorhersage auf einem eigenen Bild
python workshop_image_classification.py --predict --image ./mein_foto.jpg

# Eigener Bilddatensatz (Unterordner = Klassen)
python workshop_image_classification.py --train --data-dir ./mein_datensatz/
```

Die vollständige Parameterliste und alle Aufgaben findest du in [workshop.md](workshop.md).

---

## Ausgabe-Dateien

```
saved_model/                 plots/
├── cnn_classifier.keras     ├── training_history.png
└── class_names.txt          ├── confusion_matrix.png
                             ├── sample_predictions.png
                             └── prediction.png
```

> Hinweis: `saved_model/` und `plots/` werden über die `.gitignore` des Repos nicht eingecheckt.

---

## Weiterführende Workshops

[Explorative Datenanalyse](../explorative_datenanalyse/) ·
[Autoencoder](../autoencoder/) ·
[Reinforcement Learning](../reinforcement_learning/)
