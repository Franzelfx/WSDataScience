# Workshop 02 · Bildklassifikation (CNN)

Trainiere ein Convolutional Neural Network (CNN) mit TensorFlow/Keras, das Farbbilder (CIFAR-10 oder ein eigener Datensatz) in Klassen einordnet – vom Laden der Daten über Augmentation und Training bis zur Vorhersage auf einzelnen Bildern.

## Lernziele
- Aufbau und Funktionsweise eines CNN verstehen (Conv2D, BatchNorm, MaxPooling, Flatten, Dense, Softmax)
- Datenaugmentation als Regularisierungstechnik kennenlernen und anwenden
- Ein Modell kompilieren, mit Early Stopping und LR-Scheduling trainieren
- Ergebnisse evaluieren und interpretieren (Accuracy, Konfusionsmatrix, Diagramme)
- Den vollständigen Workflow von Rohdaten bis zur Einzelvorhersage durchführen

## Struktur
| Pfad | Inhalt |
|------|--------|
| [`workshop.md`](workshop.md) | Ausführliche Anleitung & Erklärung |
| [`aufgabe/`](aufgabe/) | Starter-Code mit TODO-Lücken |
| [`loesung/`](loesung/) | Musterlösung |
| [`requirements.txt`](requirements.txt) | Abhängigkeiten |

## Setup
```bash
pip install -r requirements.txt
```

## Ausführen
```bash
# Modell trainieren (CIFAR-10, Standardparameter)
python aufgabe/workshop_image_classification.py --train

# Mit angepassten Hyperparametern
python aufgabe/workshop_image_classification.py --train --epochs 50 --batch-size 128 --learning-rate 0.0005

# Mit eigenem Datensatz (Unterordner pro Klasse)
python aufgabe/workshop_image_classification.py --train --data-dir ./mein_datensatz/

# Vorhersage auf einem zufälligen CIFAR-10-Testbild
python aufgabe/workshop_image_classification.py --predict

# Vorhersage auf einem eigenen Bild
python aufgabe/workshop_image_classification.py --predict --image ./mein_foto.jpg
```

> Die Musterlösung lässt sich identisch aufrufen, z. B.
> `python loesung/workshop_image_classification.py --train`.

[← Zur Workshop-Übersicht](../../README.md)
