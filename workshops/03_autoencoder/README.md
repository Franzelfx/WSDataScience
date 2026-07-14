# Workshop 03 · Autoencoder (Unüberwachtes Lernen / Anomalien)

In diesem Workshop baust und trainierst du einen Autoencoder mit TensorFlow/Keras, der Muster in tabellarischen Daten lernt. Über den Rekonstruktionsfehler leitest du einen Schwellwert ab und erkennst so Anomalien – ganz ohne Labels.

## Lernziele
- Unterschied zwischen überwachtem und unüberwachtem Lernen verstehen (Rekonstruktion statt Klassifikation).
- Daten vorbereiten: Train/Test-Split und Standardisierung.
- Encoder-/Decoder-Architektur eines Autoencoders in Keras aufbauen und trainieren.
- Anomalien über Rekonstruktionsfehler und Quantil-Schwellwert erkennen.
- Ergebnisse visualisieren und interpretieren: Loss-Verlauf, Fehler-Histogramm, Latent-Space.

## Struktur
| Pfad | Inhalt |
|------|--------|
| [`workshop.md`](workshop.md) | Ausführliche Anleitung & Erklärung |
| [`aufgabe/`](aufgabe/) | Starter-Code mit TODO-Lücken |
| [`loesung/`](loesung/) | Musterlösung |
| [`data/`](data/) | Beispieldaten (iris_fallback.csv) |
| [`requirements.txt`](requirements.txt) | Abhängigkeiten |
| [`plot_interpretation.md`](plot_interpretation.md) | Wie man die erzeugten Plots interpretiert |
| [`modellvergleich.md`](modellvergleich.md) | Autoencoder vs. VAE vs. Diffusion Model |

## Setup
```bash
pip install -r requirements.txt
```

## Ausführen
Standard (Iris-Fallback aus `data/iris_fallback.csv`):
```bash
python aufgabe/workshop_autoencoder.py       # Starter-Code mit TODOs
python loesung/workshop_autoencoder.py       # Musterlösung
```

Eigenen CSV-Datensatz nutzen (nicht-numerische Spalten werden verworfen):
```bash
python loesung/workshop_autoencoder.py --data pfad/zu/deinen_daten.csv
```

[← Zur Workshop-Übersicht](../../README.md)
