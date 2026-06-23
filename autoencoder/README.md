# Workshop: Unüberwachtes Lernen mit Autoencoder (TensorFlow / Keras)

## Überblick

In diesem Workshop trainierst du einen **Autoencoder** – ein neuronales Netz, das
lernt, seine Eingabe zu **komprimieren** (Encoder) und wieder zu **rekonstruieren**
(Decoder). Da keine Labels nötig sind, ist das **unüberwachtes Lernen**. Über den
**Rekonstruktionsfehler** lassen sich anschließend **Anomalien** erkennen.

Als Datensatz dient standardmäßig der **Iris**-Datensatz (numerisch, als Fallback
in `iris_fallback.csv` gespeichert); alternativ kann eine eigene CSV genutzt werden.

Die ausführliche Anleitung steht in **[workshop.md](workshop.md)**.

---

## Lernziele

- Den Unterschied zwischen **überwachtem** und **unüberwachtem** Lernen verstehen
  (Rekonstruktion statt Klassifikation)
- Datenvorbereitung: **Train/Test-Split** und **Skalierung**
- Die Architektur eines **Autoencoders** (Encoder · Latent Space · Decoder) in Keras aufbauen
- Evaluation über **Rekonstruktionsfehler** und **Schwellwert** (Quantil) zur Anomalie-Detektion
- Visualisierung von Loss-Verlauf, Fehler-Histogramm und Latent-Space
- Optional: **Denoising-Autoencoder** und Clustering im Latent Space

---

## Voraussetzungen

- **Python 3.9+**

```bash
pip install tensorflow scikit-learn pandas matplotlib numpy
```

---

## Dateien

| Datei                                                | Beschreibung                                              |
|------------------------------------------------------|----------------------------------------------------------|
| [`workshop.md`](workshop.md)                         | Vollständige Workshop-Anleitung                          |
| [`workshop_autoencoder.py`](workshop_autoencoder.py) | Starter-Code mit zentralem `CONFIG`-Block                |
| [`plot_interpretation.md`](plot_interpretation.md)   | Hilfestellung: Wie interpretiere ich die erzeugten Plots? |
| [`modellvergleich.md`](modellvergleich.md)           | Hintergrund: Autoencoder vs. VAE vs. Diffusion Model      |
| [`workshop.pdf`](workshop.pdf)                       | Anleitung als PDF                                        |
| `iris_fallback.csv`                                  | Fallback-Datensatz (wird bei Bedarf erzeugt)             |

---

## Ausführen

```bash
cd autoencoder

# Standard (Iris-Fallback)
python workshop_autoencoder.py

# Eigene CSV nutzen (nicht-numerische Spalten werden verworfen)
python workshop_autoencoder.py --data data/workshop_data.csv
```

Die wichtigsten Hyperparameter (`LATENT_DIM`, `HIDDEN_DIMS`, `LEARNING_RATE`,
`DROPOUT`, `L2`, `THRESHOLD_QUANTILE`, `NOISE_FACTOR` …) stehen gesammelt im
`CONFIG`-Block oben im Skript. Alle Aufgaben findest du in [workshop.md](workshop.md).

---

## Was passiert beim Ausführen?

1. Daten laden (Iris-Fallback oder eigene CSV) und skalieren
2. Autoencoder bauen und trainieren (mit Early Stopping)
3. **Rekonstruktionsfehler** auf Train/Test berechnen
4. **Schwellwert** aus einem Quantil ableiten → Anomalie-Flag
5. Plots: Loss-Verlauf, Fehler-Histogramm, Latent-Space (Farbe = Fehler)

Zur Deutung der Diagramme siehe [plot_interpretation.md](plot_interpretation.md).

---

## Weiterführende Workshops

[Explorative Datenanalyse](../explorative_datenanalyse/) ·
[Bildklassifikation](../iamge_classification/) ·
[Reinforcement Learning](../reinforcement_learning/)
