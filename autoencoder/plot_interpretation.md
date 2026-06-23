# Interpretation der Autoencoder-Plots

Diese Datei erklaert, wie du die Ausgaben aus `workshop_autoencoder.py` interpretierst.

## 1) Training & Validation Loss

Plot-Titel: `Training & Validation Loss`

Was du siehst:
- `train` ist der Fehler auf den Trainingsdaten.
- `val` ist der Fehler auf dem Validierungsanteil aus den Trainingsdaten.
- Y-Achse ist MSE (Mean Squared Error), also Rekonstruktionsfehler.

Interpretation:
- Gut: Beide Kurven fallen und stabilisieren sich auf niedrigem Niveau.
- Overfitting: `train` sinkt weiter, `val` stagniert oder steigt.
- Underfitting: Beide Kurven bleiben relativ hoch und verbessern sich kaum.
- Starkes Rauschen in den Kurven kann auf zu hohe Learning Rate oder zu kleine Batch-Groesse hindeuten.

Typische Maßnahmen:
- Bei Overfitting: `DROPOUT` erhoehen, `L2` erhoehen, Modell kleiner machen (`HIDDEN_DIMS` reduzieren), frueher stoppen (`PATIENCE`).
- Bei Underfitting: Modellkapazitaet erhoehen (`HIDDEN_DIMS`, `LATENT_DIM`), laenger trainieren (`EPOCHS`), Learning Rate vorsichtig anpassen.

## 2) Fehlerverteilung (Histogramm)

Plot-Titel: `Fehlerverteilung`

Was du siehst:
- Histogramm der `Train`-Fehler und `Test`-Fehler.
- Gestrichelte Linie: `Threshold` (aus dem Train-Quantil, z. B. 95%).

Interpretation:
- Train links und schmal, Test aehnlich: Datenverteilung ist aehnlich, wenig auffaellige Punkte.
- Test deutlich weiter rechts: Testdaten enthalten mehr ungewoehnliche Muster.
- Viele Testwerte rechts vom Threshold: hoher Anteil erkannter Anomalien.
- Fast keine Testwerte rechts vom Threshold: konservativer Threshold oder sehr saubere Testdaten.

Wichtiger Punkt:
- Der Threshold basiert auf `train_errors` und `THRESHOLD_QUANTILE`.
- Hoeheres Quantil (z. B. 0.99): weniger Anomalien (strenger).
- Niedrigeres Quantil (z. B. 0.90): mehr Anomalien (sensibler).

## 3) Latent-Space (farbkodiert nach Fehler)

Plot-Titel: `Latent-Space (Farben = Rekonstruktionsfehler)`

Was du siehst:
- Jeder Punkt ist ein Testbeispiel.
- Position im 2D-Latent-Raum (oder PCA-Projektion, falls noetig).
- Farbe entspricht Rekonstruktionsfehler (MSE).

Interpretation:
- Kompakte Cluster mit aehnlichen Farben: Modell hat stabile Struktur gelernt.
- Isolierte Punkte mit deutlich hoeherem Fehler ("heisse" Farben): Kandidaten fuer Anomalien.
- Wenn fast alle Farben aehnlich sind, trennt das Modell normale und ungewoehnliche Punkte kaum.
- Wenn der Raum chaotisch wirkt, kann die Modellarchitektur oder Trainingskonfiguration unpassend sein.

## 4) Konsistente Gesamtinterpretation

Du solltest die drei Plots immer zusammen lesen:

1. Loss-Kurven pruefen: Lernt das Modell stabil und ohne klares Overfitting?
2. Histogramm pruefen: Gibt es klare Trennung zwischen normalen und auffaelligen Fehlern?
3. Latent-Space pruefen: Entsprechen hohe Fehler visuell isolierten Bereichen?

Wenn alle drei Signale zusammenpassen, ist deine Anomalie-Erkennung deutlich glaubwuerdiger.

## 5) Bezug zur Konsolenausgabe

Im Terminal siehst du zusaetzlich:
- `Threshold`
- Mittelwert und Standardabweichung der Train-/Test-Fehler
- `Anteil Test-"Anomalien"`

Schnellregel:
- Sehr hoher Anomalie-Anteil bei gleichzeitig schlechtem Validation-Verlauf deutet eher auf Modell-/Trainingsproblem hin als auf echte Ausreisser.
- Moderater Anomalie-Anteil plus stabile Lernkurven und klare Randpunkte im Latent-Space spricht eher fuer plausible Anomalien.