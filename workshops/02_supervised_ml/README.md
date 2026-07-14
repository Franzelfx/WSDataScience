# Workshop 02 · Überwachtes Lernen: Churn-Vorhersage (scikit-learn)

In diesem Workshop baust du eine komplette, praxisnahe Machine-Learning-Pipeline mit scikit-learn, um die Kundenabwanderung (Churn) eines Telekommunikationsanbieters vorherzusagen – ein klassisches Problem der **binären Klassifikation** auf gemischten Tabellendaten (numerisch + kategorisch).

## Lernziele
- Den vollständigen Supervised-ML-Workflow verstehen: Split → Preprocessing → Modellwahl → Evaluation → Deployment.
- Preprocessing sauber in eine `Pipeline`/`ColumnTransformer` verpacken und so **Data Leakage** vermeiden.
- Modelle fair per **stratifizierter Cross-Validation** vergleichen (LogisticRegression, RandomForest, GradientBoosting).
- Mit **Klassen-Ungleichgewicht** umgehen und die **richtigen Metriken** wählen (Precision/Recall/F1, ROC-AUC, PR-AUC statt Accuracy).
- Die **Entscheidungsschwelle** nach Business-Kosten wählen statt starr bei 0.5.
- Modelle **interpretieren** (Permutation Importance / Koeffizienten, optional SHAP) sowie **speichern, laden und für Inferenz** nutzen.

## Struktur
| Pfad | Inhalt |
|------|--------|
| [`workshop.md`](workshop.md) | Ausführliche Anleitung & Erklärung |
| [`aufgabe/`](aufgabe/) | Starter-Code mit TODO-Lücken |
| [`loesung/`](loesung/) | Musterlösung |
| [`data/`](data/) | Datensatz (synthetisch generiert oder echter Telco-CSV) |
| [`requirements.txt`](requirements.txt) | Abhängigkeiten |

## Setup
```bash
pip install -r requirements.txt
```

## Ausführen
Führe den Code im Ordner `aufgabe/` aus (sobald die TODOs implementiert sind) oder in `loesung/` für die fertige Musterlösung. Beim ersten Lauf werden automatisch synthetische Daten unter `data/telco_churn.csv` erzeugt.

```bash
# Modellvergleich (CV), bestes Modell trainieren + auf Testset evaluieren
python aufgabe/workshop_supervised_ml.py --train --evaluate

# Bestimmtes Modell erzwingen (logreg | rf | gb)
python aufgabe/workshop_supervised_ml.py --train --evaluate --model rf

# Zusätzlich Hyperparameter-Tuning per GridSearchCV
python aufgabe/workshop_supervised_ml.py --train --evaluate --tune

# Interpretierbarkeit (Permutation Importance, optional SHAP)
python aufgabe/workshop_supervised_ml.py --train --evaluate --explain

# Eigene CSV verwenden
python aufgabe/workshop_supervised_ml.py --train --evaluate --data pfad/zu/telco.csv

# Fertige Musterlösung
python loesung/workshop_supervised_ml.py --train --evaluate
```

[← Zur Workshop-Übersicht](../../README.md)
