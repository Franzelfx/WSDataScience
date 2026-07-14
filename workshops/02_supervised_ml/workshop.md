# Workshop: Überwachtes Lernen – Churn-Vorhersage mit scikit-learn

## Überblick

In diesem Workshop baust du eine **komplette, praxisnahe Machine-Learning-Pipeline**
für ein reales Business-Problem: die Vorhersage der **Kundenabwanderung (Churn)**
eines Telekommunikationsanbieters. Es geht **nicht** um Deep Learning, sondern um
das klassische, im Alltag am häufigsten eingesetzte **überwachte Lernen auf
Tabellendaten** mit `scikit-learn`.

Du durchläufst den vollständigen Workflow:

1. **Daten laden** – echte CSV oder deterministisch erzeugte synthetische Daten
2. **Klassenverteilung ansehen** – das Problem ist bewusst **unbalanciert** (~26 % Churn)
3. **Stratifizierter Train/Test-Split**
4. **Preprocessing-Pipeline** (`ColumnTransformer`): Skalierung + One-Hot-Encoding – ohne **Data Leakage**
5. **Modellvergleich** per **Stratified K-Fold Cross-Validation**
6. **Klassen-Ungleichgewicht** behandeln (`class_weight="balanced"`)
7. **Evaluation**: `classification_report`, ROC-AUC, Confusion Matrix, ROC- & Precision-Recall-Kurve
8. **Threshold-Wahl** nach Business-Kosten (statt starrer 0.5)
9. **Hyperparameter-Tuning** per `GridSearchCV`
10. **Interpretierbarkeit**: Permutation Importance / Koeffizienten (+ optional SHAP)
11. **Modell speichern/laden** (joblib) und **Inferenz** auf Beispielkunden

> Dieser Workshop gehört zur Data-Science-Workshop-Reihe und nutzt denselben
> Aufbau (zentraler `CONFIG`-Block, CLI-Parameter, gespeicherte Plots).

---

## Der Business-Case: Warum Churn?

Einen bestehenden Kunden zu **halten** ist meist um ein Vielfaches günstiger, als
einen neuen zu **gewinnen**. Wenn wir abwanderungsgefährdete Kunden **früh**
erkennen, kann das Marketing gezielt mit Rabatten oder besserem Service
gegensteuern (**Retention**). Genau das ist die Aufgabe unseres Modells: Es soll
zu jedem Kunden die Wahrscheinlichkeit `P(Churn)` schätzen.

Dabei sind die Fehler **unterschiedlich teuer**:

- Ein **übersehener Churner** (False Negative) = verlorener Kunde → **teuer**.
- Ein **Fehlalarm** (False Positive) = unnötige Retention-Maßnahme → **günstiger**.

Diese Asymmetrie ist der Grund, warum wir später die Entscheidungsschwelle nicht
naiv bei 0.5 lassen.

---

## Zentrale Konzepte

### Überwachtes Lernen: Klassifikation vs. Regression

Beim **überwachten Lernen (supervised learning)** kennen wir zu jeder Beobachtung
das **Label** und lernen eine Abbildung `Features → Label`.

- **Klassifikation**: Label ist eine Kategorie (hier: `Churn = Yes/No`, also **binär**).
- **Regression**: Label ist eine kontinuierliche Zahl (z. B. Umsatzhöhe).

Wir machen **binäre Klassifikation** und lassen uns von den Modellen nicht nur die
Klasse, sondern eine **Wahrscheinlichkeit** (`predict_proba`) geben – die ist für
die Threshold-Wahl entscheidend.

### Train/Test-Split & Cross-Validation

- Der **Test-Split** wird beiseitegelegt und erst ganz am Ende angefasst. Er
  simuliert „neue, ungesehene Kunden".
- **Stratifizierung** (`stratify=y`) sorgt dafür, dass das Churn-Verhältnis in
  Train und Test gleich bleibt – bei seltenen Klassen unverzichtbar.
- Zum **fairen Modellvergleich** nutzen wir **Stratified K-Fold
  Cross-Validation**: Die Trainingsdaten werden in *k* Teile zerlegt, es wird
  *k*-mal trainiert/validiert, und wir mitteln die Scores. Das ist robuster als
  ein einzelner Validierungs-Split.

### Data Leakage – und wie die Pipeline es verhindert

**Data Leakage** entsteht, wenn Information aus den Testdaten ins Training
sickert. Klassischer Fehler: den `StandardScaler` auf dem **gesamten** Datensatz
`fit`ten und *dann* splitten – der Scaler „kennt" dann schon die Testverteilung.

Die Lösung: **Preprocessing gehört IN die Pipeline.** Dann wird bei jedem
CV-Fold der Scaler/Encoder nur auf den jeweiligen Trainingsdaten `fit`tet und auf
den Validierungsdaten nur `transform`t. `Pipeline` + `ColumnTransformer` erledigen
das automatisch korrekt.

### Klassen-Ungleichgewicht & die richtige Metrik

Nur ~26 % der Kunden churnen. Ein Modell, das **immer „kein Churn"** sagt, hätte
~74 % **Accuracy** – und wäre trotzdem **wertlos**. Deshalb:

- **Accuracy ist hier irreführend.** Nicht als Hauptmetrik verwenden.
- Wichtiger sind:
  - **Precision** = Von den als Churn markierten – wie viele churnen wirklich?
  - **Recall** (Sensitivität) = Von allen echten Churnern – wie viele finden wir?
  - **F1** = harmonisches Mittel aus Precision und Recall.
  - **ROC-AUC** = wie gut trennt das Modell die Klassen über alle Schwellen (0.5 = Zufall, 1.0 = perfekt).
  - **PR-AUC / Average Precision** = Fläche unter der Precision-Recall-Kurve.
- Gegenmittel beim Training: `class_weight="balanced"` gewichtet die seltene
  Klasse stärker (bei `LogisticRegression` und `RandomForest`).
  `GradientBoosting` kennt das nicht – dort steuern wir über die **Threshold-Wahl**.

### ROC-Kurve vs. Precision-Recall-Kurve

- Die **ROC-Kurve** (True Positive Rate über False Positive Rate) ist beliebt,
  kann bei **starkem Ungleichgewicht** aber zu optimistisch wirken, weil sehr
  viele echte Negative die False Positive Rate „verdünnen".
- Die **Precision-Recall-Kurve** fokussiert die seltene Positiv-Klasse und ist
  bei Churn/Betrug/seltenen Ereignissen oft **aussagekräftiger**. Ihre Baseline
  ist der Positiv-Anteil (~0.26), nicht 0.5.

### Threshold-Trade-off

`predict()` sagt standardmäßig „Churn", wenn `P(Churn) ≥ 0.5`. Diese **0.5 ist
willkürlich**. Verschiebt man die Schwelle:

- **niedriger** → mehr Recall (weniger übersehene Churner), aber mehr Fehlalarme,
- **höher** → höhere Precision, aber mehr verpasste Churner.

Wir wählen die Schwelle **nach Business-Kosten**: Wir minimieren
`FN · COST_FN + FP · COST_FP`. Weil ein verpasster Churner teurer ist als ein
Fehlalarm, rutscht das Optimum meist deutlich **unter** 0.5.

### Interpretierbarkeit

Ein gutes Modell soll auch **erklärbar** sein:

- **Permutation Importance** (modellagnostisch): Wie stark fällt die Güte, wenn
  man eine Spalte zufällig durchmischt? Große Verschlechterung = wichtige Spalte.
- **Koeffizienten** (LogisticRegression, in Log-Odds) bzw. **`feature_importances_`**
  (Baum-Modelle) zeigen den modell-internen Einfluss.
- **SHAP** (optional) liefert konsistente, pro-Vorhersage-Erklärungen – wird nur
  genutzt, wenn `shap` installiert ist (keine harte Abhängigkeit).

---

## Projektstruktur

| Pfad | Inhalt |
|------|--------|
| `aufgabe/workshop_supervised_ml.py` | **Starter-Code** mit TODO-Lücken – hier programmierst du. |
| `loesung/workshop_supervised_ml.py` | **Musterlösung** – zum Vergleichen und Nachschlagen. |
| `data/` | Datensatz (synthetisch generiert oder echter Telco-CSV) + Daten-README |
| `requirements.txt` | Abhängigkeiten (pandas, numpy, scikit-learn, matplotlib, seaborn, joblib; optional shap) |

**So arbeitest du:** Öffne `aufgabe/workshop_supervised_ml.py` und fülle die mit
`# TODO (Aufgabe X)` markierten Lücken aus. Das restliche Gerüst (Daten,
Plot-Funktionen, `argparse`, `main()`, Speichern/Laden) ist bereits fertig. Wenn
du nicht weiterkommst, schau in `loesung/`.

---

## Voraussetzungen

- **Python 3.10+** (empfohlen)

```bash
pip install -r requirements.txt
```

> **Hinweis:** Beim ersten Lauf werden – falls keine `data/telco_churn.csv`
> existiert – automatisch **synthetische Daten** erzeugt und dort gespeichert.
> Details siehe [`data/README.md`](data/README.md).

---

## Der Datensatz

Gemischte Tabellendaten mit numerischen und kategorischen Spalten:

| Typ | Spalten |
|-----|---------|
| **numerisch** | `tenure` (Monate), `MonthlyCharges`, `TotalCharges` |
| **kategorisch** | `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `Contract`, `InternetService`, `PaymentMethod`, `PaperlessBilling`, `TechSupport` |
| **Ziel** | `Churn` (`Yes`/`No`), ~26 % `Yes` (**unbalanciert**) |

Das Signal ist realistisch: kurze `tenure`, `Contract = Month-to-month` und hohe
`MonthlyCharges` erhöhen das Churn-Risiko; lange Bindung, `Two year`-Vertrag und
aktiver `TechSupport` senken es. Neukunden (`tenure = 0`) haben absichtlich einen
fehlenden `TotalCharges`-Wert → die Pipeline muss **imputieren**.

---

## Ausführen

> Die Befehle rufst du im jeweiligen Ordner auf – `aufgabe/`, sobald die TODOs
> implementiert sind, oder `loesung/` für die fertige Musterlösung.

```bash
# Modellvergleich (CV), bestes Modell trainieren + auf Testset evaluieren
python workshop_supervised_ml.py --train --evaluate

# Bestimmtes Modell erzwingen
python workshop_supervised_ml.py --train --evaluate --model rf

# Zusätzlich Hyperparameter-Tuning
python workshop_supervised_ml.py --train --evaluate --tune

# Interpretierbarkeit (Permutation Importance, optional SHAP)
python workshop_supervised_ml.py --train --evaluate --explain

# Gespeichertes Modell erneut evaluieren (ohne Neutraining)
python workshop_supervised_ml.py --evaluate
```

### Alle CLI-Parameter

| Parameter    | Beschreibung                                              | Standard |
|--------------|----------------------------------------------------------|----------|
| `--train`    | Modellvergleich (CV), bestes Modell trainieren & speichern | –        |
| `--evaluate` | Modell auf dem Testset evaluieren (Metriken + Plots)     | –        |
| `--tune`     | Hyperparameter-Tuning per `GridSearchCV`                 | –        |
| `--explain`  | Interpretierbarkeit (Permutation Importance, optional SHAP) | –     |
| `--model`    | Modell erzwingen: `logreg` \| `rf` \| `gb`                | CV-Auswahl |
| `--data`     | Pfad zu einer eigenen CSV                                | synthetisch |

---

## Programmier-Aufgaben (TODOs im Starter-Code)

Diese vier Lücken füllst du in `aufgabe/workshop_supervised_ml.py` aus. Jede ist
im Code klar mit `# TODO (Aufgabe X)` markiert und enthält direkt den passenden
Hinweis. Erst wenn alle vier implementiert sind, läuft das Skript durch.

### Aufgabe 1 – Preprocessing mit `ColumnTransformer` (`build_preprocessor`)

Baue zwei kleine Pipelines und fasse sie im `ColumnTransformer` zusammen:

- **numerisch**: `SimpleImputer(strategy="median")` → `StandardScaler()`
- **kategorisch**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

**Worauf achten:**
- `handle_unknown="ignore"` verhindert Fehler, wenn bei der Inferenz eine
  **unbekannte Kategorie** auftaucht (sie wird zum Nullvektor).
- Median-Imputation ist robuster gegen Ausreißer als der Mittelwert.
- Alles gehört **in** den Transformer – so passiert `fit` später nur auf
  Trainingsdaten (kein Leakage).

### Aufgabe 2 – Modell-Pipeline zusammenbauen (`build_pipeline`)

Setze eine `Pipeline` aus zwei Schritten zusammen:
`("preprocessor", build_preprocessor(cfg))` und
`("classifier", build_classifier(name, cfg))`.

**Worauf achten:** Weil Preprocessing **und** Modell in **einem** Objekt stecken,
lässt sich die ganze Pipeline sauber per Cross-Validation bewerten, tunen und mit
`joblib` speichern. Beim `fit(X_train, y_train)` wird die komplette Kette
trainiert.

### Aufgabe 3 – Cross-Validation auswerten (`compare_models`)

Iteriere über `["logreg", "rf", "gb"]`, baue je Modell die Pipeline und rufe
`cross_validate(...)` mit der vorbereiteten `StratifiedKFold`-CV und dem
`scoring`-Set auf. Gib je Metrik **Mittelwert ± Std** über die Folds aus und
bestimme das beste Modell nach der Primär-Metrik (`cfg["PRIMARY_SCORING"]`,
Standard `roc_auc`).

**Worauf achten:**
- `cross_validate` liefert ein dict; die Fold-Scores stehen unter
  `scores["test_<metrik>"]` (ein numpy-Array).
- Der **Mittelwert** sagt, wie gut das Modell ist; die **Std** sagt, wie stabil.
- `roc_auc` als Auswahlkriterium ist schwellenunabhängig und bei
  Ungleichgewicht sinnvoller als Accuracy.

### Aufgabe 4 – Threshold nach Business-Kosten (`choose_threshold`)

Bestimme aus den Kandidaten-Schwellen der Precision-Recall-Kurve:

- **(a) kosten-optimal**: für jede Schwelle `FN·COST_FN + FP·COST_FP` berechnen und
  das **Minimum** nehmen,
- **(b) Recall-Ziel**: die kleinste Schwelle, die `recall ≥ TARGET_RECALL` erfüllt,
  darunter die mit der besten Precision.

**Worauf achten:**
- `y_proba` ist die Wahrscheinlichkeit der **Positiv-Klasse** (`Churn = Yes`).
- Ein hoher `COST_FN` drückt die Schwelle **nach unten** (mehr Recall, weniger
  verpasste Churner) – das ist der Kern der bewussten Threshold-Wahl.
- Vergleiche im Ergebnis die Reports bei 0.5 und bei der Kosten-Schwelle:
  typischerweise steigt der **Recall** deutlich.

---

## Interpretation der Plots

Nach `--train --evaluate` (ggf. `--explain`) liegen unter `plots/`:

| Datei | Was sie zeigt / wie lesen |
|-------|---------------------------|
| `confusion_matrix.png` | 2×2-Matrix bei der gewählten Schwelle. Diagonale = korrekt. Unten links = **übersehene Churner** (FN, teuer!), oben rechts = **Fehlalarme** (FP). |
| `roc_curve.png` | TPR über FPR. Je weiter **oben links**, desto besser. AUC nahe 1.0 = starke Trennung, 0.5 = Zufall (gestrichelte Linie). |
| `precision_recall_curve.png` | Precision über Recall. Baseline = Churn-Anteil (~0.26). Bei Ungleichgewicht aussagekräftiger als ROC. |
| `threshold_cost.png` | Erwartete Gesamtkosten je Schwelle. Das Minimum (schwarze Linie) ist die gewählte Schwelle; die graue 0.5-Linie zeigt, wie viel die naive Wahl kosten würde. |
| `permutation_importance.png` | Wichtigkeit der **rohen** Eingabespalten (Rückgang der ROC-AUC beim Durchmischen). Erwartung: `Contract`, `tenure`, `MonthlyCharges` weit oben. |
| `model_feature_importance.png` / `model_coefficients.png` | Modell-interne Wichtigkeiten (Bäume) bzw. Koeffizienten (LogReg, in Log-Odds; positiv = erhöht Churn-Risiko). |
| `shap_summary.png` | (nur mit `--explain` **und** installiertem `shap`) Beitrag jedes Features je Vorhersage. |

**Sinnvolle Ergebnisse erkennst du daran**, dass die wichtigsten Treiber genau die
fachlich erwarteten sind: kurze Vertragsdauer, Month-to-month-Verträge und hohe
monatliche Kosten treiben das Churn-Risiko nach oben.

---

## Hyperparameter & Konfiguration

Im `CONFIG`-Block am Anfang des Skripts findest du alle Parameter gesammelt:

| Parameter | Beschreibung | Werte zum Testen |
|-----------|--------------|------------------|
| `MODEL` | Standardmodell | `logreg`, `rf`, `gb` |
| `PRIMARY_SCORING` | Auswahl-Metrik im Modellvergleich | `roc_auc`, `average_precision`, `f1` |
| `CV_FOLDS` | Anzahl CV-Folds | 3, 5, 10 |
| `TEST_SIZE` | Anteil Testdaten | 0.2, 0.25, 0.3 |
| `COST_FN` / `COST_FP` | Kosten pro FN / FP für die Threshold-Wahl | z. B. 300/50, 500/20 |
| `TARGET_RECALL` | Mindest-Recall für die Recall-Strategie | 0.7, 0.8, 0.9 |
| `N_SYNTHETIC` | Zeilen der synthetischen Daten | 4000–7000 |

---

## Experimente

1. **Baseline**: `--train --evaluate` ausführen. Welches Modell gewinnt in der CV?
   Wie unterscheiden sich ROC-AUC und Average Precision?
2. **Accuracy-Falle**: Vergleiche Accuracy mit Recall/F1 im `classification_report`.
   Warum wäre ein „immer Nein"-Modell trotz hoher Accuracy nutzlos?
3. **Threshold verschieben**: Setze `COST_FN` von 300 auf 800 – wie wandert die
   kostenoptimale Schwelle und wie ändert sich der Recall?
4. **Tuning**: `--tune` einschalten. Bringt das kleine Grid einen messbaren
   Gewinn beim CV-Score?
5. **Interpretierbarkeit**: `--explain` ausführen und prüfen, ob die wichtigsten
   Features fachlich Sinn ergeben. Optional `pip install shap`.
6. **Modelltausch**: `--model logreg` vs. `--model gb` – Trade-off zwischen
   Erklärbarkeit (Koeffizienten) und Vorhersagegüte.

---

## Tipps & Tricks

- **`predict` vs. `predict_proba`**: Für Threshold-Experimente immer mit
  `predict_proba(...)[:, 1]` arbeiten.
- **Reproduzierbarkeit**: `RANDOM_SEED` fixiert Split, Modelle und CV – für faire
  Vergleiche gleich lassen.
- **Kein Leakage**: Niemals außerhalb der Pipeline auf dem Gesamtdatensatz
  skalieren/encoden.
- **Metrik zur Aufgabe passend wählen**: Bei seltenen, teuren Ereignissen zählt
  meist **Recall** (bzw. PR-AUC) mehr als Accuracy.
- **Modell neu erzeugen**: `data/telco_churn.csv` löschen, um neue synthetische
  Daten zu generieren.

---

## Weiterführende Ideen

- **SMOTE / Resampling** (via `imbalanced-learn`) als Alternative zu `class_weight`.
- **Kalibrierung** der Wahrscheinlichkeiten (`CalibratedClassifierCV`).
- **RandomizedSearchCV** für größere Suchräume.
- Weitere Modelle: `HistGradientBoostingClassifier`, XGBoost, LightGBM.
- **Fairness**: Churn-Vorhersagen über Gruppen (`gender`, `SeniorCitizen`) prüfen.

---

## Referenzen

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn: Pipelines & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Metriken für Klassifikation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [IBM Telco Customer Churn (Beispieldatensatz)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [SHAP – Interpretierbarkeit](https://shap.readthedocs.io/)
