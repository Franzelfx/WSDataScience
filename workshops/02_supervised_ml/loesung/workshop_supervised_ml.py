#!/usr/bin/env python3
"""
Workshop: Überwachtes Lernen auf Tabellendaten – Churn-Vorhersage (scikit-learn)
================================================================================
Dieses Skript demonstriert den vollständigen, praxisnahen Workflow eines
klassischen Machine-Learning-Projekts für **binäre Klassifikation** am Beispiel
der **Kundenabwanderung (Churn)** eines Telekommunikationsanbieters:

  1. Daten laden (echte CSV oder deterministisch erzeugte synthetische Daten)
  2. Klassenverteilung ansehen (das Problem ist bewusst UNBALANCIERT ~26% Churn)
  3. Stratifizierter Train/Test-Split
  4. Preprocessing-Pipeline (ColumnTransformer): Skalierung + One-Hot-Encoding
     -> vermeidet Data Leakage, weil ALLES in der Pipeline steckt
  5. Modellvergleich per Stratified K-Fold Cross-Validation (roc_auc & Co.)
  6. Klassen-Ungleichgewicht behandeln (class_weight="balanced")
  7. Bestes Modell evaluieren: classification_report, ROC-AUC,
     Confusion Matrix, ROC-Kurve, Precision-Recall-Kurve
  8. Entscheidungsschwelle nach Business-Kosten wählen (statt starrer 0.5)
  9. Hyperparameter-Tuning per GridSearchCV (Flag --tune)
 10. Interpretierbarkeit: Permutation Importance / Koeffizienten (+ optional SHAP)
 11. Modell speichern/laden (joblib) und Inferenz auf Beispielkunden

Überwachtes Lernen (supervised learning) bedeutet: Wir haben zu jeder Beobachtung
ein bekanntes **Label** (hier: hat der Kunde gekündigt, Ja/Nein) und lernen eine
Abbildung Features -> Label. Bei der Churn-Vorhersage ist das ein echter
Business-Case: einen Kunden zu HALTEN ist meist deutlich billiger als einen neuen
zu gewinnen – wir wollen abwanderungsgefährdete Kunden früh erkennen.

Aufruf:
  python workshop_supervised_ml.py --train --evaluate
  python workshop_supervised_ml.py --train --evaluate --model rf
  python workshop_supervised_ml.py --train --evaluate --tune
  python workshop_supervised_ml.py --evaluate --explain
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # << Nicht-interaktives Backend (Server-tauglich)
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# Pfade robust relativ zu diesem Skript (funktioniert auch aus loesung/ heraus).
_HERE = os.path.dirname(__file__)
DEFAULT_DATA_CSV = os.path.join(_HERE, "..", "data", "telco_churn.csv")
DEFAULT_MODEL_PATH = os.path.join(_HERE, "..", "saved_model", "churn_model.joblib")


# ╔══════════════════════════════════════════════════════════╗
# ║               KONFIGURATION / HYPERPARAMETER             ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Hier kannst du alle wichtigen Parameter zentral         ║
# ║  anpassen und experimentieren.                           ║
# ╚══════════════════════════════════════════════════════════╝
CONFIG = {
    # ── Daten ──────────────────────────────────────────────
    "DATA_CSV": DEFAULT_DATA_CSV,     # Pfad zur (echten) CSV; sonst synthetisch
    "N_SYNTHETIC": 6000,              # << Zeilen der synthetischen Daten
    "TARGET": "Churn",                # Zielspalte (Yes/No)
    "POSITIVE_LABEL": "Yes",          # Was gilt als "positiv" (= Kunde wandert ab)

    # Feature-Listen (welche Spalte ist numerisch, welche kategorisch?)
    "NUMERIC_FEATURES": ["tenure", "MonthlyCharges", "TotalCharges"],
    "CATEGORICAL_FEATURES": [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "Contract", "InternetService", "PaymentMethod",
        "PaperlessBilling", "TechSupport",
    ],

    # ── Split & Cross-Validation ───────────────────────────
    "TEST_SIZE": 0.2,                 # << Anteil Testdaten
    "CV_FOLDS": 5,                    # << Anzahl Folds der Stratified K-Fold CV
    "RANDOM_SEED": 42,                # Fester Seed = reproduzierbar

    # ── Modelle ────────────────────────────────────────────
    "MODEL": "gb",                    # << Standardmodell: logreg | rf | gb
    "PRIMARY_SCORING": "roc_auc",     # << Metrik, nach der das beste Modell gewählt wird

    # ── Threshold nach Business-Kosten ─────────────────────
    # Ein verpasster Churner (False Negative) ist teuer (Kunde geht verloren),
    # ein Fehlalarm (False Positive) kostet nur eine Retention-Maßnahme.
    "COST_FN": 300.0,                 # << Kosten für einen übersehenen Churner (€)
    "COST_FP": 50.0,                  # << Kosten für einen Fehlalarm (€)
    "TARGET_RECALL": 0.80,            # << Alternative: Mindest-Recall, den wir halten wollen

    # ── Sonstiges ──────────────────────────────────────────
    "PLOT_DIR": os.path.join(_HERE, "plots"),
    "MODEL_PATH": DEFAULT_MODEL_PATH,
}


# ╔══════════════════════════════════════════════════════════╗
# ║                     HILFSFUNKTIONEN                      ║
# ╚══════════════════════════════════════════════════════════╝

def set_seeds(seed: int) -> None:
    """Reproduzierbare Ergebnisse durch fixierten Zufallsseed."""
    np.random.seed(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║              SYNTHETISCHE CHURN-DATEN ERZEUGEN          ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Falls keine echte CSV vorliegt, erzeugen wir            ║
# ║  deterministisch (fester Seed) einen realistischen       ║
# ║  Telco-Churn-Datensatz mit echtem Signal: kurze tenure,  ║
# ║  Month-to-month-Vertrag und hohe MonthlyCharges erhöhen  ║
# ║  das Abwanderungsrisiko.                                  ║
# ╚══════════════════════════════════════════════════════════╝

def generate_synthetic_churn(n: int, seed: int) -> pd.DataFrame:
    """Erzeugt einen realistischen, unbalancierten synthetischen Churn-Datensatz."""
    rng = np.random.default_rng(seed)

    # ── Rohe Features ziehen (Verteilungen ähnlich zum echten Telco-Datensatz) ──
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n)                       # 0..72 Monate
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    techsupport = rng.choice(["Yes", "No"], n, p=[0.29, 0.71])

    # MonthlyCharges hängen vom Produkt (Internet) ab
    base = np.where(internet == "Fiber optic", 70.0,
                    np.where(internet == "DSL", 50.0, 20.0))
    monthly = np.clip(base + rng.normal(0, 10, n), 18.0, 120.0)
    # TotalCharges ~ tenure * MonthlyCharges (+ etwas Rauschen)
    total = np.clip(monthly * tenure + rng.normal(0, 20, n), 0.0, None)

    # ── Churn-Wahrscheinlichkeit als logistische Funktion der Features ──
    # Die Vorzeichen kodieren plausible Domänen-Logik.
    logit = (
        -1.10                                                    # Intercept (Basisrate)
        + 1.40 * (contract == "Month-to-month")                  # kurzfristiger Vertrag -> Risiko hoch
        - 0.90 * (contract == "Two year")                        # 2-Jahres-Vertrag bindet
        - 0.030 * tenure                                         # je länger dabei, desto treuer
        + 0.015 * (monthly - 60.0)                              # teure Tarife -> unzufriedener
        + 0.50 * (internet == "Fiber optic")                    # Fiber-Kunden churnen mehr (Preis/Erwartung)
        + 0.40 * (payment == "Electronic check")                # Zahlart korreliert mit Churn
        + 0.30 * (paperless == "Yes")
        - 0.40 * (techsupport == "Yes")                         # Support bindet
        + 0.25 * senior
        - 0.15 * (partner == "Yes")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    df = pd.DataFrame({
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "TechSupport": techsupport,
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total, 2),
        "Churn": churn,
    })

    # Wie im echten Telco-Datensatz: Neukunden (tenure == 0) haben noch keinen
    # TotalCharges-Eintrag -> fehlende Werte, die wir später imputieren müssen.
    df.loc[df["tenure"] == 0, "TotalCharges"] = np.nan
    return df


def load_dataset(cfg: dict) -> pd.DataFrame:
    """
    Lädt den Churn-Datensatz. Existiert die CSV unter cfg['DATA_CSV'], wird sie
    verwendet. Andernfalls werden synthetische Daten erzeugt UND als CSV
    gespeichert (damit spätere Läufe denselben Datensatz sehen).
    """
    path = cfg["DATA_CSV"]
    if path and os.path.exists(path):
        print(f"[INFO] Lade Datensatz aus: {path}")
        df = pd.read_csv(path)
    else:
        print("[INFO] Keine CSV gefunden -> erzeuge synthetischen Churn-Datensatz "
              f"({cfg['N_SYNTHETIC']} Zeilen, Seed={cfg['RANDOM_SEED']}).")
        df = generate_synthetic_churn(cfg["N_SYNTHETIC"], cfg["RANDOM_SEED"])
        _ensure_dir(os.path.dirname(path))
        df.to_csv(path, index=False)
        print(f"[INFO] Synthetische Daten gespeichert: {path}")
    return df


def summarize_dataset(df: pd.DataFrame, cfg: dict) -> None:
    """Kurzer Überblick + Klassenverteilung (zeigt das Ungleichgewicht)."""
    target = cfg["TARGET"]
    print(f"[INFO] Datenform: {df.shape[0]} Zeilen x {df.shape[1]} Spalten")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"[INFO] Fehlende Werte insgesamt: {n_missing} "
              "(werden in der Pipeline imputiert).")
    counts = df[target].value_counts()
    share = df[target].value_counts(normalize=True)
    print("[INFO] Klassenverteilung (Churn):")
    for label in counts.index:
        print(f"        {label:>4}: {counts[label]:5d}  ({100 * share[label]:.1f}%)")
    pos = cfg["POSITIVE_LABEL"]
    if pos in share:
        print(f"[INFO] Positiv-Klasse '{pos}' = {100 * share[pos]:.1f}%  "
              "-> UNBALANCIERT. Accuracy allein ist hier irreführend!")


# ╔══════════════════════════════════════════════════════════╗
# ║            PREPROCESSING & MODELL-PIPELINES             ║
# ╠══════════════════════════════════════════════════════════╣
# ║  WICHTIG: Preprocessing gehört IN die Pipeline. So wird  ║
# ║  der Scaler/Encoder nur auf den Trainingsdaten "gefittet"║
# ║  (fit) und dann auf Testdaten nur angewendet (transform) ║
# ║  -> kein Data Leakage, korrekte Cross-Validation.        ║
# ╚══════════════════════════════════════════════════════════╝

def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """
    Baut den ColumnTransformer:
      - numerische Spalten -> Imputation (Median) + StandardScaler
      - kategorische Spalten -> Imputation (häufigster Wert) + OneHotEncoder
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": unbekannte Kategorien im Test -> 0-Vektor,
        # statt eines Fehlers. Wichtig für robuste Inferenz.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, cfg["NUMERIC_FEATURES"]),
        ("cat", categorical_pipeline, cfg["CATEGORICAL_FEATURES"]),
    ])
    return preprocessor


def build_classifier(name: str, cfg: dict):
    """Liefert einen (noch untrainierten) Klassifikator per Kurzname."""
    seed = cfg["RANDOM_SEED"]
    if name == "logreg":
        # class_weight="balanced": gewichtet die seltene Churn-Klasse stärker.
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    if name == "gb":
        # GradientBoosting kennt kein class_weight -> wir behandeln das
        # Ungleichgewicht später über die Threshold-Wahl (statt beim Fit).
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unbekanntes Modell: {name!r} (erlaubt: logreg, rf, gb)")


def build_pipeline(name: str, cfg: dict) -> Pipeline:
    """Baut die komplette Pipeline: Preprocessing -> Klassifikator."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(cfg)),
        ("classifier", build_classifier(name, cfg)),
    ])


# ╔══════════════════════════════════════════════════════════╗
# ║                MODELLVERGLEICH PER CROSS-VAL           ║
# ╚══════════════════════════════════════════════════════════╝

def compare_models(X_train, y_train, cfg: dict) -> dict:
    """
    Vergleicht LogisticRegression, RandomForest und GradientBoosting per
    Stratified K-Fold Cross-Validation. Liefert die CV-Scores je Modell.
    """
    cv = StratifiedKFold(
        n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_SEED"]
    )
    scoring = ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"]

    print(f"\n[CV] Modellvergleich per {cfg['CV_FOLDS']}-facher Stratified K-Fold CV")
    print(f"      (Auswahl-Metrik: {cfg['PRIMARY_SCORING']})\n")

    results = {}
    for name in ["logreg", "rf", "gb"]:
        pipe = build_pipeline(name, cfg)
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1,
            return_train_score=False,
        )
        results[name] = scores
        print(f"  Modell: {name}")
        for metric in scoring:
            vals = scores[f"test_{metric}"]
            print(f"      {metric:18s}: {vals.mean():.3f} ± {vals.std():.3f}")
        print()

    # Bestes Modell nach der Primär-Metrik (Mittelwert über die Folds)
    primary = cfg["PRIMARY_SCORING"]
    best_name = max(
        results, key=lambda k: results[k][f"test_{primary}"].mean()
    )
    best_score = results[best_name][f"test_{primary}"].mean()
    print(f"[CV] Bestes Modell laut {primary}: '{best_name}' "
          f"({best_score:.3f}).")
    return {"scores": results, "best_name": best_name}


# ╔══════════════════════════════════════════════════════════╗
# ║               HYPERPARAMETER-TUNING (--tune)           ║
# ╚══════════════════════════════════════════════════════════╝

def tune_model(name: str, X_train, y_train, cfg: dict) -> Pipeline:
    """Kleines GridSearchCV je Modell. Liefert die beste (gefittete) Pipeline."""
    cv = StratifiedKFold(
        n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_SEED"]
    )
    pipe = build_pipeline(name, cfg)

    # Parameter-Grid (klein gehalten, damit es schnell läuft). Die Präfixe
    # "classifier__" adressieren den Schritt in der Pipeline.
    grids = {
        "logreg": {
            "classifier__C": [0.05, 0.1, 0.5, 1.0, 5.0],
            "classifier__penalty": ["l2"],
        },
        "rf": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 6, 12],
            "classifier__min_samples_leaf": [1, 5],
        },
        "gb": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
    }
    param_grid = grids[name]

    print(f"\n[TUNE] GridSearchCV für '{name}' "
          f"(scoring={cfg['PRIMARY_SCORING']}) ...")
    search = GridSearchCV(
        pipe, param_grid, scoring=cfg["PRIMARY_SCORING"], cv=cv, n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"[TUNE] Beste Parameter: {search.best_params_}")
    print(f"[TUNE] Bester CV-Score ({cfg['PRIMARY_SCORING']}): "
          f"{search.best_score_:.3f}")
    return search.best_estimator_


# ╔══════════════════════════════════════════════════════════╗
# ║                 THRESHOLD NACH BUSINESS-KOSTEN         ║
# ╠══════════════════════════════════════════════════════════╣
# ║  Standardmäßig sagt ein Klassifikator "Churn", wenn      ║
# ║  P(Churn) >= 0.5. Bei unbalancierten Daten mit           ║
# ║  ungleichen Fehlerkosten ist 0.5 fast nie optimal.       ║
# ╚══════════════════════════════════════════════════════════╝

def choose_threshold(y_true, y_proba, cfg: dict) -> dict:
    """
    Wählt die Entscheidungsschwelle. Wir betrachten zwei Strategien:
      (a) Kosten-optimal: minimiere  FN*COST_FN + FP*COST_FP
      (b) Recall-Ziel:    kleinste Schwelle, die TARGET_RECALL noch erfüllt

    y_proba ist die vorhergesagte Wahrscheinlichkeit für die Positiv-Klasse.
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    # Kandidaten-Schwellen aus der Precision-Recall-Kurve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve liefert einen Punkt mehr als Schwellen -> abschneiden
    precision, recall = precision[:-1], recall[:-1]

    # (a) Kostenminimum über alle Kandidaten-Schwellen
    cost_fn, cost_fp = cfg["COST_FN"], cfg["COST_FP"]
    costs = []
    for thr in pr_thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
    costs = np.asarray(costs)
    best_idx = int(np.argmin(costs))
    cost_threshold = float(pr_thresholds[best_idx])

    # (b) Kleinste Schwelle, die den Ziel-Recall noch erreicht
    target_recall = cfg["TARGET_RECALL"]
    ok = np.where(recall >= target_recall)[0]
    if len(ok):
        # unter allen zulässigen: die mit der besten Precision
        recall_idx = ok[int(np.argmax(precision[ok]))]
        recall_threshold = float(pr_thresholds[recall_idx])
    else:
        recall_threshold = 0.5

    print("\n[THRESHOLD] Wahl der Entscheidungsschwelle")
    print(f"    Positiv/Negativ im Test: {n_pos}/{n_neg}")
    print(f"    (a) Kosten-optimal  (FN={cost_fn:.0f}€, FP={cost_fp:.0f}€): "
          f"thr={cost_threshold:.3f}, erwartete Kosten={costs[best_idx]:.0f}€")
    print(f"    (b) Recall-Ziel >= {target_recall:.2f}: thr={recall_threshold:.3f}")
    print(f"    (Vergleich Standard-Schwelle 0.5 siehe classification_report)")

    return {
        "cost_threshold": cost_threshold,
        "recall_threshold": recall_threshold,
        "pr_thresholds": pr_thresholds,
        "costs": costs,
        "precision": precision,
        "recall": recall,
    }


# ╔══════════════════════════════════════════════════════════╗
# ║                        PLOTS                            ║
# ╚══════════════════════════════════════════════════════════╝

def plot_confusion(y_true, y_pred, cfg: dict, threshold: float) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No (0)", "Yes (1)"])
    ax.set_yticklabels(["No (0)", "Yes (1)"])
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title(f"Confusion Matrix (Schwelle={threshold:.3f})")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "confusion_matrix.png")
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] Confusion Matrix gespeichert: {out}")


def plot_roc(y_true, y_proba, cfg: dict) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, color="C0", label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Zufall")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC-Kurve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "roc_curve.png")
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] ROC-Kurve gespeichert: {out}")


def plot_pr(y_true, y_proba, cfg: dict) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)
    fig = plt.figure(figsize=(5.2, 4.6))
    plt.plot(recall, precision, color="C1", label=f"PR (AP={ap:.3f})")
    plt.axhline(baseline, ls="--", color="grey",
                label=f"Baseline (Churn-Anteil={baseline:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall-Kurve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "precision_recall_curve.png")
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] Precision-Recall-Kurve gespeichert: {out}")


def plot_cost_curve(thr_info: dict, cfg: dict) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    thresholds = thr_info["pr_thresholds"]
    costs = thr_info["costs"]
    best = thr_info["cost_threshold"]
    fig = plt.figure(figsize=(5.6, 4.2))
    plt.plot(thresholds, costs, color="C3")
    plt.axvline(best, ls="--", color="black",
                label=f"Kostenminimum @ {best:.3f}")
    plt.axvline(0.5, ls=":", color="grey", label="Standard 0.5")
    plt.xlabel("Entscheidungsschwelle")
    plt.ylabel("Erwartete Gesamtkosten (€)")
    plt.title("Kosten in Abhängigkeit der Schwelle")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], "threshold_cost.png")
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] Kosten-Kurve gespeichert: {out}")


def plot_feature_importance(names, importances, cfg: dict,
                            title: str, fname: str, top_k: int = 15) -> None:
    _ensure_dir(cfg["PLOT_DIR"])
    order = np.argsort(np.abs(importances))[::-1][:top_k]
    names = np.asarray(names)[order][::-1]
    vals = np.asarray(importances)[order][::-1]
    fig = plt.figure(figsize=(7.2, 5.2))
    colors = ["C0" if v >= 0 else "C3" for v in vals]
    plt.barh(range(len(vals)), vals, color=colors)
    plt.yticks(range(len(vals)), names)
    plt.xlabel("Wichtigkeit / Einfluss")
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(cfg["PLOT_DIR"], fname)
    plt.savefig(out, dpi=120); plt.close(fig)
    print(f"[PLOT] Feature Importance gespeichert: {out}")


# ╔══════════════════════════════════════════════════════════╗
# ║                 EVALUATION & INTERPRETATION            ║
# ╚══════════════════════════════════════════════════════════╝

def evaluate_model(model: Pipeline, X_test, y_test, cfg: dict) -> dict:
    """
    Evaluiert das finale Modell auf dem Testset:
      classification_report, ROC-AUC, Average Precision, Confusion Matrix (@0.5
      und @Business-Schwelle), sowie ROC/PR/Kosten-Plots.
    """
    y_proba = model.predict_proba(X_test)[:, 1]     # P(Churn == Yes)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print("\n[EVAL] Testset-Metriken (schwellenunabhängig):")
    print(f"    ROC-AUC          : {auc:.3f}")
    print(f"    Average Precision: {ap:.3f}")

    # Standard-Schwelle 0.5
    y_pred_05 = (y_proba >= 0.5).astype(int)
    print("\n[EVAL] classification_report @ Schwelle 0.5:")
    print(classification_report(y_test, y_pred_05, target_names=["No", "Yes"],
                                digits=3))

    # Business-Schwelle wählen
    thr_info = choose_threshold(y_test, y_proba, cfg)
    thr = thr_info["cost_threshold"]
    y_pred_thr = (y_proba >= thr).astype(int)
    print(f"\n[EVAL] classification_report @ Kosten-Schwelle {thr:.3f}:")
    print(classification_report(y_test, y_pred_thr, target_names=["No", "Yes"],
                                digits=3))

    # Plots
    plot_roc(y_test, y_proba, cfg)
    plot_pr(y_test, y_proba, cfg)
    plot_cost_curve(thr_info, cfg)
    plot_confusion(y_test, y_pred_thr, cfg, threshold=thr)

    return {"roc_auc": auc, "average_precision": ap, "threshold": thr,
            "y_proba": y_proba}


def get_feature_names(model: Pipeline, cfg: dict) -> list:
    """Liest die (aufgeblähten) Feature-Namen aus dem gefitteten Preprocessor."""
    pre = model.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        # Fallback: rohe Feature-Namen
        return cfg["NUMERIC_FEATURES"] + cfg["CATEGORICAL_FEATURES"]


def explain_model(model: Pipeline, X_test, y_test, cfg: dict,
                  use_shap: bool = False) -> None:
    """
    Interpretierbarkeit:
      - Permutation Importance (modellagnostisch, immer)
      - Koeffizienten (logreg) bzw. feature_importances_ (Bäume), falls vorhanden
      - Optional SHAP (nur wenn installiert und --explain gesetzt)
    """
    print("\n[EXPLAIN] Permutation Importance (auf Testdaten) ...")
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=cfg["RANDOM_SEED"], scoring="roc_auc", n_jobs=-1,
    )
    # Permutation Importance bezieht sich auf die ROHEN Eingabespalten
    raw_names = cfg["NUMERIC_FEATURES"] + cfg["CATEGORICAL_FEATURES"]
    plot_feature_importance(
        raw_names, perm.importances_mean, cfg,
        title="Permutation Importance (Rückgang ROC-AUC)",
        fname="permutation_importance.png",
    )

    # Modell-eigene Wichtigkeiten
    clf = model.named_steps["classifier"]
    feat_names = get_feature_names(model, cfg)
    if hasattr(clf, "feature_importances_"):
        plot_feature_importance(
            feat_names, clf.feature_importances_, cfg,
            title="Feature Importances (Modell-intern)",
            fname="model_feature_importance.png",
        )
    elif hasattr(clf, "coef_"):
        plot_feature_importance(
            feat_names, clf.coef_.ravel(), cfg,
            title="Logistische-Regression-Koeffizienten (Log-Odds)",
            fname="model_coefficients.png",
        )

    if not use_shap:
        return

    # SHAP ist OPTIONAL – keine harte Abhängigkeit.
    try:
        import shap
    except ImportError:
        print("[EXPLAIN] SHAP ist nicht installiert. Überspringe SHAP-Analyse. "
              "(Installation: pip install shap)")
        return
    try:
        print("[EXPLAIN] Berechne SHAP-Werte (kann etwas dauern) ...")
        pre = model.named_steps["preprocessor"]
        X_trans = pre.transform(X_test)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        # kleine Stichprobe, damit es schnell bleibt
        sample = X_trans[:200]
        explainer = shap.Explainer(clf, sample, feature_names=feat_names)
        shap_values = explainer(sample)
        _ensure_dir(cfg["PLOT_DIR"])
        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feat_names,
                          show=False)
        out = os.path.join(cfg["PLOT_DIR"], "shap_summary.png")
        plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
        print(f"[EXPLAIN] SHAP-Summary gespeichert: {out}")
    except Exception as exc:
        print(f"[EXPLAIN] SHAP-Analyse fehlgeschlagen: {exc}")


# ╔══════════════════════════════════════════════════════════╗
# ║              SPEICHERN / LADEN / INFERENZ              ║
# ╚══════════════════════════════════════════════════════════╝

def save_model(model: Pipeline, cfg: dict) -> None:
    _ensure_dir(os.path.dirname(cfg["MODEL_PATH"]))
    joblib.dump(model, cfg["MODEL_PATH"])
    print(f"[SAVE] Modell gespeichert: {cfg['MODEL_PATH']}")


def load_model(cfg: dict) -> Pipeline:
    if not os.path.exists(cfg["MODEL_PATH"]):
        raise FileNotFoundError(
            f"Kein gespeichertes Modell unter {cfg['MODEL_PATH']}. "
            "Bitte zuerst mit --train trainieren."
        )
    print(f"[LOAD] Lade Modell: {cfg['MODEL_PATH']}")
    return joblib.load(cfg["MODEL_PATH"])


def example_customers(cfg: dict) -> pd.DataFrame:
    """Zwei Beispielkunden: einer risikoarm, einer risikoreich."""
    cols = cfg["NUMERIC_FEATURES"] + cfg["CATEGORICAL_FEATURES"]
    rows = [
        {   # treuer Langzeitkunde, Zweijahresvertrag -> geringes Risiko
            "tenure": 65, "MonthlyCharges": 45.0, "TotalCharges": 2900.0,
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "Yes", "Contract": "Two year", "InternetService": "DSL",
            "PaymentMethod": "Bank transfer (automatic)", "PaperlessBilling": "No",
            "TechSupport": "Yes",
        },
        {   # Neukunde, Month-to-month, teurer Fiber-Tarif -> hohes Risiko
            "tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
            "Dependents": "No", "Contract": "Month-to-month",
            "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
            "PaperlessBilling": "Yes", "TechSupport": "No",
        },
    ]
    return pd.DataFrame(rows)[cols]


def run_inference(model: Pipeline, cfg: dict, threshold: float = 0.5) -> None:
    """Zeigt Vorhersagen (Wahrscheinlichkeit + Label) für Beispielkunden."""
    df = example_customers(cfg)
    proba = model.predict_proba(df)[:, 1]
    print("\n[INFER] Vorhersage für Beispielkunden "
          f"(Schwelle={threshold:.3f}):")
    for i, p in enumerate(proba):
        label = "CHURN" if p >= threshold else "bleibt"
        print(f"    Kunde {i+1}: P(Churn)={p:.3f}  ->  {label}  "
              f"(Contract={df.iloc[i]['Contract']}, tenure={df.iloc[i]['tenure']})")


# ╔══════════════════════════════════════════════════════════╗
# ║                          MAIN                            ║
# ╚══════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised-ML-Workshop: Churn-Vorhersage mit scikit-learn"
    )
    p.add_argument("--train", action="store_true",
                   help="Modellvergleich (CV), bestes Modell trainieren & speichern")
    p.add_argument("--evaluate", action="store_true",
                   help="Gespeichertes/frisches Modell auf dem Testset evaluieren")
    p.add_argument("--tune", action="store_true",
                   help="Hyperparameter-Tuning per GridSearchCV")
    p.add_argument("--explain", action="store_true",
                   help="Interpretierbarkeit (Permutation Importance, optional SHAP)")
    p.add_argument("--model", choices=["logreg", "rf", "gb"], default=None,
                   help="Modell erzwingen (überschreibt CV-Auswahl / CONFIG['MODEL'])")
    p.add_argument("--data", type=str, default=None,
                   help="Pfad zu einer eigenen CSV (überschreibt CONFIG['DATA_CSV'])")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)
    if args.data is not None:
        cfg["DATA_CSV"] = args.data
    if args.model is not None:
        cfg["MODEL"] = args.model

    set_seeds(cfg["RANDOM_SEED"])

    if not (args.train or args.evaluate):
        print("Nichts zu tun. Nutze --train und/oder --evaluate "
              "(optional --tune --explain --model {logreg,rf,gb}).")
        print("Beispiel:  python workshop_supervised_ml.py --train --evaluate")
        return

    # 1) Daten laden + Überblick
    df = load_dataset(cfg)
    summarize_dataset(df, cfg)

    # Zielspalte in 0/1 kodieren (Yes -> 1 = Churn)
    target = cfg["TARGET"]
    y = (df[target] == cfg["POSITIVE_LABEL"]).astype(int)
    feature_cols = cfg["NUMERIC_FEATURES"] + cfg["CATEGORICAL_FEATURES"]
    X = df[feature_cols].copy()

    # 2) Stratifizierter Train/Test-Split (Klassenverhältnis bleibt erhalten!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["TEST_SIZE"], stratify=y,
        random_state=cfg["RANDOM_SEED"],
    )
    print(f"\n[SPLIT] Train={len(X_train)}  Test={len(X_test)}  "
          f"(stratify=y, Churn-Anteil Train={y_train.mean():.3f}, "
          f"Test={y_test.mean():.3f})")

    model = None
    if args.train:
        # 3) Modellvergleich per CV -> bestes Modell bestimmen
        comparison = compare_models(X_train, y_train, cfg)
        chosen = cfg["MODEL"] if args.model is not None else comparison["best_name"]
        print(f"\n[TRAIN] Trainiere finales Modell: '{chosen}'")

        # 4) (optional) Tuning, sonst Standard-Pipeline fitten
        if args.tune:
            model = tune_model(chosen, X_train, y_train, cfg)
        else:
            model = build_pipeline(chosen, cfg)
            model.fit(X_train, y_train)

        save_model(model, cfg)
    else:
        # Kein --train: gespeichertes Modell laden
        model = load_model(cfg)

    # 5) Evaluieren
    threshold = 0.5
    if args.evaluate:
        eval_res = evaluate_model(model, X_test, y_test, cfg)
        threshold = eval_res["threshold"]

    # 6) Interpretierbarkeit
    if args.explain:
        explain_model(model, X_test, y_test, cfg, use_shap=True)

    # 7) Inferenz auf Beispielkunden (mit der Business-Schwelle)
    run_inference(model, cfg, threshold=threshold)

    print("\n[FERTIG] Alle Plots liegen unter:", cfg["PLOT_DIR"])


if __name__ == "__main__":
    main()
