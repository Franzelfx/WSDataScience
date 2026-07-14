# Daten – Telco Customer Churn

Dieser Ordner enthält den Datensatz für den Workshop.

## Synthetische Daten (Standard)

Beim **ersten Lauf** des Skripts wird – falls noch keine Datei
`telco_churn.csv` existiert – **automatisch ein realistischer, synthetischer
Churn-Datensatz erzeugt** und hier als `telco_churn.csv` gespeichert
(ca. 6000 Zeilen, fester Zufalls-Seed = reproduzierbar).

Die Daten sind bewusst so gebaut, dass sie **echtes Signal** enthalten:

- kurze `tenure` (Vertragsdauer), `Contract = Month-to-month` und hohe
  `MonthlyCharges` erhöhen das Abwanderungsrisiko,
- lange Bindung, `Two year`-Vertrag und aktiver `TechSupport` senken es.

Die Zielspalte `Churn` (`Yes`/`No`) ist **unbalanciert** (rund 26 % `Yes`) –
genau wie in echten Churn-Problemen. Neukunden (`tenure = 0`) haben absichtlich
einen fehlenden `TotalCharges`-Wert, damit die Imputation in der Pipeline geübt
werden kann.

Solange die CSV existiert, wird sie in allen weiteren Läufen wiederverwendet.
Zum Neu-Erzeugen einfach `telco_churn.csv` löschen.

## Optional: den echten IBM-Datensatz verwenden

Wenn du mit echten Daten arbeiten möchtest, kannst du den bekannten
**IBM „Telco Customer Churn"**-Datensatz verwenden (frei verfügbar über
Kaggle bzw. die IBM-Sample-Data-Sammlung). Lege die Datei einfach unter

```
data/telco_churn.csv
```

ab (oder gib einen eigenen Pfad via `--data pfad/zu/datei.csv` an). Erwartet
werden u. a. die Spalten `tenure`, `MonthlyCharges`, `TotalCharges`,
`Contract`, `InternetService`, `PaymentMethod`, `gender`, `SeniorCitizen`,
`Partner`, `Dependents`, `PaperlessBilling`, `TechSupport` sowie die Zielspalte
`Churn` (`Yes`/`No`). Der Code funktioniert mit beiden Varianten identisch.

> Hinweis: `data/telco_churn.csv` wird beim Lauf angelegt und ist nicht Teil
> des Repos.
