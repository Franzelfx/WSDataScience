# Workshop: Explorative Datenanalyse (EDA)

## Überblick

Einstiegs-Workshop in die **Data-Science-Grundlagen**: Du lernst, einen unbekannten
Datensatz systematisch zu erkunden, zu bereinigen und zu visualisieren – bevor
überhaupt ein Modell trainiert wird. Als Beispiel dient der klassische
**Titanic-Datensatz** 🚢.

Der Workshop ist als **Jupyter-Notebook** aufgebaut: Jede Aufgabe wird durch ein
Konzept eingeleitet (DataFrame, fehlende Werte, Histogramm, Boxplot, Korrelation …)
und anschließend praktisch umgesetzt.

---

## Lernziele

- Daten mit **pandas** laden und ihre Struktur verstehen (`head`, `info`, `describe`)
- **Datentypen** und **fehlende Werte (NaN)** erkennen und behandeln
- **Deskriptive Statistik** (Mittelwert, Median, Quantile) interpretieren
- Verteilungen und Kategorien mit **matplotlib / seaborn** visualisieren
  (Histogramm, Balkendiagramm, Boxplot)
- **Zusammenhänge** über Korrelationen und Gruppenvergleiche aufdecken
- Erkenntnisse aus Daten ableiten und zusammenfassen

---

## Voraussetzungen

- **Python 3.10+** und Jupyter

```bash
pip install pandas matplotlib seaborn jupyter
```

> Der Titanic-Datensatz wird über seaborn (`sns.load_dataset("titanic")`) geladen
> und benötigt beim ersten Aufruf eine Internetverbindung.

---

## Dateien

| Datei                                                  | Beschreibung                                         |
|--------------------------------------------------------|------------------------------------------------------|
| [`workshop_teilnehmer.ipynb`](workshop_teilnehmer.ipynb) | **Teilnehmer-Notebook** mit Aufgaben und Lücken      |
| [`workshop_loesung.ipynb`](workshop_loesung.ipynb)       | **Musterlösung** inkl. Bonus-Aufgaben (für Dozenten) |

---

## Ausführen

```bash
cd explorative_datenanalyse
jupyter notebook workshop_teilnehmer.ipynb
```

Arbeite die Zellen von oben nach unten durch und fülle die markierten Stellen aus.
Bei Bedarf kannst du deine Ergebnisse mit der Lösung abgleichen.

---

## Inhalt (Notebook-Kapitel)

1. Bibliotheken importieren
2. Daten laden
3. Erste Eindrücke – Daten anschauen (`head` / `tail`)
4. Datentypen und Struktur verstehen
5. Deskriptive Statistik
6. Fehlende Werte erkennen
7. Visualisierung – Verteilungen (Histogramm)
8. Visualisierung – Kategorien vergleichen (Balkendiagramm)
9. Visualisierung – Boxplot
10. Zusammenhänge erkennen – Korrelation
11. Gruppenvergleiche
12. Kombinierte Visualisierung
13. Zusammenfassung und Erkenntnisse

---

## Weiterführende Workshops

Nach der EDA folgt das Modellieren:
[Bildklassifikation](../iamge_classification/) ·
[Autoencoder](../autoencoder/) ·
[Reinforcement Learning](../reinforcement_learning/)
