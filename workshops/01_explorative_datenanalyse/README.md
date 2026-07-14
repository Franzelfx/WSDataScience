# Workshop 01 · Explorative Datenanalyse (EDA)

Einstieg in die explorative Datenanalyse mit pandas, matplotlib und seaborn –
untersucht Schritt für Schritt den Titanic-Datensatz und macht Muster,
fehlende Werte und Zusammenhänge sichtbar.

## Lernziele
- Daten mit pandas laden, überblicken und Datentypen sowie Struktur verstehen
- Deskriptive Statistik lesen und fehlende Werte erkennen und sinnvoll behandeln
- Verteilungen und Ausreißer mit Histogramm und Boxplot visualisieren
- Kategorien und Gruppen vergleichen (`countplot`, `barplot`, `groupby`)
- Zusammenhänge über Korrelation und Heatmap deuten – und Korrelation von Kausalität unterscheiden

## Struktur
| Pfad | Inhalt |
|------|--------|
| [`workshop.md`](workshop.md) | Ausführliche Anleitung & Erklärung |
| [`aufgabe/`](aufgabe/) | Teilnehmer-Version mit Lücken |
| [`loesung/`](loesung/) | Musterlösung |
| [`requirements.txt`](requirements.txt) | Benötigte Python-Pakete |

## Setup
Python 3.9+ vorausgesetzt. Abhängigkeiten installieren (idealerweise in einer
virtuellen Umgebung):

```bash
cd workshops/01_explorative_datenanalyse
pip install -r requirements.txt
```

Basis-Pakete: **pandas**, **matplotlib**, **seaborn**, **jupyter**. Der
Titanic-Datensatz ist in seaborn eingebaut – kein separater Download nötig.

## Ausführen
Jupyter starten und das Aufgaben-Notebook öffnen:

```bash
jupyter notebook aufgabe/workshop.ipynb
```

Alternativ `jupyter lab` verwenden. Die Zellen von oben nach unten ausführen
(Shift + Enter) und die `# TODO`-Stellen ergänzen. Die Musterlösung liegt unter
[`loesung/workshop.ipynb`](loesung/workshop.ipynb).

[← Zur Workshop-Übersicht](../../README.md)
