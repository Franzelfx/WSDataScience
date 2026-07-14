# Workshop 01 · Explorative Datenanalyse (EDA) – Anleitung & Erklärung

Diese Anleitung begleitet dich durch den Workshop im Notebook. Sie erklärt die
**Konzepte** hinter der explorativen Datenanalyse, führt **Schritt für Schritt**
durch die einzelnen Aufgaben und gibt Hinweise darauf, **worauf du bei den Lücken
achten musst** und **wie du die Ergebnisse und Visualisierungen interpretierst**.

Kein Code-Dump: Die vollständigen Code-Lösungen findest du im Notebook
`loesung/workshop.ipynb`. Hier geht es ums Verstehen.

---

## Worum geht es? – Explorative Datenanalyse (EDA)

Die **explorative Datenanalyse (EDA)** ist der erste Schritt in jedem
Data-Science-Projekt – noch bevor ein einziges Modell trainiert wird. Das Ziel
ist, die Daten wirklich zu **verstehen**, statt blind Algorithmen darauf
loszulassen. Typische Fragen der EDA:

- Wie sehen die Daten aus? Welche Spalten (Merkmale) gibt es und welchen Typ haben sie?
- Wie viele Beobachtungen (Zeilen) gibt es?
- Fehlen Werte? Wo und wie viele?
- Wie sind einzelne Merkmale verteilt (typische Werte, Streuung, Ausreißer)?
- Welche Zusammenhänge (Korrelationen) stecken zwischen den Merkmalen?
- Unterscheiden sich Gruppen voneinander?

Wer diese Fragen sauber beantwortet, trifft später bessere Entscheidungen bei
Datenbereinigung, Feature-Auswahl und Modellwahl. EDA ist damit weniger eine
Technik als eine **Denkhaltung**: neugierig, kritisch, immer mit einer Frage im Kopf.

### Der Datensatz: Titanic

Wir arbeiten mit dem klassischen **Titanic-Datensatz** (in `seaborn` eingebaut,
891 Passagiere). Jede Zeile ist ein Passagier, jede Spalte ein Merkmal. Die
wichtigsten Spalten:

| Spalte | Bedeutung |
|---|---|
| `survived` | 0 = nicht überlebt, 1 = überlebt (Zielgröße) |
| `pclass` / `class` | Passagierklasse als Zahl (1–3) bzw. Text (First/Second/Third) |
| `sex` | Geschlecht |
| `age` | Alter in Jahren |
| `fare` | Ticketpreis |
| `embarked` / `embark_town` | Einschiffungshafen (S/C/Q bzw. Ortsname) |
| `deck` | Deck (viele fehlende Werte) |
| `alone` | Reiste die Person allein? |

### Werkzeuge

| Bibliothek | Zweck |
|---|---|
| **pandas** | Daten laden, filtern, aggregieren – das „Excel für Python" |
| **matplotlib** | Basis-Bibliothek für Diagramme |
| **seaborn** | Statistische Visualisierungen mit schönen Defaults (baut auf matplotlib auf) |

---

## So arbeitest du im Notebook

- **Markdown-Zellen** erklären das jeweilige Konzept.
- **Code-Zellen** sind teils vorgegeben, teils enthalten sie ein `# TODO`.
- Führe die Zellen **von oben nach unten** aus (Shift + Enter). Die Reihenfolge
  ist wichtig, weil spätere Zellen auf Ergebnissen früherer aufbauen (z. B. wird
  `df` einmal geladen und dann durchgehend verwendet).
- Wenn etwas schiefläuft: Kernel neu starten und „Run All" bis zur aktuellen Zelle.

---

## Ablauf der Aufgaben – Schritt für Schritt

### 1–2. Bibliotheken importieren & Daten laden

Nur ausführen. `sns.load_dataset("titanic")` liefert einen **DataFrame** (`df`) –
eine Tabelle. Mit `df.shape` prüfst du (Zeilen, Spalten). Merke dir: 891 Zeilen,
15 Spalten. Das ist dein Ausgangspunkt.

### 3. Erste Eindrücke: `.head()` / `.tail()`

`.head()` zeigt die ersten 5 Zeilen, `.tail()` die letzten 5.

- **Lücke:** Statt `.head()` sollst du `.tail()` aufrufen.
- **Worauf achten:** Beides sind Methoden – die Klammern `()` nicht vergessen.
  Ein schneller Blick auf Kopf und Schwanz verrät oft schon, ob Daten sortiert
  sind oder ob am Ende „krumme" Werte stehen.

### 4. Datentypen: `.info()` / `.dtypes`

Jede Spalte hat einen **Datentyp**: `int64`/`float64` (Zahlen), `object` (Text),
`bool` (wahr/falsch), `category` (Kategorien). Das ist entscheidend, weil man mit
Zahlen rechnen kann, mit Text nicht.

- **Lücke:** `df.dtypes` ausgeben.
- **Worauf achten:** `.info()` liefert zusätzlich die Anzahl der **nicht-leeren**
  Werte pro Spalte – ein erster Hinweis auf fehlende Daten (siehe Schritt 6).

### 5. Deskriptive Statistik: `.describe()`, `.mean()`, `.median()`

`.describe()` berechnet für jede numerische Spalte: `count`, `mean`, `std`
(Standardabweichung = Streuung), `min`/`max` und die Quartile (25 %, 50 % = Median,
75 %).

- **Lücken:**
  - `durchschnittspreis = df["fare"].mean()`
  - `medianalter = df["age"].median()`
- **Worauf achten:** Der Zugriff auf eine Spalte erfolgt über
  `df["spaltenname"]`, danach die Kennzahl-Methode. Achte darauf, dass die
  Zuweisung vor dem `print` steht, sonst schlägt die f-String-Formatierung fehl.
- **Interpretationshinweis – warum Median statt Mittelwert?** Der **Median** ist
  robust gegen **Ausreißer**. Beim `fare` gibt es wenige sehr teure Tickets, die
  den Mittelwert nach oben ziehen; der Median beschreibt den „typischen"
  Passagier besser. Genau deshalb füllen wir gleich fehlende Alterswerte mit dem
  Median auf.

### 6. Fehlende Werte (NaN)

Reale Datensätze sind selten vollständig. `df.isnull().sum()` zählt fehlende
Werte je Spalte.

- **Lücke:** Anteil in Prozent berechnen:
  `fehlende_prozent = (df.isnull().sum() / len(df)) * 100`.
- **Worauf achten:** `len(df)` ist die Gesamtzahl der Zeilen. Die Division wird
  spaltenweise auf die ganze Serie angewendet – kein Loop nötig.
- **Interpretationshinweis:** Erwartetes Ergebnis – `age` fehlt zu ~20 %,
  `deck` zu ~77 %, `embarked`/`embark_town` je ~0,2 %. Faustregel: Spalten mit
  sehr vielen Lücken (wie `deck`) sind oft unbrauchbar; wenige Lücken kann man
  auffüllen oder löschen.
- **Danach (vorgegeben):** Fehlendes `age` wird mit dem **Median** aufgefüllt
  (`fillna`). Achtung: Ab hier ist `df["age"]` verändert – deshalb Reihenfolge
  einhalten. Nach dem Auffüllen fehlen in `age` 0 Werte.

### 7. Histogramm – Verteilungen

Ein **Histogramm** teilt den Wertebereich in Balken (bins) und zählt, wie viele
Beobachtungen in jeden Bereich fallen. Es beantwortet: „Wie sind die Werte
verteilt?"

- **Lücke:** Histogramm für `fare` erstellen – analog zum vorgegebenen
  Alters-Histogramm, aber `x="fare"`, plus passender Titel und x-Achsenbeschriftung.
- **Worauf achten:** `data=df` und `x="fare"` als Argumente; `kde=True`
  zeichnet zusätzlich eine geglättete Dichtekurve.
- **Interpretationshinweis:** Die Altersverteilung ist annähernd „glockenförmig"
  mit Schwerpunkt bei ~20–35 Jahren. Die `fare`-Verteilung ist stark
  **rechtsschief**: sehr viele billige Tickets, wenige sehr teure – ein Musterbeispiel
  dafür, warum hier der Median aussagekräftiger ist als der Mittelwert.

### 8. Balkendiagramme – Kategorien vergleichen

Zwei Varianten:
- `sns.countplot` **zählt** Vorkommen je Kategorie (vorgegeben: Passagiere pro Klasse).
- `sns.barplot` mit `y="survived"` zeigt automatisch den **Mittelwert** je
  Kategorie – und weil `survived` nur 0/1 ist, ist dieser Mittelwert exakt die
  **Überlebensrate**.

- **Lücke:** `sns.barplot(data=df, x="sex", y="survived", palette="Set2")`.
- **Worauf achten:** Verwechsle nicht `countplot` (nur `x`) und `barplot`
  (braucht `x` **und** `y`). Der kleine Fehlerbalken zeigt die Unsicherheit
  (Konfidenzintervall).
- **Interpretationshinweis:** Frauen überlebten deutlich häufiger als Männer –
  der erste klare Hinweis auf „Frauen und Kinder zuerst".

### 9. Boxplot

Ein **Boxplot** zeigt kompakt: Median (Strich), die mittleren 50 % der Daten
(Box = Interquartilsabstand), die Spannweite (Whiskers) und **Ausreißer**
(einzelne Punkte).

- **Lücke:** Boxplot Alter nach Überlebensstatus: `x="survived"`, `y="age"`.
- **Worauf achten:** Reihenfolge der Achsen – die Kategorie (`survived`) auf `x`,
  die numerische Größe (`age`) auf `y`.
- **Interpretationshinweis:** Der Median-Altersunterschied zwischen Überlebenden
  und Nicht-Überlebenden ist klein, aber am unteren Rand (Kinder) zeigt sich ein
  Überlebensvorteil. Im Preis-Boxplot (vorgegeben) sieht man dagegen sehr
  deutlich: 1. Klasse zahlte viel mehr und hat viele Ausreißer nach oben.

### 10. Korrelation & Heatmap

Die **Korrelation** misst den linearen Zusammenhang zweier numerischer Spalten:
**+1** = starker gleichläufiger Zusammenhang, **0** = keiner, **−1** =
gegenläufig. Eine **Heatmap** stellt die ganze Korrelationsmatrix farblich dar.

- **Lücke:** Stärkste Zusammenhänge mit `survived` finden:
  `korrelation["survived"].sort_values(ascending=False)`.
- **Worauf achten:** Die Matrix `korrelation` wurde in der Zelle davor nur aus
  **numerischen** Spalten berechnet (`select_dtypes(include=["number"])`).
- **Interpretationshinweis:** `fare` korreliert am stärksten **positiv**
  (teureres Ticket → höhere Überlebenschance), `pclass` am stärksten **negativ**
  (höhere Klassenzahl = 3. Klasse → geringere Chance). Wichtig: **Korrelation ist
  nicht Kausalität** – der Preis „rettet" niemanden, er ist ein Stellvertreter
  für die Klasse und damit die Kabinenlage/den Zugang zu Rettungsbooten.

### 11. Gruppenvergleiche – `groupby`

`groupby` gruppiert Zeilen nach einer Kategorie und berechnet je Gruppe eine
Kennzahl – wie eine Pivot-Tabelle in Excel.

- **Lücken:**
  - Überlebensrate nach Geschlecht: `df.groupby("sex")["survived"].mean() * 100`
  - Nach Klasse **und** Geschlecht: `df.groupby(["class", "sex"])["survived"].mean() * 100`
- **Worauf achten:** Für **mehrere** Gruppierungsspalten eine **Liste**
  übergeben: `["class", "sex"]`. Die Multiplikation mit 100 macht aus dem Anteil
  eine Prozentangabe.
- **Interpretationshinweis:** Die Kombination zeigt die extremen Unterschiede:
  Frauen in der 1. Klasse überlebten zu ~97 %, Männer in der 3. Klasse nur zu
  einem Bruchteil davon.

### 12. Kombinierte Visualisierung – `hue`

Mit dem Parameter `hue` fügst du eine **dritte Dimension als Farbe** hinzu.

- **Lücke:** Altershistogramm nach Überlebensstatus:
  `sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True)`.
- **Worauf achten:** `hue="survived"` teilt die Balken nach Überlebensstatus ein.
- **Interpretationshinweis:** Bei den jüngsten Passagieren (Kinder) überwiegen
  die Überlebenden – erneut das Muster „Kinder zuerst".

### 13. Zusammenfassung

- **Lücke:** Gesamtüberlebensrate: `df["survived"].mean() * 100` (~38,4 %).
- Danach beantwortest du in der Markdown-Zelle die Reflexionsfragen. Erwartete
  Kernergebnisse: 891 Passagiere; große Lücken bei `deck` und `age`;
  Gesamtüberlebensrate ~38 %; höchste Überlebensrate bei Frauen der 1. Klasse;
  stärkste Zusammenhänge über `fare`/`pclass`.

### Bonus (optional)

- **Bonus 1:** Mit `pd.cut` **Altersgruppen** bilden (binning) und deren
  Überlebensrate plotten – Kinder schneiden am besten ab.
- **Bonus 2:** Ein **Pairplot** über `age`, `fare`, `survived` zeigt alle
  paarweisen Streudiagramme auf einmal; höhere Ticketpreise gehen mit häufigerem
  Überleben einher.

---

## Häufige Stolperfallen

- **Zellen-Reihenfolge:** Läuft `df` „nicht wie erwartet", wurde vermutlich eine
  frühere Zelle (z. B. das `fillna` in Schritt 6) übersprungen. Kernel neu starten.
- **Methode vs. Attribut:** `.dtypes` ohne Klammern, `.mean()`/`.median()`/
  `.head()` mit Klammern.
- **`countplot` vs. `barplot`:** Ersteres zählt (nur `x`), letzteres aggregiert
  (`x` und `y`).
- **Liste bei mehrfachem groupby:** `groupby(["class", "sex"])`, nicht
  `groupby("class", "sex")`.
- **Korrelation ≠ Kausalität:** Zusammenhänge beschreiben, nicht vorschnell als
  Ursache deuten.

## Das nimmst du mit

| Konzept | pandas / seaborn |
|---|---|
| Daten laden | `sns.load_dataset()`, `pd.read_csv()` |
| Erste Eindrücke | `.head()`, `.tail()`, `.info()`, `.dtypes` |
| Statistik | `.describe()`, `.mean()`, `.median()` |
| Fehlende Werte | `.isnull().sum()`, `.fillna()` |
| Verteilungen | `sns.histplot()`, `sns.boxplot()` |
| Kategorien | `sns.countplot()`, `sns.barplot()` |
| Zusammenhänge | `.corr()`, `sns.heatmap()` |
| Gruppen | `.groupby().mean()` |
| Extra-Dimension | `hue="spalte"` |

[← Zur Workshop-Übersicht](../../README.md)
