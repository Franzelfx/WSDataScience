# Gegenüberstellung: Autoencoder vs. VAE vs. Diffusion Model

## Kurzübersicht

| Kriterium                  | Autoencoder (AE)                          | Variational Autoencoder (VAE)                         | Diffusion Model                                        |
|----------------------------|-------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| **Grundprinzip**           | Komprimieren & Rekonstruieren             | Komprimieren in Wahrscheinlichkeitsverteilung         | Schrittweises Entrauschen von reinem Gaussian-Noise    |
| **Latent Space**           | Punktvektor (deterministisch)             | Normalverteilung μ + σ (stochastisch)                 | Kein expliziter Latent Space; Zeitschritt t als Kontext |
| **Verlustfunktion**        | MSE (Rekonstruktion)                      | MSE + KL-Divergenz                                    | MSE des Denoising-Fehlers pro Zeitschritt              |
| **Training**               | Ein einzelner Vorwärtspass                | Ein einzelner Vorwärtspass                            | Viele Zeitschritte t ∈ [0, T]                         |
| **Sampling / Generierung** | Schwach (kein strukturierter Latent Space)| Möglich (Sample aus N(0,1), decode)                  | Sehr stark (iteratives Entrauschen aus Rauschen)       |
| **Anomalieerkennung**      | Sehr gut geeignet (Rekonstruktionsfehler) | Gut geeignet (Rekonstruktion + ELBO-Score)            | Möglich, aber aufwendig                                |
| **Datenkompression**       | Ja                                        | Ja (weicher, regulierierter Raum)                     | Nein                                                   |
| **Denoising**              | Optional (Denoising AE)                   | Optional                                              | Kern des Prinzips                                      |
| **Modellkomplexität**      | Gering                                    | Mittel                                                | Hoch (viele Denoiser-Iterationen)                      |
| **Rechenaufwand Inferenz** | Sehr schnell (ein Forward-Pass)           | Sehr schnell (ein Forward-Pass)                       | Langsam (10–1000 Schritte)                             |
| **Typische Anwendung**     | Anomalie-Detektion, Kompression           | Generierung, Interpolation, Anomalie-Detektion        | Bild-/Audio-/Tabellen-Generierung                      |

---

## Was lernt jedes Modell?

### Autoencoder
Der Encoder lernt eine komprimierte Repräsentation der Daten:

$$
z = f_\theta(x)
$$

Der Decoder lernt, aus dieser Repräsentation das Original zu rekonstruieren:

$$
\hat{x} = g_\phi(z)
$$

Optimiert wird der MSE:

$$
\mathcal{L}_{\text{AE}} = \frac{1}{N}\sum_{i=1}^{N}\|x_i - \hat{x}_i\|_2^2
$$

Das Modell hat keinen Mechanismus, der erzwingt, dass der Latent Space strukturiert oder dicht belegt ist.

---

### Variational Autoencoder (VAE)
Der Encoder lernt keine einzelnen Punkte, sondern Parameter einer Verteilung:

$$
q_\theta(z | x) = \mathcal{N}(\mu_\theta(x),\, \sigma_\theta^2(x))
$$

Ein Latent-Vektor wird dann per Reparametrisierungstrick gesampelt:

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Die Verlustfunktion ist die Evidence Lower Bound (ELBO):

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}[\|x - \hat{x}\|^2]}_{\text{Rekonstruktion}} + \underbrace{D_{\text{KL}}(q_\theta(z|x) \| \mathcal{N}(0,I))}_{\text{Regularisierung}}
$$

Der KL-Term zwingt den Latent Space, nahe an einer Standardnormalverteilung zu bleiben — das macht Interpolation und Sampling aus dem Latent Space möglich.

---

### Diffusion Model
Kein Encoder/Decoder im klassischen Sinne. Stattdessen:

**Forward Process** (kein lernbarer Parameter): schrittweises Verrauschen der Daten:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I)
$$

**Reverse Process** (lernbar): ein zeitkonditionierter Denoiser lernt, den Rauschanteil vorherzusagen:

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

Verlust:

$$
\mathcal{L}_{\text{Diff}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

Neue Samples werden erzeugt, indem aus reinem Rauschen $x_T \sim \mathcal{N}(0, I)$ gestartet und schrittweise entstört wird.

---

## Intuitive Analogien

| Modell       | Analogie                                                                                     |
|--------------|----------------------------------------------------------------------------------------------|
| Autoencoder  | Foto hochkomprimieren (JPEG) und wieder dekomprimieren — was man nicht gesehen hat, fehlt.   |
| VAE          | Wie AE, aber mit einem Wörterbuch von Konzepten statt einzelner Pixelmuster.                 |
| Diffusion    | Skulptur aus einem Marmorblock: beginnt mit chaotischem Rauschen und meißelt schrittweise.   |

---

## Wann nimmt man was?

| Situation                                          | Empfehlung                  |
|----------------------------------------------------|-----------------------------|
| Anomalie-Detektion auf tabellarischen Daten        | **Autoencoder** (dieser Workshop) |
| Generierung ähnlicher Datenpunkte / Interpolation  | **VAE**                     |
| Hochwertige Bild-/Audio-Generierung                | **Diffusion Model**         |
| Sehr begrenzte Rechenressourcen                    | **AE oder VAE**             |
| Denoising als Selbstzweck                          | **Denoising AE oder Diffusion** |
| Strukturierter, interpretierbarer Latent Space     | **VAE**                     |
