# Workshops Data Science

Sammlung praxisnaher Hands-on-Workshops rund um Data Science und Machine Learning
(Deutsch). Jeder Workshop besteht aus einer Anleitung (`workshop.md`) und
lauffähigem Starter-Code mit zentralem `CONFIG`-Block zum Experimentieren.

## Inhalte

| Ordner                                            | Thema                                  | Methode                         |
|---------------------------------------------------|----------------------------------------|---------------------------------|
| [`explorative_datenanalyse/`](explorative_datenanalyse/) | Explorative Datenanalyse (EDA)   | pandas, Visualisierung          |
| [`image_classification/`](iamge_classification/)  | Bildklassifikation                     | CNN (TensorFlow/Keras)          |
| [`autoencoder/`](autoencoder/)                    | Unüberwachtes Lernen / Anomalien       | Autoencoder (TensorFlow/Keras)  |
| [`reinforcement_learning/`](reinforcement_learning/) | **Reinforcement Learning**          | Q-Learning, SARSA, DQN          |

## Schnellstart

Jeder Workshop ist eigenständig. Beispiel Reinforcement Learning:

```bash
cd reinforcement_learning
pip install -r requirements.txt
python workshop_reinforcement_learning.py --train --evaluate --render
```

Die jeweilige `workshop.md` enthält Lernziele, Aufgaben und alle Parameter.
