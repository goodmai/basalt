# Décisions dans le cloud avec Gemini Decision 1

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · [Español](../es/cloud-decisions.md) · **Français** · [日本語](../ja/cloud-decisions.md) · [Deutsch](../de/cloud-decisions.md) · [한국어](../ko/cloud-decisions.md) · [Bahasa Indonesia](../id/cloud-decisions.md)

Le modèle `gemini-decision-1` de Google répond aux questions System One (`noul`, `choice`, `score`) en une seule passe sur du texte, des images, de l'audio, de la vidéo et des PDF. Son format d'échange correspond à celui de `/v1/systemone` dans basalt, ce qui lui permet de fonctionner comme étape cloud en amont d'un modèle de décision local :

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- Les questions incertaines (confiance inférieure à `--decision-escalate-below`), les appels en échec et les limitations de débit basculent vers le modèle local.
- `/v1/systemone` accepte `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` à côté de `state`. Les médias sont refusés lorsqu'aucune étape ne peut les lire.
- Dans un fichier compose, le modèle s'écrit `source: cloud:gemini/gemini-decision-1` avec `api_key_env: GEMINI_API_KEY`.
  - Les modèles Gemini ne servent qu'aux cascades de décision.
  - Les pipelines conservent le texte perçu sur ce Mac, sauf si ce modèle définit `data: media`. Dans ce cas, les fichiers joints eux-mêmes, ainsi que leur texte perçu, sont envoyés à Google pour la décision.
- Le coût est calculé à partir de la consommation de tokens renvoyée par l'API, à $0.10 par million de tokens en entrée, et est comptabilisé dans les budgets compose.

## Un enseignant de calibration

Gemini Decision 1 renvoie des probabilités calibrées, et pas seulement des étiquettes. Google indique qu'il est en tête sur la calibration des probabilités. Cela en fait un enseignant de calibration idéal pour les modèles de décision locaux de basalt : passez les mêmes états à travers ce modèle et à travers une lecture locale, puis ajustez la température locale, ou entraînez une tête OpenJev, sur ses probabilités. La confiance du décideur local prend alors le même sens que celle du cloud, et les seuils d'escalade restent cohérents. L'intégration de ce mécanisme à l'étape de calibration est prévue avec les travaux JEV locaux de #423 (J6 : lecture de la calibration depuis le bundle).
