# Decisiones en la nube con Gemini Decision 1

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · **Español** · [Français](../fr/cloud-decisions.md) · [日本語](../ja/cloud-decisions.md) · [Deutsch](../de/cloud-decisions.md)

`gemini-decision-1` de Google responde preguntas de System One (`noul`, `choice`, `score`) en una sola pasada sobre texto, imágenes, audio, vídeo y archivos PDF. Su formato de transmisión coincide con el propio `/v1/systemone` de basalt, por lo que funciona como una etapa en la nube delante de un modelo de decisión local:

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- Las preguntas dudosas (confianza inferior a `--decision-escalate-below`), las llamadas fallidas y los límites de tasa pasan al modelo local.
- `/v1/systemone` acepta `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` junto a `state`. Los medios se rechazan cuando ninguna etapa puede leerlos.
- En un archivo compose, el modelo es `source: cloud:gemini/gemini-decision-1` con `api_key_env: GEMINI_API_KEY`.
  - Los modelos Gemini solo atienden cascadas de decisión.
  - Los pipelines mantienen el texto percibido en este Mac a menos que ese modelo establezca `data: media`. En ese caso, los propios archivos adjuntos y su texto percibido se envían a Google para la decisión.
- El coste se contabiliza a partir del uso de tokens que informa la API, a $0.10 por millón de tokens de entrada, y computa dentro de los presupuestos de compose.

## Un profesor de calibración

Gemini Decision 1 devuelve probabilidades calibradas, no solo etiquetas. Google indica que lidera en calibración de probabilidades. Eso lo convierte en un profesor de calibración ideal para los modelos de decisión locales de basalt: ejecuta los mismos estados a través de él y de una lectura local, y ajusta la temperatura local, o entrena una cabeza OpenJev, frente a sus probabilidades. De este modo, la confianza del decisor local significa lo mismo que la de la nube, y los umbrales de escalado se mantienen equivalentes. La integración de esto en el paso de calibración está planificada junto con el trabajo local de JEV del #423 (J6: lectura de calibración desde el paquete).
