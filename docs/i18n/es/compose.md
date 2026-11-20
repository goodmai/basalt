# basalt compose

> Languages: [English](../../compose.md) · [Русский](../ru/compose.md) · [中文](../zh/compose.md) · **Español** · [Français](../fr/compose.md) · [日本語](../ja/compose.md) · [Deutsch](../de/compose.md)

`basalt compose` sirve un sistema de modelos descrito en un único archivo YAML. El archivo enumera los modelos locales y en la nube, así como la forma en que cada uno se mantiene en memoria. También describe los pipelines que enrutan una solicitud entre ellos. Cada pipeline ejecuta primero la percepción (OCR, voz), luego la decisión (System One) y, por último, la respuesta (un modelo de chat). Se trata de una mezcla de modelos, al estilo de un archivo docker-compose para modelos.

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` inicia `basalt serve` con los flags en los que se expande el archivo; los imprime al arrancar. Expone las mismas superficies REST y MCP. Los flags después de `--` se pasan a serve, por ejemplo `basalt compose up -f basalt.yaml -- --port 8090`.

El servidor enlaza primero su puerto y carga los modelos después. Mientras cargan, `compose ps` los muestra como `loading`, `GET /v1/health/ready` responde 503 y el registro de ejecución ya existe, de modo que un segundo `compose up` encuentra el sistema en lugar de iniciar otro. Una vez cargados los modelos, cada uno recibe una sonda de arranque:
- el modelo de chat y cada modelo de respuesta local generan un token;
- cada modelo de decisión local responde una pregunta;
- los archivos de un modelo bajo demanda se comprueban sin cargarlo.

Un modelo que falla su sonda aparece como `failed`, con el motivo, en `compose ps`. `/v1/health/ready` permanece en 503 hasta que un `compose up` reemplaza el modelo. Un modelo que `compose up` carga o recarga recibe la misma sonda. Con `--mcp`, los modelos se cargan antes de que arranque el servidor, como antes.

## El archivo

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### Modelos
- **`source`:** un repositorio de Hugging Face, una ruta local (`./`, `/` o `~`), `cloud:openrouter/<model>`, o `cloud:gemini/<model>` para los modelos de decisión de Google como `gemini-decision-1` (solo cascadas de decisión, clave habitualmente en `GEMINI_API_KEY`).
- **`keep`** se aplica a modelos locales:
  - `resident` es el valor predeterminado y permanece cargado;
  - `on-demand` se carga con la primera solicitud que necesita el modelo y se descarga tras `idle_unload`. Las duraciones tienen el formato `30s`, `5m` o `1h`, y el valor predeterminado es 120 s;
  - `sequential` carga un perceptor solo para su etapa. En el pipeline predeterminado también puede descargar el modelo de chat alrededor del perceptor, si ambos no caben juntos.
- **Modelos en la nube:**
  - Indica la variable de entorno que contiene la clave mediante `api_key_env`. Se rechaza escribir una clave en el archivo, al igual que la interpolación `${…}`.
  - `budget: { usd_per_day }` limita el gasto diario de un modelo.
  - `data: no-perceived` es el valor predeterminado: el texto leído de documentos o audio nunca se envía a un modelo en la nube.
  - `data: media`, solo para modelos de decisión `cloud:gemini`, envía los propios archivos adjuntos de un pipeline (PDF, imágenes, audio en formato WAV) y su texto percibido a ese modelo para la decisión. Úsalo únicamente con documentos que puedan salir de este Mac.

### Memoria y plan
`compose plan` suma los bytes residentes de cada modelo local. Cuenta un modelo bajo demanda como residente, porque puede cargarse en cualquier momento.

- `memory: resident` requiere que la suma quepa en el conjunto de trabajo.
- `memory: sequential` solo necesita la etapa más grande más 2 GB.
- `memory: auto` elige `resident` cuando cabe.

Un sistema que no cabe es rechazado con las cifras exactas antes de cargar nada. Un modelo que no está descargado se notifica como de tamaño desconocido.

### Decisión (`decide`)
- **Orden:** `cascade` enumera los modelos de decisión en orden. Cada etapa responde a las preguntas sobre las que tiene confianza (al menos `escalate_below`) y pasa el resto a la siguiente.
- **Etapas en cualquier orden:** hasta cuatro etapas, en la nube y locales en cualquier orden. Un ejemplo es un juez local pequeño primero, un modelo en la nube después y un modelo local más grande al final: `[laya, jev, chat]`.
  - La última etapa es local.
  - Cada etapa local se mantiene como un motor de decisión: el modelo de chat, leído como `self`, o un modelo de decisión como un paquete de Laya Core AI o un paquete de OpenJev. Un modelo de decisión local dedicado puede establecer `keep: on-demand` (y `idle_unload`) para cargarse en la primera decisión que le llegue y descargarse cuando esté inactivo.
  - Cada pipeline nombra sus propias etapas. Un modelo de decisión nombrado por varios pipelines se carga una sola vez.
  - Un modelo de decisión solo decide; únicamente el modelo de chat responde y decide.
- **Calibración:** un modelo de decisión por lectura (`readout`) toma sus temperaturas de un archivo `basalt_readout.json` en su carpeta (`{"temperature": …, "noul_temperature": …, "noul_bias": …}`), si existe. De lo contrario, usa los valores conocidos para su nombre, y si no los hay, ninguno. #448 planea ajustar estos archivos frente a Gemini Decision 1.
- **Tamaños:** `compose plan` calcula el tamaño de un paquete de cabezal OpenJev como su modelo base más su adaptador, porque el LoRA se aplica sin fusionar sobre la base tal como está almacenada, incluido 4-bit. El plan de `serve` hace lo mismo.
  - Medido en OpenJev-0.6B: 100 estados etiquetados, 300 preguntas, M4 Pro, commit 066da0a. La referencia es la ruta `float32` fusionada, que coincide con PyTorch dentro de 1e-4 (#406).

    | Base, LoRA sin fusionar | Precisión | Misma respuesta superior que float32 | Memoria activa | Mediana por estado |
    |---|---|---|---|---|
    | float32 fusionado (referencia) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, tal como lo nombra el paquete (valor predeterminado) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - Una base de 4-bit le cuesta a este modelo de 0.6B 9 puntos, mientras que 8-bit mantiene su precisión con una cuarta parte de la memoria. `serve` y `compose` cargan la base que nombra el paquete, por lo que solo se accede a una base cuantizada directamente mediante `JevEvaluator`.
  - Vuelve a ejecutarlo con `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms`.
- **Un juez de 9B junto a Gemma 4 12B en 24 GB:** medido con `openjev/OpenJev-Flash-9B-MLX` (8-bit, 9.5 GB) y `openjev/OpenJev-Flash-9B-MLX-4bit` (5.9 GB). Gemma 4 12B (`gemma-4-12B-it-qat-4bit`) responde a las escalaciones. El conjunto consta de 60 estados etiquetados, 180 decisiones, en un M4 Pro de 24 GB en 8294bf7.
  - Usa el juez de 4-bit. Junto a Gemma mantiene 15.3 GB activos (pico de 15.9 GB) de 24 GB.
  - El juez de 8-bit ocupa unos 21 GB junto con el modelo de chat y un cabezal de 0.6B. Decide bien por sí solo (92.8% sin abstención, 1.85 s), pero en cuanto Gemma decodifica las escalaciones, la caché KV lleva al Mac a usar swap. Una rama de dos minutos se ejecutó durante más de 24 minutos y tuvo que detenerse.

    | Decisor junto a Gemma 4 12B (juez de 4-bit) | Precisión | Enviado al LLM | Tokens de LLM ahorrados | p50 de decisión |
    |---|---|---|---|---|
    | Solo LLM | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit, sin abstención | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit, abstención por debajo de 0.6 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit, abstención por debajo de 0.75 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit, abstención por debajo de 0.9 | 97.8% | 74 | 58.5% | 2.5 s |
    | lectura del propio Gemma (`self`), abstención por debajo de 0.75 | 99.4% | 1 | 99.4% | 1.9 s |
    | cabezal OpenJev-0.6B (ejecución de juez de 8-bit), abstención por debajo de 0.75 | 82.8% | 86 | 48% | 0.26 s |

  - Decidir mientras Gemma decodifica una respuesta larga reduce la decodificación de 21 a 3 tok/s (−86%). El tiempo total es de 341 s frente a 202 s ejecutados uno tras otro. Por lo tanto, los decisores en GPU permanecen serializados entre generaciones.
  - Vuelve a ejecutarlo con `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests`.
- **Lo que recibe la nube:** las solicitudes con documentos o audio omiten las etapas en la nube y pasan únicamente por las locales, en su orden.
- **Umbrales:** las preguntas cuya confianza está por debajo de `abstain_below`, o cuyas dos opciones principales están más cerca que `min_margin`, pasan al modelo de respuesta.
- **Puerta epistémica (versión 2):** `epistemic` clasifica `(support, refutation, needs_tool)` antes de decodificar un solo token de respuesta: `enabled`, `support_threshold`, `refutation_threshold`, `tool_threshold`, `on_no_evidence` (`abstain` o `escalate`), `on_conflict` (`report` o `escalate`) y el campo opcional `abstain_message`.

### Rutas de respuesta (`answer.routes`)
- **Coincidencia:** el bloque `when` de una regla coincide cuando se cumplen todas las condiciones. Condiciones:
  - `modality`: documentos, audio o texto;
  - `decide`: ID de pregunta a etiqueta, procedente de System One;
  - `prompt_tokens: { over: N }`: estimado a razón de 4 bytes por token;
  - `escalated`: la etapa de decisión escaló, o una pregunta pasó al modelo de respuesta;
  - `cost_left`: el modelo de la regla aún dispone de presupuesto.
- **Orden de intento:** primero el `model` de la regla, luego su `fallback` y, por último, el modelo local predeterminado del pipeline.
- **Omisiones:** un modelo en la nube se omite para texto percibido, sin presupuesto, tras una clave rechazada o mientras esté limitado por tasa. Cada omisión queda registrada en el recibo.

### Habla (`speak`, `#493`, `#525`)
Un pipeline puede declarar una etapa opcional `speak` para sintetizar audio (`Kokoro-82M`, `Qwen3-TTS`, `Fun-CosyVoice3`, `Chatterbox`, `Zyphra/ZONOS3-Mini-hf`) con respaldo determinista GPU → Neural Engine → voz del sistema:

```yaml
pipeline:
  answer: chat
  speak:
    chain: [tts_zonos, tts_ane, apple-speech]
    voice: Savannah
    language: en_us
    reference_audio: ./speaker.wav
    on_gpu_busy: fallback_ane
    sentence_streaming: true
    format: wav
    exaggeration: 0.5
    max_ttfa_ms: 80
```

- **`speak`:** acepta un único nombre (`speak: apple-speech`), una lista de respaldo (`speak: [tts_zonos, tts_ane, apple-speech]`) o el mapa completo.
- **`chain`:** modelos TTS locales ordenados que terminan opcionalmente con el sintetizador del sistema sin peso `apple-speech`.
- **`voice`:** identificador de voz predeterminado (`Savannah`, `alloy`, `af_heart`, o un identificador `AVSpeechSynthesisVoice`).
- **`language`:** código de idioma o configuración regional predeterminado (`en_us`, `ru`, `de`, `cmn`, etc.).
- **`reference_audio`:** clip local opcional de referencia `.wav` / `.pcm` / `.caf` / `.aiff` / `.flac` para clonación de voz zero-shot.
- **`on_gpu_busy`:** `keep`, `fallback_ane` o `fallback_apple_speech` cuando Metal está ocupado decodificando.
- **`sentence_streaming`:** transmite fragmentos de oraciones full-duplex al TTS a medida que llegan los tokens.
- **`format`:** contenedor de salida o codificación de muestras (`wav`, `pcm`, `pcm_24k`, `mp3`, `opus`, `aac`, `flac`).
- **`exaggeration`:** control expresivo de prosodia en `0.0...2.0` donde sea compatible.
- **`max_ttfa_ms`:** presupuesto máximo de latencia hasta el primer audio antes de enrutar por la cadena de respaldo.

### Imágenes (`imagine`, `#517`, `#539`)
Un pipeline puede declarar una etapa opcional `imagine` para sintetizar o editar imágenes en Apple Silicon a través de hasta cuatro etapas explícitas (reescritor `upsample_via` -> codificador de texto/visión `share_encoder` -> pintor DiT `chain` -> decodificador VAE latente `decoder_via`), además de omisión de pasos TeaCache y evaluación VLM best-of-N:

```yaml
pipeline:
  answer: chat
  imagine:
    chain: [z_turbo, sana_fast]
    upsample_via: chat
    share_encoder: encoder
    decoder_via: vae
    steps: 8
    size: 1024x1024
    format: png
    teacache_threshold: 0.15
    preview_every_steps: 2
    candidates: 3
    judge_via: vlm_judge
    on_gpu_busy: fallback
```

- **`imagine`:** acepta un único nombre de generador (`imagine: z_turbo`), una lista de respaldo (`imagine: [z_turbo, sana_fast]`) o el mapa completo.
- **`chain`:** sintetizadores de imágenes locales ordenados (`FLUX.2/3-klein`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, `Ming-Image`, `Bonsai-4B`, `Step1X-Edit`, `SonicDiffusion`).
- **`upsample_via`:** ranura local de LLM/VLM reescritor utilizada para expandir prompts breves en descripciones de escena detalladas antes de la síntesis (usa la convención completa de prompts de descripción de escena de BFL cuando apunta a `FLUX.3`).
- **`share_encoder`:** ranura local de LLM/VLM (como `Qwen3-VL-4B-Instruct` en 4-bit, 8-bit o BF16) cuyos embeddings de condicionamiento impulsan el sintetizador de imágenes.
- **`decoder_via`:** ranura opcional de decodificador VAE latente local (como el destilado `black-forest-labs/FLUX.2-small-decoder` o el completo `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE`).
- **`steps`:** número predeterminado de pasos de eliminación de ruido en `1...100`.
- **`size`:** resolución de salida predeterminada `<width>x<height>` (múltiplos de 16, p. ej. `1024x1024` o `2048x2048`).
- **`format`:** formato de imagen de salida (`png`, `webp` o `jpeg`).
- **`teacache_threshold`:** umbral L1 residual (`>= 0`) para la omisión de pasos internos de TeaCache.
- **`preview_every_steps`:** emite fotogramas de vista previa TAEF intermedios cada N pasos durante el streaming SSE (`>= 0`).
- **`candidates`:** número de imágenes candidatas con semilla (`1...64`) sintetizadas para la selección best-of-N.
- **`judge_via`:** ranura local de VLM/LLM que puntúa `candidates` frente al prompt y devuelve el artefacto ganador.
- **`on_gpu_busy`:** `queue` o `fallback` cuando Metal está ocupado decodificando.

Preajustes `FLUX.3` incluidos de 4 etapas (`basalt compose up <preset>`): `flux3-16gb-fast`, `flux3-24gb-balanced`, `flux3-36gb-moe-vision`, `flux3-64gb-studio-base` y `flux3-128gb-reference`.

### Versión 2: variables y perfiles
`version: 2` añade dos bloques para que un solo archivo sirva para varios Macs. Un archivo de versión 1 sigue funcionando; usar cualquiera de estos bloques en él es rechazado con "needs version: 2".

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables`:** nombres en minúsculas con valores simples de una línea. `${name}` se sustituye en cualquier parte del archivo. Una referencia `${…}` no declarada es rechazada, por lo que las variables de entorno nunca llegan al archivo; los secretos entran únicamente mediante `api_key_env`.
- **`profiles`:** un perfil está activo cuando su `when` se cumple en este Mac: `memory_gb` con `under` y `over` (memoria unificada en GB) y `chip` (una parte del nombre del chip sin distinción entre mayúsculas y minúsculas, como `M4 Pro` o `Max`). Un perfil sin `when` solo está activo cuando se nombra con `--profile`.
  - Las `variables` de un perfil activo sobrescriben las del archivo.
  - Un modelo o un pipeline con nombre que tenga `profiles: [...]` existe únicamente mientras uno de ellos esté activo.
- **Anulaciones:** `--set name=value` sobrescribe una variable declarada y `--profile name` activa exactamente los perfiles nombrados, para `config`, `plan` y `up`. `compose up` los pasa al servidor, y `up-as-diff` los envía junto con el archivo.
- **Rutas:** una ruta de modelo `./` o `../` es relativa a la carpeta del archivo compose.
- **`basalt compose config -f <file>`** imprime el archivo tal como se resuelve en este Mac, con los perfiles activos. **`basalt compose schema`** imprime el JSON Schema del formato para editores.

### Experimental y MCP (`#485`)
Un archivo compose puede declarar secciones de nivel superior `experimental` y `mcp` (ambas compatibles en las versiones 1 y 2) en lugar de pasar largas listas de flags después de `--`:

```yaml
experimental:
  zero_copy_body: true
  smallm_kernel: true
  speech_analyzer: true
  image_fallback: ocr
  ane_vision_tower: ./tower-bundle
  telemetry_confidence_every: 4
  adaptive_thinking_budget: true
  samplers: true
  mtp_draft_head: true
  expert_streaming: true
  expert_cache_gb: 2.5
  kv_head_split: 0.5            # or quest_top_k: 64 (mutually exclusive)

mcp:
  enabled: true
  rest: true
  research_tools: false
  enable: [surgery, intervene]
  disable: [switch_model]
  image_dirs: [./images]
```

- **`experimental`:** se traduce a los flags equivalentes de `serve` (`zero_copy_body`, `smallm_kernel`, `speech_analyzer`, `image_fallback`, `ane_vision_tower`, `telemetry_confidence_every`, `adaptive_thinking_budget`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k`), validados con números de línea YAML exactos (`telemetry_confidence_every` >= 0, `image_fallback` `ocr`, `ane_vision_tower` que contenga `metadata.json` y `tower.aimodel`, `expert_cache_gb` > 0 con `expert_streaming: true`, `kv_head_split` en `0.05...1`, `quest_top_k` en `1...4096`). Las rutas relativas (`./`, `../`, `~`) se resuelven respecto a la carpeta del archivo compose.
- **`mcp`:** configura el transporte stdio de MCP (`enabled`, `rest`, `research_tools`, `enable`, `disable`, `image_dirs`) utilizando los mismos selectores de grupo y herramienta que `--mcp-enable` / `--mcp-disable` (`serving`, `gemma`, `media`, `harness`, `xray`, `surgery`, `research`, `all`, o nombres de herramientas individuales; `disable` siempre prevalece).
- **Recarga en caliente frente a reinicio:** `speech_analyzer`, `image_fallback`, `telemetry_confidence_every` y `adaptive_thinking_budget` se actualizan en vivo con `compose up` (`POST /v1/compose/apply`, impreso bajo `experimental:`). Los flags vinculados al motor y al transporte (`zero_copy_body`, `smallm_kernel`, `ane_vision_tower`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k` y `mcp`) informan `needs a restart` en `compose diff` y son rechazados en un `apply` en vivo.

### Benchmarks y envíos a tablas de clasificación (versión 2, `#492`)
Un archivo compose `version: 2` puede declarar un bloque de nivel superior `benchmarks` con valores compartidos en `defaults` y suites con nombre en `suites` (ya sea bajo `suites:` o como claves abreviadas de suite directamente bajo `benchmarks:`):

```yaml
benchmarks:
  defaults:
    seed: 42
    temperature: 0.0
    gpu_lock: true
    output_dir: ./bench-out
  suites:
    smoke:
      - adapter: basalt:throughput
        target: model:chat
        params: { iterations: 2, max_tokens: 64 }
        gates: { min_decode_tps: 15, max_peak_gb: 20, max_p50_ms: 5000 }
    frontier:
      profiles: [big]
      budget: { usd_max: 5.0, max_minutes: 30 }
      tasks:
        - adapter: arc-agi-2
          dataset: ./arc-eval.json
          target: pipeline
          gates: { min_accuracy: 0.25, max_ppl: 12.0, max_ece: 0.08 }
          submit:
            target: arc-prize
            api_key_env: ARC_PRIZE_API_KEY
            endpoint: https://arcprize.org/api/v1/submissions
            subset: public-eval
            split: test
```

- **`defaults`:** campos opcionales `seed`, `temperature`, `gpu_lock` y `output_dir`.
- **`suites`:** cada suite es una lista de tareas abreviada (`smoke: [...]`) o `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }`.
- **`tasks`:** cada tarea nombra un `adapter` (`basalt:throughput`, `basalt:perplexity`, `basalt:system-one`, `basalt:passkey`, `basalt:mom`, `basalt:harness`, `arc-agi`, `arc-agi-2`, `terminal-bench`, `swe-bench`, `lm-eval`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `evalplus`), un `dataset` opcional, `target` (`pipeline`, `pipeline:<name>` o `model:<slot>`), `params` escalares, umbrales de regresión `gates` (`min_decode_tps`, `max_peak_gb`, `min_accuracy`, `max_p50_ms`, `max_ppl`, `max_ece`) y `submit` opcional (`target`, `api_key_env`, `endpoint`, `subset`, `split`).
- **Comandos:** `basalt compose plan` imprime las suites resueltas; `basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` ejecuta la suite y exporta paquetes verificados para tablas de clasificación; `basalt bench list|compare|submit` inspecciona adaptadores, compara artefactos y exporta paquetes en modo simulacro (`dry-run`) en los diez destinos (`arc-prize`, `terminal-bench`, `swe-bench`, `hf-leaderboard`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `weave-otel`).

### Fallos de nube y gasto
- Una clave rechazada (HTTP 401/403) desactiva ese modelo en la nube hasta que se reinicie el servidor, dejándolo registrado en una línea de log.
- Un límite de tasa (429) lo pausa durante `Retry-After`, o 60 s si falta la cabecera.
- Mientras un modelo en la nube está desactivado o pausado, responde la siguiente etapa o el respaldo.
- El gasto del día por modelo y en total se guarda en `~/.basalt/cloud-spend.json`, por lo que un reinicio no restablece un límite. Los días son UTC.

## Ejecución de sistemas
- **`compose pull`:** descarga cada modelo local del archivo en la caché de Hugging Face y verifica cada archivo, imprimiendo el progreso cada décimo archivo. Después de un `pull`, `compose plan` dispone de tamaños exactos y `compose up` carga sin esperar descargas.
- **Tamaños en `compose plan`:** un modelo en caché se mide en disco. Un modelo no descargado se dimensiona a partir de la lista de archivos de la API de Hugging Face. Si ninguno conoce el tamaño, `plan` rechaza la operación y nombra el modelo: ejecuta `compose pull` o pasa `--assume-size <GB>` para contarlo con ese tamaño. `plan` ya no cuenta un modelo desconocido como 0 bytes.
- **`compose diff`:** muestra lo que `compose up` cambiaría en el sistema en ejecución con el mismo nombre, sin modificarlo (`POST /v1/compose/diff`). Enumera cargas, descargas, claves y presupuestos, un reinicio necesario o un rechazo: una clave faltante o un sistema que no cabe. Sale con código distinto de cero ante un reinicio o un rechazo.
- **Eventos:** `GET /v1/compose/events` es un flujo de eventos enviados por el servidor (SSE) de lo que hace el sistema. Los tipos de eventos son `loading`, `ready`, `registered` (bajo demanda), `failed` (una sonda de arranque), `applied`, `refused` y `up`. Un cliente nuevo recibe primero los últimos 100 eventos. `compose up --wait` los imprime mientras espera.
- **`compose up` sobre un sistema en ejecución:** si se está ejecutando un sistema con el mismo `name`, `compose up` le envía el archivo (`POST /v1/compose/apply`) y aplica la diferencia.
  - Mientras el sistema sigue cargando o aplicando otro cambio, responde 503. `compose up` espera y vuelve a enviar el archivo, hasta `--wait-timeout` segundos (por defecto 1800).
  - Con `--wait`, `compose up` espera después hasta que `GET /v1/health/ready` responda 200, imprime los modelos por los que está esperando y sale con código distinto de cero si el sistema no está listo dentro del tiempo límite.
  - Carga los modelos añadidos, recarga los modificados y descarga los eliminados.
  - Intercambia los pipelines entre solicitudes.
  - Comprueba primero el nuevo sistema frente a la memoria de este Mac, como hace `compose plan`, contando los modelos de cada pipeline. Un sistema que no cabe es rechazado y nada cambia.
  - Cuando el nuevo sistema cabe pero los modelos antiguos y nuevos no caben juntos lado a lado, descarga los modelos que reemplaza antes de cargar sus sucesores. Una solicitud que necesite uno de ellos en esa ventana espera la carga o falla. Si falla una carga posterior, vuelve a cargar los modelos antiguos.
  - Imprime `load`, `reload`, `unload` y `keep` por modelo, `budget` para cada presupuesto en la nube modificado (`total` para el del archivo) y `key` para cada modelo en la nube cuya `api_key_env` haya cambiado.
  - El presupuesto o la variable de clave de un modelo en la nube cambian sin recarga. Los nuevos límites se aplican de inmediato, y un modelo desactivado por una clave rechazada vuelve a activarse cuando cambia su variable de clave.
  - Un modelo de chat distinto se carga en el lugar del modelo en servicio y se imprime como `chat`. Las solicitudes esperan mientras carga. Si un paso posterior del `apply` falla, se vuelve a cargar el modelo de chat anterior.
  - Un nuevo modelo de decisión local de un pipeline se carga como cualquier modelo añadido. El del pipeline predeterminado también se imprime como `decide`. Los pipelines y `/v1/systemone` lo usan a partir de la siguiente solicitud. Un modelo que empieza o deja de decidir se recarga en su nuevo rol.
- **Cambios que requieren reiniciar:** un modo de memoria distinto, un perceptor secuencial, flags de arranque del motor o HTTP bajo `experimental`, o cambios bajo `mcp`. Ejecuta `compose down` y luego `compose up`.
- **`compose ps`:** enumera cada sistema en ejecución (`GET /v1/compose`) con sus pipelines y, por modelo:
  - su estado, uno de `loading`, `ready`, `unloading`, `unloaded`, `idle-unloaded` (bajo demanda, no cargado ahora), `decision` (mantenido por el motor de decisión), `cloud`, `no-key`, `over-budget`, `rate-limited`, `refused` y `failed`;
  - para un modelo bajo demanda, cuándo se usó por última vez y cuándo se descarga;
  - para un modelo en la nube, su gasto de hoy frente a su límite y un rechazo si corresponde.
- **Salud (`Health`):** `GET /v1/health/live` responde 200 mientras el servidor atienda HTTP. `GET /v1/health/ready` responde 200 una vez que cada modelo puede servir. Hasta entonces responde 503 y enumera los modelos que siguen esperando, como `chat (loading)`. Un modelo bajo demanda inactivo, un modelo en la nube sin presupuesto y un modelo en la nube limitado por tasa cuentan como aptos para servir; la cascada o el enrutador los saltan.
- **Claves de nube:** `compose up` se niega a arrancar cuando `api_key_env` de un modelo en la nube no está definida y nombra la variable. `up-as-diff` comprueba en su lugar el entorno del servidor en ejecución.
- **`compose down`:** envía SIGINT al servidor. Finaliza las solicitudes en curso y descarga los modelos. Si el proceso sigue ejecutándose después de 30 s, `down` lo indica y sale con código distinto de cero.
- **Dónde reside el estado:** el PID, el puerto y la hora de inicio del proceso del sistema en ejecución están en `~/.basalt/compose/<name>.json`. Un registro cuyo PID pertenezca ahora a otro proceso, porque el PID fue reutilizado, cuenta como no en ejecución.
- **Clave de API:** cuando el servidor tiene una clave de API, define `BASALT_API_KEY` para `ps` y `up-as-diff`.

## Adaptadores de dominio (`version: 2`)
Un archivo `version: 2` puede declarar `adapters:` entrenados sobre los preajustes de Hugging Face NVIDIA Nemotron (`nemotron:post-v2`, `nemotron:post-v1`, `nemotron:code`, `nemotron:math`, `nemotron:crossthink`, `nemotron:personas`), conjuntos de datos `hf:<org>/<repo>` o archivos locales `.jsonl`, y enrutar solicitudes entre ellos sobre un único modelo `base` residente sin recargar pesos:

```yaml
adapters:
  code_oft:
    base: chat
    method: oft              # lora | phm | oft
    rank: 16
    scale: 10.0
    hypercomplex_dim: 8
    factor_rank: 1
    shared_basis: true
    block_size: 16
    parameterization: cayley # cayley | matrixExp
    block_shared: false
    layers: [0, 1, 2, 3]
    output: ./adapters/code_oft
    iterations: 200
    batch_size: 2
    learning_rate: 0.0001
    max_seq_len: 2048
    checkpoint_every: 50
    corpus:
      source: nemotron:code
      subset: default
      split: split_0
      max_rows: 512
      min_pass_rate: 0.9
      include_reasoning: true
      seed: 42
    gate:
      max_kl: 0.2
      max_ppl_rise: 0.05
      held_out: ["Explain what a hash table is in two sentences."]
```

Ejecuta `basalt compose train -f basalt.yaml` (opcionalmente con `--adapter`, `--force`, `--resume`) para entrenar los adaptadores declarados en disco y luego selecciona un `adapter:` en `pipeline.answer` o en cualquier entrada de `pipeline.answer.routes`.

## Recibos
Cada respuesta de `/v1/pipeline` y cada registro de log de sesión incluye un recibo. Por etapa indica el modelo, los milisegundos y los tokens. Las etapas de decisión también incluyen su coste y cualquier etapa en la nube que se haya omitido. Las etapas de respuesta incluyen la regla que se activó (`route`), el coste y los modelos omitidos. `swap_ms` es el tiempo invertido en cargar modelos bajo demanda y secuenciales.

## Perfiles calibrados por hardware incluidos (`#526`)

`basalt compose` y `Basalt.app` (`Compose Profiles`, `⌥⌘C`, además de **Settings → Engine → Bundled Profile**) incluyen **84 perfiles calibrados por hardware** distribuidos en 7 niveles de memoria unificada de Apple Silicon (`16gb`, `24gb`, `32-36gb` [alias `32gb` y `36gb`], `48gb`, `64gb`, `128gb`, `256gb` [alias `192gb` y `236gb`]) × 12 arquetipos de carga de trabajo:

| Arquetipo | Principal | Enfoque |
|---|---|---|
| `daily-chat` | ★ | Chat diario, percepción de documentos con Apple Vision, entrada de voz local |
| `coding-agent` | ★ | Kernel especulativo small-M, enrutamiento especializado en código, herramientas MCP |
| `voice-zonos` | ★ | TTS neuronal Zyphra `ZONOS3-Mini` a 44.1 kHz + respaldo ANE `Kokoro-82M` |
| `multimodal-desk` | ★ | OCR, percepción de audio, triaje System One, chat y voz ZONOS3 |
| `reasoning-lab` | | Presupuestos de pensamiento adaptativos + cascada System One para matemáticas y lógica |
| `document-ocr` | | OCR neuronal (`PaddleOCR-VL-1.5`) + puerta epistémica de Belnap de cuatro valores |
| `voice-duplex` | | Transcripción full-duplex y síntesis en streaming con TTFA por debajo de 80 ms |
| `creative-studio` | | Síntesis de imágenes nativa en MLX (`Z-Image-Turbo` / `FLUX`) con TeaCache |
| `rag-knowledge` | | Recuperación fundamentada de contexto largo con puertas de conflicto y abstención |
| `hybrid-cloud` | | Chat local centrado en la privacidad con escalación a la nube limitada a `$1/day` |
| `moe-frontier` | | MoE disperso (`Flint 40B-A2.5B`, `Nemotron-3.5`, `Qwen3.8`) + streaming de expertos |
| `research-xray` | | Model X-Ray, cirugía de pesos y benchmarks rápidos `basalt:throughput` |

Al iniciarse, `Basalt.app` detecta la memoria unificada física (`ComposeMemoryTier.detect`) y fija los 4 perfiles principales con estrella para este Mac en **Recommended for this Mac**, manteniendo todos los 84 perfiles buscables y seleccionables. Cualquier perfil se ejecuta de forma idéntica desde la CLI (`basalt compose plan 24gb-voice-zonos`, `basalt compose up 36gb-moe-frontier`, `basalt compose up 236gb-multimodal-desk`).

## Ejemplos
- `docs/examples/compose/any-mac.yaml`: un archivo de versión 2 para Macs de 16, 24 y 64 GB. El modelo de chat proviene de un perfil elegido según la memoria, y el perfil de 64 GB añade un modelo de 27B bajo demanda y un pipeline `long`.
- `docs/examples/compose/mac-16gb.yaml`: Ornith 9B local, OCR bajo demanda, prompts largos a Claude Haiku dentro de $0.50 al día.
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B, OCR bajo demanda, código a Haiku, un segundo pipeline `support`, $1 al día en total.
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B con Qwen3.8-27B y Nemotron Transcribe bajo demanda, sin respuestas en la nube.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` y `too-big.yaml`: los archivos detrás de las mediciones siguientes.

Hay más mediciones en #412.

## Mediciones

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

Carga de trabajo mixta de 55 solicitudes a través de `/v1/pipeline` (30 decisiones, 10 de código, 10 de chat, 5 documentos), M4 Pro, compilación release (`scripts/compose_mom_bench.swift`, `docs/examples/compose/mom.yaml`):

| Configuración | Etiquetas correctas | Nube por cada 1000 solicitudes | p50 | p95 | Tiempo en etapa local |
|---|---|---|---|---|---|
| solo local (Gemma 4 12B decide y responde) | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| nube primero (Jev decide, Haiku responde todo el texto) | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM (Jev decide, código a Haiku mientras dure el presupuesto, el resto Gemma) | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

Medido con `compose up` en un M4 Pro con 24 GB, compilación release:

| Archivo | Solicitud | Recibo |
|---|---|---|
| document-qa-cloud | documento (primero) | OCR cargado bajo demanda (489 ms), decidido localmente, respondido: 6.1 s |
| | documento (de nuevo) | OCR ya cargado (0 ms): 5.0 s |
| | tras 40 s inactivo | OCR descargado, huella 12 → 11 GB |
| | pregunta escrita | decidido por Jev en la nube: 0.6 s |
| | documento | OCR recargado (267 ms): 5.3 s |
| voice-agent (incluido) | audio | chat descargado alrededor de Nemotron Transcribe (intercambio 6.9 s): 17.3 s |
| too-big (ejemplo) | `compose plan` | rechazado: una etapa necesita 20.7 GB más 2 GB frente a un conjunto de trabajo de 19.1 GB |


## Extensiones del esquema `version: "2.2"`

Compose `version: "2.2"` (compatible con `version: 2`) añade configuración declarativa para LLM de difusión por bloques, borradores especulativos `DFlash`, voz a voz dúplex, codificadores de texto compartidos VLM/DiT, escalado en 1 paso `SeedVR2`, DiT musical `ACE-Step`, DiT de vídeo VAE 3D causal, entrenamiento por preferencias por segmentos y clústeres multi-Mac por Thunderbolt/LAN:

- Por modelo: `speculative` (`mode`, `drafter`, `block_size`, `verify`) y `diffusion_text` (`dual_kv_cache`, `confidence_threshold`, `contrastive_alpha`).
- Extensiones de `imagine`: `share_text_encoder_with`, `kv_cache_references`, `rgba` y `upscale` (`model`, `steps`, `target_resolution`).
- `voice` en la raíz (`mode`, `sts_model`, `denoise`, `turn_detection` con `model` y `diarization`, `tts_fallback`, `aligner`).
- `music` en la raíz (`model`, `steps`, `duration_seconds`) y `video` (`model`, `vae_tiling`, `sta_sparse`, `fps`, `frames`).
- `train` en la raíz (`method`, `segments`, `ema_decay`) y `cluster` (`topology`, `peers`, `compression`).
- Enrutamiento empresarial, caché y gobernanza (`v0.12`): enrutamiento de 5 componentes `BusinessOrchestratorCore`, caché semántica `LANTERN` (`SemanticCacheCore` / `SemanticCacheActor`, `#625`), puerta de aprobación `HITL` (`/v1/mcp/approvals`) + `SymbolicKnowledgeGraph` de LAAS-CNRS (`#626`) y `TieredKVHierarchyCore` L0/L1/L2 consciente de UMA + `AdaptiveHeterogeneousPlanner` (`#627`).
