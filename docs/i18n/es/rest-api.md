# API REST

> Languages: [English](../../rest-api.md) · [Русский](../ru/rest-api.md) · [中文](../zh/rest-api.md) · **Español** · [Français](../fr/rest-api.md) · [日本語](../ja/rest-api.md) · [Deutsch](../de/rest-api.md)

URL base: `http://127.0.0.1:8080` — **no requiere autenticación**.

## Seguridad

En loopback cualquier proceso local puede llamar a la API sin clave, pero una página web en tu navegador no puede hacerlo:

- **Host.** Enlazado a loopback, el servidor responde únicamente a solicitudes cuyo `Host` sea `localhost`, `127.0.0.1` o `[::1]` (cualquier puerto); cualquier otro recibe 403; esto frena el DNS rebinding, donde el dominio de un atacante vuelve a resolverse a 127.0.0.1 y su página lee tus respuestas. Enlazado a una dirección específica, esa dirección también se acepta. Enlazado a `0.0.0.0`/`::`, se acepta cualquier `Host`: allí la clave API es obligatoria.
- **Origin.** Una solicitud con cabecera `Origin` se rechaza (403) a menos que sea el propio origen loopback del servidor (`http(s)://localhost|127.0.0.1|[::1]` en el puerto donde escucha el servidor; sin puerto significa 80 para http, 443 para https) o figure en `--allow-origin` / `allowed_origins` (`scheme://host[:port]`, sin ruta, sin barra final, sin `*`; los puertos por defecto coinciden, así que `https://app.example:443` equivale a `https://app.example`). Otros puertos locales no son de confianza: de otro modo una página en un servidor de desarrollo (`localhost:3000`) o Jupyter podría cargar modelos y ejecutar generación, por lo que una interfaz así necesita `--allow-origin http://localhost:3000`. Los clientes que no envían `Origin` (curl, SDKs, Claude Code, Codex) no se ven afectados.
- **Content-Type.** Todo `POST` debe enviar `Content-Type: application/json` (un parámetro `charset` es válido), de lo contrario devuelve 415. `curl -d` sin `-H "Content-Type: application/json"` envía un tipo de formulario y es rechazado.
- **IDs de modelo.** Los valores `model` en REST son IDs de repositorio de Hugging Face: `/v1/models/load` solo acepta `org/name`; en el `model` de una solicitud de generación, una forma de ruta (que empiece por `/`, `.` o `~`, o contenga `..`, incluso oculta en `claude-local/..--x`; los espacios en blanco alrededor se ignoran) devuelve 400, mientras que cualquier otro valor que no sea un ID de repositorio (`basalt`, `qwen/qwen3-8b-instruct:free`) es un alias: responde el modelo cargado y no se cambia nada. Las rutas locales siguen disponibles para `basalt serve --model-path` y MCP. Una carga fallida deja cargado el modelo actual, y los cuerpos de error nunca contienen rutas del sistema de archivos (quedan en el registro del servidor).

## Gestión de modelos

```bash
# List all locally cached models (OpenAI-compatible format)
curl http://localhost:8080/v1/models

# Currently loaded model + readiness status
curl http://localhost:8080/v1/models/current

# Hot-swap to a different model at runtime (blocks until loaded)
curl -s http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` devuelve IDs con el prefijo `claude-local/` para que el selector automático de modelos de Claude Code los añada. El campo `display_name` contiene el ID original del repositorio de HuggingFace.

## Generación directa

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| Campo | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `prompt` | string | obligatorio | Texto de entrada |
| `maxTokens` | int | 8192 | Máximo de tokens a generar |
| `temperature` | float | modelo | Temperatura de muestreo (0–2); sin definir toma `generation_config.json` del modelo, o si no 0.7 |
| `topP` | float | modelo | Muestreo nucleus (0–1); sin definir toma `generation_config.json`, o si no 0.9 |
| `topK` | int | modelo | Conserva los k tokens más probables; sin definir toma `--top-k`, luego `generation_config.json`, o si no desactivado |
| `presencePenalty` | float | desactivado | −2…2, restado de cada token entre los últimos 1024 tokens generados (el prompt no se penaliza) |
| `repetitionPenalty` | float | `--repetition-penalty` | (0, 2], penalización multiplicativa sobre tokens recientes |
| `thinkingBudget` | int | desactivado | Tokens de razonamiento antes de que el servidor fuerce el fin del pensamiento (Qwen: "Considering the limited time by the user…" + `</think>`) para que la respuesta quepa en `maxTokens`; se reduce para ajustarse a `maxTokens` menos la transición y una reserva de respuesta de min(512, maxTokens/4); un `maxTokens` que deje menos de 64 tokens tras la reserva devuelve 400; se ignora cuando el pensamiento está desactivado, el modelo no tiene protocolo de razonamiento o es un modelo no Qwen cuyo protocolo no declara transición |
| `minP` | float | servidor | Min-P 0–1: descarta tokens por debajo de `minP × max p`. Sobrescribe `--min-p` y el perfil del modelo; `0` lo desactiva |
| `dynamicTemperature` | bool | false | Temperatura dinámica basada en entropía (EDT): por token `T = T0·0.8^(1.5/H) + 0.5·R`, acotada a 0.05–2 (una solicitud más fría conserva su propia temperatura como suelo; `top_p`/`min_p` recortan la distribución original), donde `H` es la entropía de la distribución del siguiente token y `R ∈ [0,1]` un índice de repetición. Casi greedy en tokens seguros, más cálida cuando empieza un bucle. Sin efecto con `temperature: 0` |
| `speculativeLookup` | bool | `--speculative-lookup` | Decodificación especulativa por búsqueda en prompt para esta solicitud: hasta 16 tokens copiados de una coincidencia de n-gramas en el prompt o la salida se verifican en una sola pasada forward; tras un borrador rechazado pausa la generación de borradores durante 1, 2, 4… hasta 64 tokens. Con `serve --speculative-recycling`, cuando no coincide ningún n-grama genera un borrador de hasta 6 tokens a partir de los propios sucesores argmax del modelo registrados dos veces seguidas (Token Recycling, #38), con su propia pausa. Modelos solo de atención e híbridos recurrentes densos (Qwen3.5/Ornith, Nemotron-H: un borrador rechazado restaura el estado recurrente y reproduce los tokens aceptados), modelos de ventana deslizante (Gemma 4: cada ronda de verificación se prepara por separado, así que un borrador rechazado nunca sobrescribe el anillo de la ventana), sin `--kv-bits` y con entrada solo de texto; los híbridos recurrentes con enrutamiento de expertos (LFM2.5 MoE) se ignoran. En Gemma 4 26B-A4B (MoE) lookup gana 1.97× en ediciones de código y cuesta un 3–6% en prosa y razonamiento |

`/v1/chat/completions`, `/v1/responses` y `/v1/messages` aceptan los mismos dos campos como `min_p` y `dynamic_temperature`; `gemma_generate` en MCP como `minP` y `dynamicTemperature`. Los endpoints de OpenAI aceptan `top_k`, `presence_penalty`, `repetition_penalty` y `thinking_budget`; `/v1/messages` acepta `top_k` y `thinking: {"type": "enabled", "budget_tokens": N}` como presupuesto de pensamiento (1 ≤ N < `max_tokens`, de lo contrario 400). Los tres aceptan `speculative_lookup` (un valor no booleano devuelve 400); `gemma_generate` en MCP acepta `speculativeLookup` (un valor no booleano devuelve -32602). El campo de uso reporta `completion_tokens_details.reasoning_tokens` (chat; emitido en stream con `stream_options.include_usage`) y `output_tokens_details.reasoning_tokens` (Responses): `0` cuando el modelo no razonó, y la transición forzada cuenta como razonamiento.

## Compatible con OpenAI

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Soporta streaming SSE, system prompts y conversación multiturno. Enviar un ID de modelo de HuggingFace en el campo `"model"` dispara un cambio en caliente automáticamente.

El pensamiento de un modelo de razonamiento se devuelve como `message.reasoning` (en stream: `delta.reasoning`), el nombre de OpenRouter / Ollama, y el mismo texto de nuevo como `reasoning_content`, el nombre que leen los clientes de vLLM, DeepSeek, llama.cpp y LM Studio. Los turnos del asistente devueltos con cualquiera de los dos campos se aceptan; el razonamiento en sí no se vuelve a introducir en el prompt.

## Compatible con Anthropic

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Implementa la secuencia completa de eventos SSE de Anthropic (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Acepta formatos de contenido en cadena de texto simple o arreglo de bloques.

Los bloques `thinking` llevan una `signature` vacía: basalt no tiene clave para firmarlos. Hacen ida y vuelta a través de basalt sin problema, pero la API real de Anthropic rechaza una conversación que los reenvíe, así que descártalos antes de enviar el mismo historial allí.

**Salida restringida.** `response_format` restringe la respuesta, tras cualquier bloque de razonamiento, con XGrammar:
- `{"type": "json_object"}` o `{"type": "json_schema", "json_schema": {"schema": …}}`;
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}`;
- `{"type": "grammar", "grammar": "root ::= …"}`, dialecto EBNF/GBNF de XGrammar con regla raíz `root`;
- `{"type": "structural_tag", "format": …}`, JSON structural-tag de XGrammar (`sequence`, `const_string`, `regex`, `json_schema`, `tag`, …).

Los campos de solicitud `guided_regex` y `guided_grammar` de vLLM también funcionan, uno cada vez. La herramienta `generate` de MCP acepta `regex` o `grammar`. Un código fuente de más de 64 KB, o uno que XGrammar no compile, devuelve 400 indicando el nombre del formato.

**Prefill.** Una conversación que termina con un mensaje `assistant` continúa ese mensaje en lugar de responderle en un turno nuevo, en `/v1/messages`, `/v1/chat/completions` y `/v1/responses`, igual que en la API de Anthropic, `continue_final_message` de vLLM y llama.cpp. Los turnos anteriores se renderizan hasta el prompt de generación con el pensamiento desactivado, el texto del asistente va a continuación sin cerrarse, y la respuesta contiene solo lo que el modelo escribe después: haz prefill de `{"name":` y la respuesta es ` "Whiskers", "age": 3}`. Con `response_format` o una herramienta forzada, el prefill es el inicio del documento JSON (un prefill con el que el esquema no pueda empezar devuelve 400), y los argumentos de una herramienta forzada lo incluyen. Un prefill que abre un bloque de pensamiento (`<think>…`) conserva el pensamiento solicitado y continúa como razonamiento. Varios mensajes `assistant` al final se unen en un solo prefill; uno vacío se descarta; un mensaje `assistant` con llamadas a herramientas sigue siendo un turno terminado. Un prompt con imágenes no puede continuarse (400). Los tokens de control en el prefill (`<|im_end|>`, `<|im_start|>`, …) se neutralizan igual que el resto de la conversación, de modo que un prefill no puede falsificar un turno; solo los marcadores de pensamiento que abren y cierran su razonamiento siguen siendo reales, y en una plantilla que ya abre `<think>` por sí misma (Ornith, Qwen3.5+) el `<think>` inicial del propio prefill se descarta en lugar de duplicarse. Con `--cloud-fallback`, una solicitud con prefill siempre permanece en local, ya que no todos los proveedores continúan un mensaje final del asistente. En Nemotron Nano v2, que desactiva el pensamiento con una línea `/no_think` en el turno de sistema, una solicitud con prefill renderiza un turno de sistema distinto al de una solicitud con pensamiento activo sobre el mismo historial, así que ambas solo comparten la prompt-cache anterior a ese punto.

## Entrada de imágenes

Un modelo de visión (Gemma 4, Qwen3.5-VL, … — cualquier modelo que cargue MLXVLM) acepta imágenes en `/v1/chat/completions`, `/v1/responses`, `/v1/messages` y la herramienta `generate` de MCP:

- **OpenAI:** una parte de contenido `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (`/v1/responses`: `{"type": "input_image", "image_url": "data:…"}`).
- **Anthropic:** un bloque de contenido `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`.
- **MCP:** `images: [{"data": "<base64>", "mime": "image/png"}]`, o `[{"path": "/abs/file.png"}]` para un archivo dentro de un directorio indicado con `serve --image-dir <dir>` (repetible). La ruta se comprueba tras resolver enlaces simbólicos; sin `--image-dir` toda ruta se rechaza.

PNG, JPEG, WebP y HEIC, como máximo 8 imágenes por solicitud, 20 MB de bytes codificados por imagen y 32 MB para todas las imágenes de una solicitud; cada exceso devuelve 400 indicando el límite. Las URLs remotas (`http(s)://` en `image_url`, `source.type: "url"`) se rechazan con 400 — basalt no las descarga; envía la imagen como URI `data:`. Una imagen enviada a un modelo que no es de visión devuelve 400 (`<model> does not take images`), nunca un descarte silencioso, y una solicitud con imágenes nunca va a `--cloud-fallback`. La prompt-cache se indexa por el SHA-256 de cada imagen: las mismas imágenes en el mismo orden reutilizan el prefijo en caché, imágenes distintas no reutilizan nada. Los registros de sesión guardan digests y conteos de bytes, nunca los bytes de las imágenes. Las imágenes dentro de resultados de herramientas siguen mostrándose al modelo como `[image omitted]`.

## Streaming SSE (API nativa)

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

Cada evento es una línea `data: {...}`; el stream termina con `data: [DONE]`. Un fallo a mitad del stream llega como `event: error` con `data: {"error": ...}`.

## Telemetría y Prometheus

- **`GET /api/v1/stats`** devuelve la instantánea de telemetría como JSON.
  - También incluye `windows`: las últimas ventanas de 1 minuto, 5 minutos y 1 hora con solicitudes, tokens, p50/p95 de tiempo hasta el primer token y tok/s, la tasa de aciertos de prompt-cache y la tasa de aceptación de borradores.
  - También incluye `decisions`: por modelo de decisión, solicitudes, respuestas, abstenciones, escalados, coste en la nube, latencia y respuestas por decil de confianza.
- Con `--experimental-expert-streaming`, también incluye `experts`: aciertos, fallos, expulsiones y bytes leídos de la caché de expertos transmitidos en streaming, y por capa MoE los expertos enrutados desde la carga (conteo, distintos, los ocho más usados y su proporción).
- Con `--telemetry-confidence-every N`, cada N-ésima generación registra su probabilidad del token principal, entropía y margen entre los dos primeros tokens. Las solicitudes que lo pidan se registran siempre. `confidence` cuenta entonces generaciones por decil de la menor probabilidad de token principal, la probabilidad media y el menor margen, y por intervalo de medio nat de entropía media. Cada token medido cuesta una sincronización de GPU, así que el valor por defecto `0` mide solo las solicitudes que lo solicitan.
- **`GET /api/v1/metrics`** devuelve los mismos datos como texto OpenMetrics para Prometheus o Grafana.
  - Los nombres de métricas empiezan por `basalt_`. Ventana y cuantil son etiquetas, al igual que el modelo de decisión.
  - Pasa por la clave API del servidor como cualquier otro endpoint: facilita el token bearer al scraper.

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

La telemetría almacena únicamente conteos, tiempos e histogramas: sin texto de prompts ni de respuestas (#450).

## Endpoints multimodales, de conversión, X-Ray y gobernanza (`v0.12`)

- **Video Studio (`/v1/videos/generations`, `/v1/videos/{id}`)** — Generación por fases para 24 GB UMA (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video MoE`, `Wan 2.2`) que devuelve `.mp4` multiplexado mediante `AVAssetWriter` (`#608`–`#611`).
- **Audio y música (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/music/generations`)** — `ZONOS3-Mini`, `Kokoro-82M`, códecs dúplex Mimi/SNAC `LFM2.5-Audio`, `SmartTurn`/`Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `ACE-Step 1.5` (`#604`, `#605`).
- **Imágenes (`/v1/images/generations`, `/v1/images/edits`)** — `FLUX.3`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, escalado `SeedVR2`, soporte RGBA (`#542`, `#605`).
- **Conversor universal (`/v1/models/convert`, `/v1/models/convert/jobs/{id}`)** — Cuantización asíncrona any-to-any (`MACaQ`, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`, `#546`, `#623`).
- **Atribución de tokens y riesgo de alucinación (`POST /v1/xray/tokens`)** — Logprobs por token, entropía top-k, atribución por capa/cabeza/MoE y señales de riesgo de alucinación (`#613`, `#614`).
- **Gobernanza HITL (`/v1/mcp/approvals`)** — Cola de aprobación con intervención humana (human-in-the-loop) para llamadas a herramientas MCP de alto impacto, emparejada con `HITLGovernanceCore` y `SymbolicKnowledgeGraph` (`#626`).

## Swagger UI

Visita `http://localhost:8080/swagger` para consultar la documentación interactiva de la API.
