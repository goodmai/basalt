# Radiografía del modelo y herramientas de investigación (#328)

> Languages: [English](../../research-tools.md) · [Русский](../ru/research-tools.md) · [中文](../zh/research-tools.md) · **Español** · [Français](../fr/research-tools.md) · [日本語](../ja/research-tools.md) · [Deutsch](../de/research-tools.md) · [한국어](../ko/research-tools.md) · [Bahasa Indonesia](../id/research-tools.md)

El servidor MCP de Basalt puede mostrar qué está haciendo el modelo cargado y ejecutar experimentos pequeños y reversibles sobre él. Las herramientas de radiografía de solo lectura siempre aparecen listadas. Las herramientas de experimentación solo se listan y se aceptan cuando el servidor se inicia con `basalt serve --mcp --research-tools`.

Cada llamada pasa por la puerta de generación con prioridad en segundo plano, de modo que las peticiones de los usuarios tienen prioridad. Las respuestas están acotadas: estadísticas y pequeños fragmentos, nunca un tensor completo ni una matriz de atención entera.

## Herramientas

Los ejemplos corresponden al objeto `arguments` de una petición MCP `tools/call`.

### Radiografía de solo lectura

| Herramienta | Qué devuelve | Argumentos de ejemplo |
|---|---|---|
| `model_describe` | Tipo de modelo, arquitectura, tipo por capa (atención completa, ventana deslizante, atención lineal, mamba, conv), cabezas, cabezas KV, tamaño de cabeza, ventana, expertos, clase de caché KV, cuantización (incluidos manifiestos de módulos `MACaQ` / `FlatQuant` / `LIRQuant` junto con la precisión base, `#630`), número de parámetros y bytes | `{}` |
| `model_tensors` | Nombres de parámetros con forma, dtype y si están cuantizados; `filter` es una subcadena del nombre | `{"filter": "layers.3.mlp"}` |
| `model_tensor` | Media, desviación estándar, mínimo, máximo, L2 y los percentiles p1/p5/p50/p95/p99 de \|x\| para un tensor (descuantizado), más hasta `maxValues` valores de un fragmento. Con `"spectrum": true`, también el espectro de la matriz, o de un experto de una pila 3-D seleccionado por el fragmento: varianza de ruido de Marchenko–Pastur, bordes del cuerpo espectral y masa de autovalores por encima del cuerpo, rango de Gavish–Donoho, exponente `alpha` de la cola de ley de potencias y los segundos que tardó (Gram WᵀW del lado menor en la GPU, `eigvalsh` en la CPU; lado menor ≤ 8192; solo con `serve --research-tools`, ya que retiene el modelo durante segundos) | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}`, `{"name": "model.layers.3.mlp.down_proj.weight", "maxValues": 0, "spectrum": true}` |
| `kv_inspect` | El árbol de caché de prompts por espacio de nombres: nodos, hojas, tokens, bytes, instantáneas recurrentes, fijados (pins), último uso; presupuesto y desalojos | `{"namespace": "claude-code"}` |
| `trace_run` | Una ejecución de captura de un prompt: norma del estado oculto por capa, top-k de logit lens, posiciones más atendidas y entropía por cabeza en una posición, elecciones del router MoE | `{"prompt": "The capital of France is", "layers": [0, 12, 23], "topK": 5}` |
| `telemetry_query` | Agregados en vivo del proceso servidor (#450): totales; ventanas móviles con p50/p95 de TTFT y tok/s, tasas de acierto de caché de prompts y aceptación de borrador; conteos, coste y deciles de confianza por modelo de decisión; contadores de caché de expertos en streaming y enrutamiento por capa; deciles de confianza por token; salud y memoria. Solo conteos y tiempos | `{"sections": ["windows", "decisions"]}` |

### Experimentos (`--research-tools`)

| Herramienta | Qué hace | Argumentos de ejemplo |
|---|---|---|
| `intervene` | Suma un vector de direccionamiento (steering) al flujo residual tras una capa, o pone a cero una cabeza de atención o un experto. Con `when`, solo en los tokens que selecciona un disparador (ablación condicional). Se aplica únicamente a llamadas de investigación (`experiment_compare`, `trace_run`). | `{"kind": "ablate_head", "layer": 10, "head": 3}`, `{"kind": "ablate_expert", "layer": 20, "expert": 7, "when": {"kind": "projection", "layer": 14, "vector": [0.01, …], "threshold": 2, "window": 16}}` |
| `weights_delta` | Suma `values × scale` a un fragmento de un tensor flotante en memoria (un único valor se difunde). Modifica el modelo en vivo, por lo que las peticiones de usuarios también lo ven (ver más abajo). | `{"tensor": "model.layers.10.post_attention_layernorm.weight", "slice": [[0, 128]], "values": [0.05], "scale": 1}` |
| `overlays_list` | Capas superpuestas (overlays) activas: id, descripción, hora de creación y de expiración | `{}` |
| `overlay_undo` | Revierte un overlay por id; un delta de pesos se restaura bit a bit | `{"id": 4}` |
| `experiment_compare` | Comparación greedy antes/después sobre un conjunto de prompts con los overlays indicados: ambos textos, divergencia KL de los logits, ambas perplejidades | `{"prompts": ["2 + 2 =", "Name a prime above 100."], "overlays": [3, 4], "maxTokens": 32}` |
| `adapter_apply` | Aplica un adaptador LoRA al modelo en vivo: un directorio de adaptador mlx-lm (`adapter_config.json` + `adapters.safetensors`), o una actualización de rango 1 `scale × (x·input) output` de una capa lineal, incluidas capas cuantizadas. Marca el modelo como modificado igual que un delta de pesos; `overlay_undo` restaura las capas bit a bit. Con `gate`, la puerta de calidad se ejecuta tras aplicarlo: si pasa, el adaptador permanece sin expiración; si falla, se revierte y el informe vuelve con `overlay: null`. Sin `gate` expira tras 30 minutos. | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | Ejecuta la puerta de calidad sobre el modelo actual frente al modelo intacto (retirando temporalmente todos los adaptadores y deltas de pesos activos para la línea base): cada comprobación con su valor, límite y veredicto, y las filas de comparación del conjunto reservado (`heldOut`) | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | Reempaqueta el peso de una capa lineal en memoria: `affine` (`bits` 2, 3, 4, 5, 6 u 8, `groupSize` 32, 64 o 128), `mxfp4`, `nf4:32`, `nf4:64` o `nf4:128` (el libro de códigos NormalFloat de 16 niveles de QLoRA con una escala absmax en fp16 por grupo, mantenido como un `QuantizedLinear` nativo nf4 que sirven los kernels nf4 del fork), o `ternary` (escala absmean·{-1, 0, 1} por grupo de 64). Una capa cuantizada se descuantiza primero. La capa se sustituye en vivo, como un delta de pesos (ver más abajo) | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | Reemplaza en vivo una capa lineal cuantizada por una capa densa `float16` o `bfloat16` (por defecto `bfloat16`) que contiene su peso descuantizado | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |
| `train_experts` | Autoentrenamiento (#332 modo 3, de larga duración): recopila respuestas a tareas verificables, conserva solo las verificadas, entrena un LoRA, lo guarda, informa la pérdida y la precisión reservada antes/después y luego lo aplica a través de la puerta de calidad. Véase [Autoentrenamiento](#autoentrenamiento-train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |
| `distill` | Destilación profesor → estudiante (de larga duración): un profesor detrás de un endpoint compatible con OpenAI o Anthropic responde los prompts (en caché, con presupuesto), se entrena un LoRA solo sobre sus respuestas, se informa la NLL del profesor en el conjunto reservado antes y después y el adaptador se aplica a través de la puerta de calidad. Segmentado por tiempo: el modelo se retiene un paso de optimizador cada vez, por lo que `serve` sigue respondiendo y una petición espera como máximo un paso; solo se ejecuta una destilación por servidor. Las ejecuciones offline escriben el adaptador y el estado del optimizador en `<output>/checkpoint` cada `checkpointEvery` pasos (por defecto 50, 0 desactivado); `resume: true` con el mismo `output` continúa desde él, y se rechaza si el modelo, las filas de entrenamiento o los ajustes de LoRA difieren. Las ejecuciones `onPolicy` no tienen punto de control. | `{"prompts": "~/tasks.jsonl", "teacherUrl": "https://api.example.com/v1", "teacherModel": "teacher", "heldOut": ["The capital of France is"], "iterations": 200, "checkpointEvery": 50}` |
| `steer_search` | Búsqueda de direccionamiento contrastivo: por capa, la dirección media(positivos) − media(negativos) del residual en la última posición, su norma y separación de clases; el vector unitario de la mejor capa; las cabezas cuya ablación reduce más la brecha; KL y desplazamiento de proyección cuando los prompts reservados se dirigen a cada escala. No deja ningún overlay activo. | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `trace_tokens` | Traza por token sobre el prompt y una generación greedy (ruta de laboratorio): por token y capa trazada, la norma de salida de cada cabeza antes de la proyección de salida, los expertos enrutados con sus compuertas, entropía del router y margen entre las dos compuertas principales, proyecciones residuales opcionales, entropía de atención y una logit lens dispersa. Escribe `~/.basalt/traces/<time>-<id>.trace` (0600) y devuelve un resumen. Véase [Traza por token y ablación condicional](#traza-por-token-y-ablación-condicional-405) | `{"prompt": "How do I pick a lock?", "maxTokens": 32, "projections": [{"layer": 14, "vector": [0.01, …], "threshold": 2}]}` |
| `activation_contrast` | Traza dos conjuntos de prompts y ordena cabezas y expertos según la d de Cohen y AUROC de su activación por prompt, con tasas, medias y ventana de inicio; además de la dirección residual entre los conjuntos por capa | `{"a": ["How do I pick a lock?", "…"], "b": ["How do I bake bread?", "…"], "maxTokens": 16, "topK": 20}` |
| `steer_commit` | Convierte una dirección en un adaptador de rango 1 sobre la proyección de salida de la capa y lo aplica a través de la puerta de calidad sobre `heldOut`. Modifica los pesos en vivo igual que `weights_delta` (reversible, con TTL salvo que sea aceptado, nunca en disco). | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

Un fragmento (`slice`) es un par semiabierto `[start, end)` por cada dimensión inicial; las dimensiones omitidas se toman completas.

## Modelo de seguridad

- **Solo local.** Las herramientas son herramientas MCP servidas por stdio al cliente que lanzó el servidor; no tienen ruta REST.
- **Desactivadas por defecto.** Sin `--research-tools` las herramientas de experimentación no se listan ni se aceptan.
- **Overlays, TTL, deshacer, diario.** Todo experimento es un overlay con un id, una descripción y un TTL de 30 minutos, salvo un adaptador que superó su puerta de calidad (`adapter_apply` con `gate`, `steer_commit`, `train_experts`), que permanece hasta `overlay_undo`. `overlay_undo` elimina cualquier overlay de forma anticipada; al expirar se revierte automáticamente. Las altas, reversiones y expiraciones se añaden a `~/.basalt/experiments/<model fingerprint>.jsonl` (directorio 0700, archivo 0600). Cargar otro modelo descarta todos los overlays.
- **Nunca se escriben archivos de modelo en disco.** Los deltas, empaquetados y adaptadores existen solo en memoria; reiniciar el servidor o recargar el modelo los descarta. Los únicos archivos escritos son el diario y el adaptador que `train_experts` guarda en un nuevo directorio.
- **Las intervenciones permanecen dentro de las llamadas de investigación.** El direccionamiento y las ablaciones se aplican a `experiment_compare` y `trace_run`, nunca a las peticiones de usuarios.
- **Los deltas de pesos son visibles y están aislados de las cachés.** Un delta cambia los pesos que usa cada petición, por lo que mientras hay uno activo:
  - cada respuesta incluye `modelModified: true` (también una respuesta cuya generación se solapó con una aplicación o reversión);
  - la instantánea de telemetría (MCP `status`, `GET /api/v1/stats`) lista los deltas activos en `activeDeltas`;
  - la reutilización de caché de prompts queda desactivada, incluidas las restauraciones desde la caché en disco, porque el KV en caché se calculó con los pesos antiguos;
  - las cachés de prompts en memoria se vacían al aplicar y al revertir, y una petición que estaba en curso durante cualquiera de los cambios no guarda su caché.
- **Validado en la frontera.** El tensor debe existir, el fragmento debe estar dentro de rango y no ser vacío (como máximo 16M elementos), debe haber un valor por elemento o uno para difundir, cada valor debe ser finito y `|scale| ≤ 10`. Solo un delta activo por tensor.
- **Solo tensores flotantes.** Los deltas escriben normas, sesgos y embeddings o proyecciones sin cuantizar (float16, bfloat16, float32). Se rechaza cualquier tensor de una capa cuantizada (tanto su peso empaquetado como sus escalas y sesgos): una ruta de escritura cuantizada queda fuera de alcance.

## Puerta de calidad y adaptadores (#332)

`gate_check` y el `gate` de `adapter_apply` comparan el modelo actual con el modelo intacto, dentro de una única llamada exclusiva al modelo que retira todos los adaptadores y deltas de pesos activos para la línea base y los vuelve a colocar. Las comprobaciones:

- **`probe perplexity`**: la perplejidad de la sonda de carga sobre su texto fijo debe mantenerse por debajo del techo de la familia o, cuando el modelo intacto ya está por encima, no subir más de `maxPerplexityRise` respecto al valor intacto.
- **`probe known answer …`**: cada respuesta conocida de la sonda de carga que el modelo intacto acierta debe seguir siendo correcta (valor 1 o 0, límite el resultado intacto).
- **`held-out mean KL`** y **`held-out perplexity rise`**: en cada prompt `heldOut`, se ejecuta la maquinaria de `experiment_compare` con el modelo intacto como línea base; la divergencia KL media de las distribuciones del siguiente token a lo largo del texto base debe ser como máximo `maxKL` (por defecto 0.5) y el aumento relativo medio de perplejidad de ese texto como máximo `maxPerplexityRise` (por defecto 0.1).
- **`target …`**: la respuesta greedy de cada prompt objetivo (primero la plantilla de chat, luego el prompt sin formato, 16 tokens) debe contener `expected`, sin distinguir mayúsculas y minúsculas.

Los valores no finitos se notifican como el mayor flotante finito y fallan. Adaptadores:

- Un adaptador de directorio se carga con el soporte LoRA de mlx-swift-lm; un adaptador de directorio a la vez, y sus tensores deben coincidir con las formas de las capas y el rango configurado.
- Un adaptador de rango 1 necesita un valor `input` por característica de entrada y un valor `output` por característica de salida de la capa; la capa debe ser una `Linear` o `QuantizedLinear` simple sin adaptador.
- Los adaptadores envuelven únicamente capas `Linear` y `QuantizedLinear` simples, de modo que revertir vuelve a colocar exactamente los mismos pesos y los logits coinciden bit a bit con el modelo intacto. No se escribe nada en disco.

## Cómo conduce esto a bucles de aprendizaje (#328 capa 3)

Estas herramientas son el instrumento de medición y el entorno de pruebas para una capa posterior: `distill.collect` recopila prompts y respuestas de referencia, `adapter.train` / `apply` / `evaluate` entrenan y prueban un pequeño adaptador, y `weights.propose` recibe una corrección propuesta externamente. Cada candidato se ejecuta primero aquí como un overlay o un delta, se compara con `experiment_compare` y solo se acepta si la sonda de carga, los logits de referencia, el ejecutor de paridad y la matriz de características siguen pasando. Nada en esta capa persiste cambios; el bucle de aprendizaje decide qué pasa a ser permanente, si es que algo lo hace.

## Empaquetado y desempaquetado de tensores (#332)

Tanto `tensor_pack` como `tensor_unpack` modifican una capa lineal en su lugar: leen su peso (descuantizando una capa cuantizada), lo convierten e intercambian el módulo entero (`Linear` ↔ `QuantizedLinear`), de modo que los bytes cambian realmente. La respuesta informa el coste de la conversión (valores ilustrativos):

```json
{"overlay": {"id": 7, "description": "tensor_pack model.layers.10.mlp.down_proj.weight: 4-bit affine g64 → 2-bit affine g64", "createdAt": "…", "expiresAt": "…"},
 "bytesBefore": 9437184, "bytesAfter": 5505024, "rmse": 0.0041, "maxAbsError": 0.031, "relativeError": 0.29}
```

`rmse`, `maxAbsError` y `relativeError` (‖W − Ŵ‖ / ‖W‖) comparan el peso descuantizado de la nueva capa con los valores que contenía la capa anterior. Desempaquetar al tipo flotante propio de las escalas es exacto (los tres valen 0).

El empaquetado es un overlay igual que un delta de pesos: el modelo se marca como modificado, la reutilización de caché de prompts se desactiva, `overlay_undo` (o el TTL de 30 minutos) devuelve la capa original bit a bit y nada se escribe en disco. Solo se empaquetan capas `Linear` y `QuantizedLinear` simples; las pilas de expertos MoE, embeddings, capas envueltas en LoRA y cualquier capa que ya tenga un delta u otro empaquetado se rechazan.

`experiment_compare` se ejecuta sobre los pesos en vivo, por lo que el efecto de un empaquetado es la diferencia entre una ejecución anterior y una posterior:

1. `experiment_compare` `{"prompts": ["2 + 2 =", "Name a prime above 100."], "maxTokens": 32}`: anota `baselineText` y `baselinePerplexity`.
2. `tensor_pack` `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "ternary"}`: anota el id del overlay y el error.
3. `experiment_compare` con los mismos argumentos: el cambio en `baselineText` y `baselinePerplexity` respecto al paso 1 es el efecto del empaquetado sobre los logits.
4. `overlay_undo` `{"id": 7}`, y después el paso 1 de nuevo produce la respuesta del paso 1 bit a bit.

## Introspección (`introspect`, #332 modo 1)

Con `--research-tools`, `introspect` permite que el modelo servido se estudie a sí mismo. Dado un objetivo, el servidor ejecuta el modelo por la ruta normal de generación ofreciendo las herramientas de radiografía como definiciones de herramientas, ejecuta cada llamada que realiza por la misma ruta de investigación controlada que las herramientas MCP, le devuelve el resultado y vuelve a preguntar.

| Argumento | Significado |
|---|---|
| `goal` (obligatorio) | Qué averiguar, p. ej. `"find which heads attend to the last tool call"` |
| `maxSteps` | Turnos del modelo antes de que el bucle se detenga, 1–32 (por defecto 8) |
| `allowInterventions` | Ofrecer también `intervene`, `experiment_compare` y `overlay_undo` (por defecto false; las herramientas de solo lectura siempre se ofrecen) |
| `resultBudget` | Caracteres de cada resultado de herramienta devueltos al modelo, 200–20000 (por defecto 2000); los resultados más largos se recortan y se marcan |

- **Una llamada por turno.** El bucle se detiene en una respuesta sin llamada a herramienta, tras `maxSteps`, o cuando el modelo repite una llamada que ya hizo (misma herramienta y argumentos, ignorando el orden de las claves); `stop` en el informe indica cuál fue.
- **Hipótesis.** Se pide al modelo que escriba cada hipótesis con líneas `Hypothesis:` / `Test:` / `Outcome:`. Se analizan de forma tolerante (markdown, numeración, `Result:` por `Outcome:`), y una hipótesis reformulada reemplaza a la anterior conservando su `Test:` o `Outcome:` si la reformulación omite esa línea.
- **Los errores vuelven al modelo.** Una llamada a herramienta mal formada o fallida se convierte en un resultado `error: …` que el modelo ve; una generación fallida o una cancelación termina el bucle.
- **Las intervenciones nunca sobreviven al bucle.** Todo overlay creado durante el bucle se revierte cuando termina, también ante un error o una cancelación; `undone` los lista. Si el servidor rechaza la limpieza (ocupado), `cleanupFailure` lo indica y esos overlays expiran en su TTL. `overlay_undo` rechaza overlays que existían antes del bucle.
- **Un turno fallido conserva el informe.** Si una generación falla a mitad del bucle, los pasos realizados hasta ese momento regresan con `stop: "failed"` y el error en `failure`.
- **Diario.** Cada paso añade una línea `introspect` (modelo, paso, herramienta, argumentos y el inicio del resultado) a `~/.basalt/experiments/introspect.jsonl`.

El informe contiene `goal`, `steps` (el texto del modelo y, para una llamada, su `tool`, `arguments` y `result` truncado), `hypotheses` (`statement`, `test`, `outcome`), `answer`, `stop`, `failure`, `undone` y `cleanupFailure`.

```json
{"goal": "find which heads attend to the last tool call", "maxSteps": 6, "allowInterventions": true}
```

## Autoentrenamiento (`train_experts`)

Una sola llamada ejecuta todo el bucle con prioridad en segundo plano a través de la puerta de generación, reteniendo el modelo únicamente para el entrenamiento en sí:

1. **Tareas.** `count` + `seed` generan problemas enteros de `+`, `-`, `*` con respuestas conocidas (la misma semilla produce los mismos problemas); en su lugar, `file` lee líneas JSONL `{"prompt": "...", "answer": "..."}`. Una quinta parte de las tareas (al menos una) se reserva y nunca se entrena sobre ella.
2. **Recopilación.** El modelo responde cada tarea de forma greedy a través de la plantilla de chat (sin pensamiento, 64 tokens). Con `teacher` (URL base de un servidor compatible con OpenAI; `teacherModel` indica su modelo, `BASALT_TEACHER_API_KEY` se envía como token bearer solo cuando el esquema, host y puerto del profesor coinciden con `BASALT_TEACHER_URL`, para que un cliente no pueda enviar la clave a otro sitio), el profesor responde también las tareas de entrenamiento.
3. **Verificación.** Una respuesta se conserva solo si su entero final (ignorando comas) coincide con una respuesta numérica conocida, o si su última línea, ignorando mayúsculas/minúsculas y puntuación circundante, coincide con una respuesta de texto (o termina en `: <answer>`). Las respuestas erróneas, vacías o mal formadas se descartan. **Nunca se aprende nada sin verificar:** si ninguna respuesta supera la verificación, el informe devuelve `verified: 0`, no se entrena ningún adaptador, no se escribe nada y el modelo permanece intacto.
4. **Entrenamiento.** Se entrena un LoRA (`rank`, Adam con `learningRate`, `iterations`) mediante `LoRATrain` del fork sobre los pares prompt + respuesta verificados tal como los vio el modelo, en las proyecciones de atención y densas o de experto compartido de `layers` (por defecto todas). No hay LoRA por experto disponible — las capas de expertos `SwitchLinear` del fork no aceptan adaptadores — por lo que `experts` se rechaza con un error. También se rechaza un modelo que ya lleve un adaptador. Cancelar la llamada detiene el entrenamiento en la siguiente iteración y deja el modelo sin cambios.
5. **Guardado.** `adapter_config.json` + `adapters.safetensors` (el formato de adaptadores de mlx-lm) se guardan en `output` o en `~/.basalt/adapters/<model fingerprint>/<timestamp>` (0700). `output` no debe existir todavía, y se rechaza uno dentro del directorio del modelo: no se sobrescribe nada ni se escribe nada junto a los pesos fuente.
6. **Medición y aplicación.** Se mide la precisión reservada con el adaptador colocado, se retiran las capas de entrenamiento y el adaptador se aplica a través de `applyAdapter` con la puerta de calidad construida con `heldOut`, `maxKL` y `maxPerplexityRise`: pasa a ser un overlay marcado sin expiración que `overlay_undo` elimina, o es rechazado por la puerta.

El informe lista `collected` (respuestas reunidas para entrenar), `verified`, `losses` (pérdida media de entrenamiento por paso reportado, como máximo 20), `accuracyBefore` / `accuracyAfter` en las tareas reservadas, `adapterDirectory` y el `verdict` de la puerta.

## Direccionamiento contrastivo ("meditación", #332 modo 2)

Esto es una búsqueda dirigida de direcciones de activación, no autoconciencia. Nada aquí significa que el modelo reflexione sobre sí mismo: basalt ejecuta dos conjuntos de prompts por el modelo, resta los promedios de los estados ocultos y mide qué hace la diferencia.

`steer_search` ejecuta cada prompt positivo y negativo en una pasada forward sondeada y conserva el residual tras cada capa en la última posición.

- **Dirección por capa:** media(positivos) − media(negativos). `norm` es su longitud; `separation` es la brecha entre las medias de clase de las proyecciones sobre la dirección unitaria (igual a `norm`) dividida por la desviación estándar combinada de esas proyecciones. La mejor capa tiene la mayor separación, y `direction` es su vector unitario (tamaño oculto).
- **Cabezas:** se realiza una ablación por turnos de cada cabeza de atención de la mejor capa y de la capa anterior (la intervención `ablate_head`), y `effect` es cuánto se reduce la brecha de proyección positivo-menos-negativo en la mejor capa. Las cabezas de capas posteriores no pueden cambiar ese residual y no se prueban. Se informan como máximo `topHeads`, en orden decreciente de efecto. Esto supone una pasada forward por cabeza y prompt, así que mantén pequeños los conjuntos de prompts en modelos grandes o pasa `topHeads: 0`.
- **Pruebas:** para cada escala, los prompts reservados se dirigen con `direction × scale` después de la mejor capa (la intervención `steer`, solo para investigación). `klDivergence` es la KL media de la distribución del siguiente token frente a la ejecución sin direccionamiento; `projectionShift` es el cambio medio de la proyección sobre `direction` leída tras la última capa. En la capa dirigida ese desplazamiento es la escala por construcción, así que la última capa muestra cuánto del empuje sobrevive.
- No se registra ningún overlay: las ablaciones y el direccionamiento existen únicamente dentro de la llamada.

`steer_commit` hace permanente para la sesión una dirección que ayudó, a través de la puerta. Elige la proyección densa de la capa que escribe en el residual (`mlp.down_proj`, si no `feed_forward.w2`, si no `self_attn.o_proj`; se rechaza una capa MoE sin proyección densa), mide la entrada media `m` de ese módulo en la última posición sobre los prompts `positive`, y llama a la ruta de adaptadores con `.rankOne(module: <full module path>, input: m / (m·m), output: direction, scale: scale)`, de modo que la actualización es ΔW = scale · direction ⊗ m / (m·m).

La aproximación, expuesta con claridad:

- La actualización suma `scale · (x·m / m·m) · direction` a la salida de la proyección para cada token. Eso equivale exactamente al vector de direccionamiento solo donde la entrada del módulo `x` tiene la misma proyección sobre `m` que la media positiva; otros tokens reciben más, menos o lo contrario. `steer` suma el mismo vector en todas las posiciones.
- Solo es exacta en el residual cuando la salida de la proyección se suma directamente al flujo residual (bloques pre-norm como Qwen y LFM). Donde sigue una norma tras la proyección (las normas post-attention y post-feedforward de Gemma), el vector se reescala y solo se conserva aproximadamente su dirección. Cuando se usa el respaldo `self_attn.o_proj`, el vector también pasa por el MLP de la capa antes de alcanzar el residual posterior a la capa.
- El adaptador se acepta únicamente si la puerta pasa (`maxKL`, `maxPerplexityRise` sobre `heldOut`); ejecuta primero pruebas con `steer_search` y confirma una escala cuya KL haya sido aceptable.

## Traza por token y ablación condicional (#405)

El procedimiento para un comportamiento como el rechazo: trazar → contrastar → ablación condicional → comparar.

1. **`trace_tokens`** sobre unos cuantos prompts que muestren el comportamiento. `ForwardProbe.headOutputs` del fork entrega la salida de atención de cada cabeza antes de la proyección de salida (antes de la compuerta de salida de Qwen3.5), de modo que la atención fusionada permanece activa; `routerTopK` entrega los expertos y las compuertas que la capa combina realmente (incluidos sesgo de experto, group top-k y escalado). Por token y capa trazada, el archivo guarda:
   - `heads`: norma L2 de la salida de cada cabeza;
   - `experts` (enteros) y `gates`; `routerEntropy` (softmax sobre los logits del router) y `routerMargin` (compuerta superior menos la segunda);
   - `projection`: residual · direction después de la capa, para cada entrada de `projections`;
   - `attentionEntropy` (opcional, atención explícita, más lenta) y `lens`/`lensProbabilities` (logit lens top-k opcional).

   Formato de archivo: una línea de cabecera JSON (`layout.floats` y `layout.ints` listan segmentos `{layer, field, width}` en orden de columnas), una línea JSON por token (`position`, `id`, `token`, `generated`, `ints`), y después `tokens × floatsPerToken` valores float32 little-endian. Límites comprobados antes de ejecutar: prompt más `maxTokens` ≤ 2048 tokens, capas dentro del modelo, como máximo 16 proyecciones del tamaño oculto en capas trazadas y un tamaño estimado ≤ `maxFileBytes` (≤ 64 MiB; el tamaño codificado se vuelve a comprobar antes de escribir).
2. **`activation_contrast`** con el conjunto `a` (muestra el comportamiento) y el conjunto `b` (no lo muestra). La activación media por prompt de cada componente (norma de cabeza; compuerta de experto cuando se enruta, si no 0) proporciona la d de Cohen y AUROC; `rateA`/`rateB` cuentan tokens donde la norma de una cabeza está por encima de su media combinada o donde el experto fue enrutado; `onsetA`/`onsetB` promedian los `onsetWindow` tokens de prompt antes de que comience la generación. `direction` en `bestLayer` es la dirección residual entre los conjuntos.
3. **`intervene` con `when`**: realiza la ablación de una cabeza o experto clasificados (o aplica steering con escala negativa para restar una dirección) solo cuando se activa un disparador — la proyección sobre la dirección de contraste por encima de un umbral, o un experto enrutado con una compuerta por encima de `gate` — opcionalmente durante `window` tokens desde el primer disparador. La capa de la acción debe estar después de la capa del disparador; el disparador se lee en la misma pasada forward (una sincronización de GPU por overlay condicional por forward), de modo que la edición se aplica al token que la dispara, tanto en el prompt como en la generación. Un disparador que nunca se activa deja los logits bit a bit idénticos.
4. **`experiment_compare`** con el overlay condicional, frente a un overlay estático `ablate_*` del mismo componente y un adaptador `steer_commit`: KL, perplejidad, los textos y `triggerRate` (la proporción de tokens a los que se aplicó la edición condicional).

Aún no conectado: la lectura de System One como disparador o etiquetador, trazas muestreadas (no greedy), mensajes de chat como entrada (aplica tú mismo la plantilla) y un ejecutor de evaluación dedicado para tasa de rechazo, precisión y velocidad de decodificación.

## Incorporaciones recientes (`v0.12`)

- **Gestión automática de BOS en trazas X-Ray sin formato (`#630`):** `trace_run`, `trace_tokens`, `experiment_compare`, `contrast`, `activation_contrast`, `steer_search` y `TokenAttribution` comparten `ModelProbe.encoded(_:_:)` de modo que los tokenizadores cuyo `encode(addSpecialTokens: true)` omite `bosToken` (como Gemma 4) sigan recibiendo `<bos>` en la posición 0, preservando las capas attention-sink de BOS (`L43–L56`) sin doble BOS.
- **Logprobs por token, atribución y riesgo de alucinación (`#613`, `#614`):** `POST /v1/xray/tokens` y las rutas de generación exponen logprobs por token, entropía top-k, atribución por capa/cabeza/MoE y señales de riesgo de alucinación.
- **Cirugía mecanística y atribución de influencia (`#600`, `#622`):** parcheo por grupo KV de GQA, proyección al espacio nulo cognitivo, protección anti-steering, descomposición de características `MAO-SNMF`, atribución de cabezas `Score d'influence` (`HeadInfluenceCore`) y enrutamiento calibrado por temperatura (`CalibratedRouterGate`).
- **Gobernanza HITL (`#626`):** `HITLGovernanceCore` / `HITLApprovalActor` controlan acciones MCP de alto impacto mediante `/v1/mcp/approvals`, emparejados con `SymbolicKnowledgeGraph` del LAAS-CNRS.
