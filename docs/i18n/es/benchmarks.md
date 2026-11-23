# Funciones y benchmarks

> Languages: [English](../../benchmarks.md) · [Русский](../ru/benchmarks.md) · [中文](../zh/benchmarks.md) · **Español** · [Français](../fr/benchmarks.md) · [日本語](../ja/benchmarks.md) · [Deutsch](../de/benchmarks.md) · [한국어](../ko/benchmarks.md) · [Bahasa Indonesia](../id/benchmarks.md)

## Funciones de un vistazo

Cada aceleración es opcional (opt-in) y mantiene idéntica la salida greedy salvo que la fila indique lo contrario. Cifras: M4 Pro 24 GB conectado a la corriente, compilación release.

| Función | Flag | Qué hace | Medido |
|---|---|---|---|
| Prompt lookup (decodificación especulativa) | `serve --speculative-lookup` | propone borradores de hasta 16 tokens copiados del prompt o de la salida, verificados en una pasada forward | 1.84–2.66× en ediciones de código (Bonsai 2 27B), 3.45–3.62× (Gemma 4 12B); ~1.0× en prosa |
| Borrador con cabezal MTP | `serve --mtp [--mtp-block N]` | genera borradores con el cabezal de predicción multitoken del modelo (`mtp.safetensors`, p. ej. paquetes MTPLX); exacto a cualquier temperatura. Las rondas empiezan con un token propuesto y crecen hasta N−1 (por defecto 4) tras cuatro rondas aceptadas por completo; un rechazo vuelve a uno | 1.30× en código, 1.04× en prosa, 1.26× en razonamiento (Bonsai 2 27B, T=0.7); un turno posterior reutiliza la caché de prompts igual que la decodificación normal: TTFT 12.5 s → 0.75 s en un turno de 1,149 tokens (Bonsai 2 27B) |
| Ternary Bonsai 2 | (automático) | carga checkpoints ternarios rotados con Hadamard (`prism_hadamard_qwen35`) | modelo de 27B en 7.4 GB |
| Embeddings y rerank | `serve --embedding-model`, `--rerank-model` | OpenAI `/v1/embeddings`, Cohere/Jina `/v1/rerank` junto al modelo de chat | bge-small 229–299 textos/s, Qwen3-Embedding-0.6B 20 textos/s (lote 32) |
| Caché de prompts en disco | `serve --prompt-cache-disk-mb N` | una sesión reanudada o un servidor reiniciado restaura el prefijo coincidente más largo en lugar de volver a hacer prefill | — |
| Cuantización y zonas de caché KV | `serve --kv-bits 4\|8\|auto\|k8v4`, `--kv-zones` | permite contextos más largos en la misma memoria | pico de prompt de 16k: 1.03 GB frente a 3.11 GB (Qwen3-4B, zonas 4-bit) |
| Peticiones concurrentes | `serve --max-concurrent N` | cada petición en curso reserva su espacio KV; la que no cabe recibe 503 `Retry-After` en lugar de quedarse sin memoria | — |
| Muestreadores DRY y XTC (experimental) | `serve --experimental-samplers` | campos de petición `dry_*` / `xtc_*` | no es una herramienta antibucle: en un modelo de 2 bits DRY oculta los bucles tras datos corruptos (#177) |

## De dónde salen las cifras del README

Cada cifra de la primera pantalla del README proviene de una ejecución registrada, y el PR que incorporó la función contiene la tabla, el commit y la ruta del registro. M4 Pro 24 GB, compilación release, salvo que el PR indique otra cosa.

| Afirmación | Fuente |
|---|---|
| Qwen3-4B 78–89 tok/s; Ternary Bonsai 2 27B en 7.4 GB a 20.9–21.5 tok/s, 31–38 con prompt lookup en código | [#239](https://github.com/goodmai/basalt/pull/239), a partir de [#218](https://github.com/goodmai/basalt/pull/218), [#231](https://github.com/goodmai/basalt/pull/231), [#232](https://github.com/goodmai/basalt/pull/232) |
| Bonsai 2 27B con su cabezal MTP, 27.1 tok/s en código | [#263](https://github.com/goodmai/basalt/pull/263) |
| Prompt lookup 3.45–3.62× en Gemma 4 12B, salida greedy sin cambios | [#265](https://github.com/goodmai/basalt/pull/265) |
| Segundo turno de agente alrededor de 1 s en lugar de 36 s en Ornith-1.5-9B (prompt de 12.6k tokens) | [#82](https://github.com/goodmai/basalt/pull/82) |
| Pico de prompt de 16k de 1.03 GB en vez de 3.11 GB con zonas KV en Qwen3-4B | [#159](https://github.com/goodmai/basalt/pull/159) |
| Mezcla de modelos: todas las etiquetas correctas, $0.035 por 1000 peticiones, p95 4.8 s frente a 10.0 s | [#418](https://github.com/goodmai/basalt/pull/418) |
| App de Mac: ventana en caliente 257 ms, 0 despertares en reposo, 7.4 ms p95 por token transmitido, 40 ms consulta rápida | [#463](https://github.com/goodmai/basalt/pull/463) |
| Un juez de 9B junto a Gemma 4 12B en 24 GB: 4-bit cabe (15.3 GB), 98.9% con un 77% menos de tokens LLM | [#465](https://github.com/goodmai/basalt/pull/465) |
| Qwen3.6-35B-A3B-8bit (37.7 GB) transmitido desde SSD en 24 GB a 8.77 tok/s con 0 swap | [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478) |
| `--experimental-pv-tune` poscuantización: recuperación de perplejidad del 8.7–29% en 2–3 bits en menos de 3 minutos | [#486](https://github.com/goodmai/basalt/pull/486) |
| Trellis bitshift QTIP `--experimental-qtip`: `Qwen3.8-27B` 2.5 bpw en 10.00 GB supera a 3-bit afín y PV-Tuning (PPL 16k 4.621 vs 4.702 / 4.652); 2.0 bpw en 8.48 GB reduce la PPL de 2 bits en un 72.1% (5.345 vs 19.146) | [#494](https://github.com/goodmai/basalt/pull/494) |
| Ejecución nativa de `.gguf` GSQ-RCO / `i-quant` en Metal (`IQ4_XS` / `IQ3_S`) | [#433](https://github.com/goodmai/basalt/pull/433) |
| Recursive `Flint-40B-A2.5B` (`oQ3e`) a 67.02 tok/s en decodificación (273.5 tok/s en prefill) en 17.61 GB en M4 Pro 24 GB | [#522](https://github.com/goodmai/basalt/pull/522) |
| `Zyphra/ZONOS3-Mini-hf` TTS neuronal a 44.1 kHz en 2.24 GB con RTF 0.228–0.263 (3.8–4.4× tiempo real) en EN/DE/RU | [#527](https://github.com/goodmai/basalt/pull/527) |
| 84 perfiles de Compose por niveles de hardware (`16–256 GB × 12`) con recomendación automática de nivel de RAM en `Basalt.app` | [#528](https://github.com/goodmai/basalt/pull/528) |
| `MACaQSwitchLinear` WHT + `gatherQuantizedMM` en una pasada: kernel MoE `3.41×` (`2.06 → 0.60 ms`), `+82%` de extremo a extremo en Gemma 4 26B-A4B (`49.2–54.8 tok/s`), `64–75 tok/s` en Ornith 35B-A3B 3-bit | [#628](https://github.com/goodmai/basalt/pull/628) |
| `Qwen3.5-9B-PV-MACaQ-4bit`: paridad exacta `4/5` en Ladder con 8-bit, `+52%` decodificación (`43.2 tok/s` p50), `−33%` RAM (`6.68 vs 9.97 GB`), deriva máx. por capa `3.37%` | [#545](https://github.com/goodmai/basalt/pull/545), [#630](https://github.com/goodmai/basalt/pull/630) |
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`: `15.1 GB` RAM (`18.09 GB` pico) en Mac de 24 GB, `4/5` Ladder + `4/6` Lunar a `14.1 tok/s` p50 (ref. 4-bit da OOM) | [#628](https://github.com/goodmai/basalt/pull/628), [#630](https://github.com/goodmai/basalt/pull/630) |

Los modelos verificados con golden del README son comprobados por `scripts/check_readme_models.swift` en CI: todo modelo listado allí debe enlazar a un fixture existente en `Tests/BasaltTests/Fixtures/golden`.

Una página estática autocontenida de benchmarks ([`docs/benchmark-page/index.html`](../../benchmark-page/index.html)) y un manifiesto de colección de Hugging Face listo para revisión ([`docs/benchmark-page/hf-collection.json`](../../benchmark-page/hf-collection.json)) se generan a partir de estas tablas, `benchmarks/` y los fixtures golden mediante `swift scripts/build_benchmark_page.swift` (y se verifican en CI con `--check`; pasa `--logs logs` en local para añadir tablas `summary` del ejecutor). La publicación externa sigue siendo una decisión humana.

## Benchmarks

Dos bancos de pruebas, ambos apuntando a un servidor en ejecución y ambos informando cifras que puedes reproducir en tu propia máquina.

**La escalera a la Luna** — `benchmarks/ladder/run.py`. Cinco peldaños, cada uno estrictamente más difícil que el anterior, que culminan en una misión completa Tierra-Luna. Nada se califica por lo bien que suene la respuesta: se ejecuta el propio código del modelo frente a entradas que el prompt nunca le mostró, así que una respuesta verosímil pero errónea igualmente falla.

| Peldaño | Tarea | Qué evalúa realmente |
|---|---|---|
| 1 | Bicuadrática, raíces reales | forma cerrada más los casos límite: ninguna, repetida, cero |
| 2 | Bicuadrática, raíces complejas | manejo de ramas — las cuatro raíces, con multiplicidad |
| 3 | Coeficientes de Fourier | integración numérica, comprobada en diente de sierra, onda cuadrada, coseno |
| 4 | Altura de bombeo | unidades (mm vs m, L/s vs m³/s) + Colebrook, que no tiene forma cerrada |
| 5 | **Misión Tierra → Luna** | velocidad de escape a partir de `sqrt(2ηQ)`, Tsiolkovsky por etapa, propagación de cuatro cuerpos, frenado lunar, alunizaje suave — cada constante está en el prompt, nada recuperable de un libro de texto |

El peldaño superior se puntúa sobre seis invariantes en lugar de una sola respuesta, de modo que un modelo que acierta la energética pero nunca propaga la trayectoria aún muestra una puntuación. Su prompt está en `benchmarks/ladder/lunar_prompt.md`; los valores de referencia en `lunar_task.md` nunca se muestran al modelo.

`--warmup` antepone tres pruebas rápidas sencillas (álgebra, Fibonacci, traducción) que cualquier modelo instruct debería superar. No forman parte de la puntuación de la escalera.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, reglas oficiales de pass@2. El corpus no viene incluido; descárgalo primero en `benchmarks/arc-agi/dataset/` (consulta `benchmarks/arc-agi/ARC_AGI.md`).

### Resultados

Medidos en un Mac de serie M con 24 GB de memoria unificada. `thinking` es el bloque `<think>` de la plantilla de chat: activado por defecto para modelos de razonamiento, desactivado con `--reasoning-effort none`.

| Modelo | Calentamiento | Escalera | Peldaño superior | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` (thinking off, 24 GB) | 3/3 | **4/5** | 4/6 invariantes | — | 11.6–13.5 (p50 14.1) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` (thinking off, 24 GB) | 3/3 | **4/5** | 0/6 (longitud) | — | 9.4–11.4 (p50 11.6) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (thinking off) | 3/3 | **4/5** | 0/6 (prosa) | — | 35.6–41.6 (p50 43.2) |
| `mlx-community/Qwen3.5-9B-8bit` (ref, thinking off) | 3/3 | **4/5** | 0/6 (prosa) | — | 24.7–27.3 |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` (thinking off) | 3/3 | **4/5** | 5/6 invariantes | — | 56.0–63.4 |
| `goodmai/gemma-4-12B-it-qat-4bit` (thinking off) | 3/3 | **4/5** | **6/6 PASS** | — | 18.8–22.2 |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` (thinking off, `#628`) | 3/3 | 3/5 | 0/6 | — | 49.2–54.8 |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` (thinking off, `#628`) | 3/3 | 1/5 | 0/6 | — | 64.0–74.7 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking off) | 3/3 | 1/4 ² | no ejecutado | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking on) | 2/3 | 0/4 ² | no ejecutado | no ejecutado | ~43 |

Ornith 9B peldaño a peldaño, con thinking desactivado:

| Peldaño | Resultado | Detalle |
|---|---|---|
| 1 · bicuadrática real | ✅ | raíces correctas, incluidos los casos repetidos y vacíos |
| 2 · bicuadrática compleja | ❌ | `x⁴ + 1` devuelve 2 raíces en lugar de 4 |
| 3 · Fourier | ❌ | `a0 = 10.86` para `f(x) = x`, se esperaba 0 — la integración está mal, no la fórmula |
| 4 · altura de bombeo | ❌ | `v = 0.0298` m/s, se esperaba 1.79 — nunca convirtió mm a m |
| 5 · misión lunar | — | no ejecutado |

Calentamiento, thinking off: álgebra ✅ (163 tok, 4 s), Fibonacci ✅ (757 tok, 16 s), traducción ✅ (303 tok, 7 s — alemán idiomático, francés parafraseado en lugar de traducido).

² Los peldaños 1 y 2 se midieron antes de dividir la escalera en prompts separados: un único prompt pedía ambas funciones; la mitad real fue correcta y la mitad compleja no. El peldaño 5 aún no se ha ejecutado en este modelo.

¹ La columna ARC-AGI es una ejecución parcial: tres tareas de entrenamiento (`007bbfb7`, `00d62c1b`, `017c7c7b`), ninguna resuelta con pass@2, 330 s en total. Se detuvo antes de las dos restantes, así que tómalo como una prueba rápida más que como una puntuación.

Los propios validadores también se verifican, sin modelo ni GPU:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

Con thinking **activado**, las tres tareas difíciles y la traducción terminaron con `finish_reason: length` — el presupuesto se consumió dentro del bloque `<think>` y nunca se emitió ninguna respuesta. Eso es un fallo de presupuesto, no un techo de capacidad, razón por la cual la tabla anterior informa los dos modos por separado.

### Reproducir

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Añadir un modelo a la tabla es una ejecución de cada uno — los PRs con nuevas filas son bienvenidos.
