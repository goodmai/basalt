# Basalt para Mac

> Languages: [English](../../mac-app.md) · [Русский](../ru/mac-app.md) · [中文](../zh/mac-app.md) · **Español** · [Français](../fr/mac-app.md) · [日本語](../ja/mac-app.md) · [Deutsch](../de/mac-app.md) · [한국어](../ko/mac-app.md) · [Bahasa Indonesia](../id/mac-app.md)

Una aplicación nativa en torno a basalt: una ventana de chat y una ventana Monitor, sobre un motor local que la aplicación inicia y supervisa. El diseño y las fases están en #434. Esta es la fase 1.

## Compilación y ejecución

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` firma con el entorno de ejecución reforzado (hardened runtime) para distribución.
- `--bin PATH` ensambla a partir de binarios ya compilados.
- `--dry-run` imprime el plan.

Desde una carpeta de compilación sin empaquetar, apunta la aplicación al motor: `BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac`.

## Cómo funciona

- **El motor es un proceso independiente:** el binario empaquetado `basalt serve`, en `Contents/Helpers`.
  - La aplicación lo inicia en un puerto loopback libre (`--port 0 --announce-ready`) y le pasa un token por ejecución en `BASALT_API_KEY`.
  - Se comunica con el motor a través de la misma API REST que usan los editores y scripts.
- **Fallos:** un fallo del motor no provoca un fallo de la aplicación. El supervisor lo reinicia tras 1 s, luego 2 s y así sucesivamente hasta 30 s, y se detiene tras tres fallos en un minuto mostrando la última salida del motor.
- **Al salir:** cerrar la aplicación detiene el motor de forma ordenada: SIGINT y, después de 10 s, SIGKILL.
- **Reinicio de la aplicación:** la aplicación adopta un motor que siga ejecutándose a partir de su archivo de registro (`~/Library/Application Support/Basalt/engine.json`, modo 0600), por lo que el modelo no se vuelve a cargar.
- **Las conversaciones** se almacenan con SwiftData en `~/Library/Application Support/Basalt/Conversations.store`. Una respuesta interrumpida por un fallo queda marcada como detenida.
- **El modelo** se configura en Ajustes (por defecto `mlx-community/gemma-4-12B-it-qat-4bit`). Se aplica en el próximo inicio del motor.

## Barra de menús y consulta rápida

- **El motor se inicia con la aplicación**, no con la ventana. El elemento de la barra de menús (un cubo) permanece presente incluso cuando todas las ventanas están cerradas. Muestra el estado y el modelo del motor e incluye estas acciones:
  - nueva conversación y consulta rápida;
  - las cinco conversaciones más recientes;
  - iniciar, reiniciar y detener el motor;
  - Monitor, Ajustes y Salir.
- **Consulta rápida:** ⌥Space abre un panel flotante desde cualquier aplicación, incluso a pantalla completa, sin quitarle el foco a esa aplicación.
  - Escribe una pregunta y pulsa ↩. La respuesta se transmite en flujo continuo en el panel.
  - ⌘↩ abre el intercambio como una conversación normal en la ventana principal.
  - Esc cierra el panel y descarta el intercambio. Hasta pulsar ⌘↩, nada de la consulta rápida se guarda.
  - El atajo se registra con `RegisterEventHotKey`, que no necesita permiso de Accesibilidad. Se puede desactivar en Ajustes; ese cambio se aplica en el próximo inicio.
- **Ajustes:**
  - General: el modelo, consulta rápida y abrir al iniciar sesión (`SMAppService`; funciona una vez que la aplicación está firmada y en Aplicaciones).
  - Claves en la nube (Cloud Keys): variables de entorno, como `OPENROUTER_API_KEY`, cuyos valores se guardan en el Llavero (servicio `app.basalt.mac`). El motor las recibe en su próximo inicio, para modelos en la nube de compose. La aplicación nunca las escribe en disco.
- **Restauración de estado:** la conversación seleccionada vuelve tras reiniciar la aplicación. También lo hace un borrador sin enviar, por conversación, guardado en `user defaults`.

## Modelos

- **La ventana Modelos** (⌥⌘L, o Modelos en la barra de menús) lista los modelos en la caché de Hugging Face (`~/.cache/huggingface/hub`) con su tamaño y un veredicto de compatibilidad para este Mac, obtenido de `GET /v1/models/preview`.
- **Cargar** cambia el motor a ese modelo (`POST /v1/models/load`) y lo convierte en el modelo con el que inicia el motor. Un modelo cuyo veredicto es "does not fit" (no cabe) no se puede cargar desde la aplicación; la ventana explica el motivo.
- **Descargar** acepta un identificador de repositorio como `mlx-community/gemma-4-12B-it-qat-4bit`. La aplicación descarga en su propio proceso, muestra el progreso y recuerda las descargas inacabadas. En el próximo inicio las reanuda desde los archivos parciales. Una descarga fallida espera a Reintentar.
- **Eliminar** pregunta primero y luego borra la carpeta del modelo de la caché. El modelo cargado actualmente no se puede eliminar.

## Adjuntos

- **Adjunta** con el clip o soltando archivos sobre la conversación: PDF, imágenes (PNG, JPEG, HEIC, TIFF, GIF, WebP) y audio (WAV, MP3, M4A, AAC, FLAC, CAF), hasta 32 MB cada uno.
- **Un mensaje con adjuntos pasa por `/v1/pipeline`.** El motor lee documentos e imágenes con su modelo OCR, o con Apple Vision si no hay uno. Transcribe el audio con su modelo de voz o con el modelo de chat. Después, el modelo de chat responde al mensaje (o a "Summarize the attached files." cuando el mensaje está vacío).
- **El texto percibido permanece en la conversación.** Cada adjunto aparece como una etiqueta (chip) que revela el texto que leyó el motor. Los mensajes posteriores se envían con ese texto delante de la pregunta, de modo que los seguimientos pueden hacer referencia al archivo.
- **El recibo** debajo de la respuesta lista cada etapa del pipeline con su modelo y milisegundos.

## Inspector

- **⌥⌘I**, o hacer clic en una respuesta, abre el inspector junto a la conversación.
- **Conversación:** el modelo y un prompt del sistema para esta conversación. Se guarda con la conversación y se envía antes de cada mensaje. Un prompt vacío lo limpia.
- **Respuesta:** el recibo de la respuesta seleccionada: modelo, tokens de prompt y de generación, tiempo hasta el primer token, velocidad y cada etapa del pipeline con su modelo y milisegundos.
- **Adjuntos:** el texto que el motor leyó de cada archivo de la pregunta, página por página.

## Atajos, Servicios y el Dock

- **Atajos y Spotlight:** Ask Basalt, Summarize File, Transcribe Audio y Switch Model son App Intents. Atajos los ejecuta con la aplicación en segundo plano. Cada uno espera hasta que el motor esté listo (hasta tres minutos), y Switch Model rechaza un modelo que no quepa. `scripts/bundle_app.swift` extrae los metadatos de los intents a `Contents/Resources/Metadata.appintents` con `appintentsmetadataprocessor`, porque SwiftPM no lo hace.
- **Servicios:** selecciona texto en cualquier aplicación y elige Servicios → Ask Basalt. Se abre una nueva conversación con ese texto como pregunta.
- **Lanzamiento en un clic con `basalt://` desde Hugging Face:** `Basalt.app` registra el esquema de URL `basalt` (`CFBundleURLTypes`). Abrir `basalt://run?model=<owner/repo>[&role=chat|mtp|moe-stream|ocr|asr|decide|embed|rerank]` — o hacer clic en **Run with Basalt** en cualquier tarjeta de modelo de [`goodmai`](https://huggingface.co/goodmai) — valida el ID del repositorio de Hugging Face, actualiza la elección de modelo guardada, trae Basalt al frente y cambia el motor supervisado a ese modelo o rol de pipeline.
- **Dock:** un PDF, imagen o archivo de audio soltado sobre el icono del Dock inicia una conversación con ese archivo adjunto.
- **Notificaciones:** cuando una respuesta termina mientras Basalt está en segundo plano, una notificación muestra sus primeras líneas. El permiso se solicita una sola vez.

## Sistemas compose y la nube

- **Ajustes → Motor:** ejecuta un solo modelo o un sistema compose desde un archivo (`basalt serve --compose <file>`). "Restart the Engine with This" detiene el motor y lo inicia con los nuevos argumentos. La elección también se usa en el próximo lanzamiento.
- **La adopción se realiza por argumentos:** la aplicación adopta un motor que quedó en ejecución solo si se inició con los mismos argumentos. Tras un cambio de modelo o de archivo compose, se inicia un nuevo motor. Salir también detiene un motor adoptado.
- **Monitor:** para un sistema compose, muestra su nombre, pipelines y gasto en la nube hoy, y por modelo su estado (`ready`, `loading`, `no-key`, `over-budget`, etc.), política de retención, gasto y límite, y cualquier rechazo. Los modelos en la nube llevan un símbolo de nube.
- **La marca de nube:** una respuesta que usó un modelo en la nube indica "☁︎ cloud" junto con su coste en la línea del recibo y en el inspector. Todo lo que no lleva la marca se ejecutó en este Mac.

## Accesibilidad, idiomas y actualizaciones

- **VoiceOver:**
  - Todo control que solo muestra un icono tiene un nombre hablado: el elemento de la barra de menús, enviar, detener, adjuntar, eliminar y el inspector.
  - Cada mensaje se lee como un único elemento: "You: …" o "Basalt: …".
  - Los indicadores de progreso indican qué están esperando.
  - La insignia de compatibilidad dice "fits on this Mac" o "does not fit".
  - Los adjuntos pendientes se leen como "Remove <file>".
- **Idiomas:** la aplicación sigue el idioma del sistema. Actualmente se distribuyen inglés y ruso. Una traducción es un archivo `Localization/<language>.lproj/Localizable.strings`, que `scripts/bundle_app.swift` copia en el paquete y lista en `CFBundleLocalizations`. Una prueba falla cuando una vista añade una cadena que falta en la tabla rusa.
- **Actualizaciones:** una vez al día la aplicación consulta la API de GitHub para obtener la última versión de basalt. Una versión más reciente, que no sea borrador ni prelanzamiento, se muestra como "Update available" en la barra de menús, con un enlace a la página de lanzamiento. La petición no envía a GitHub más que la dirección IP y el user agent de la aplicación. Se puede desactivar en Ajustes → General.

## Distribución: firma, notarización y un cask de Homebrew

- **En local:** `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` firma con el entorno de ejecución reforzado y genera `Basalt-<version>.zip`. Envía el zip con `notarytool`, adhiere (staple) el ticket a la aplicación, vuelve a comprimirla en zip y después genera un `.dmg`.
  - `--zip` por sí solo empaqueta sin notarizar.
  - `--notarize` rechaza una firma ad hoc o ausente.
  - Crea el perfil del llavero una sola vez con `xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>`.
- **En un lanzamiento (release):** cuando el repositorio dispone de los secretos `MAC_SIGNING_IDENTITY` (el nombre del certificado en el llavero del runner) y `MAC_NOTARY_PROFILE` (ese perfil del llavero), una etiqueta `v*` compila, firma y notariza la aplicación.
  - Adjunta `Basalt-<version>.zip` al lanzamiento de GitHub.
  - Genera `Casks/basalt-app.rb` a partir de `Casks/basalt-app.rb.template` en el mismo PR que actualiza la fórmula.
  - Sin los secretos, el lanzamiento publica únicamente la fórmula, con un aviso.
- **Instalación:** una vez que un lanzamiento incluya la aplicación, ejecuta `brew install --cask goodmai/basalt/basalt-app`.

## Compose Video Studio, Convertidor Universal y Hub en vivo (`v0.12`)

- **Compose Video Studio:** generación de vídeo por fases en 24 GB UMA y reproducción integrada (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`) con exportación a `.mp4` (`#611`).
- **Convertidor Universal (`ConvertModelSheet`):** interfaz completa de cuantización cualquiera-a-cualquiera (`MACaQ` denso/MoE, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`) en inglés y ruso (`#546`, `#599`, `#623`).
- **Hub en vivo y telemetría SSE a 10 Hz:** impulsado por `swift-huggingface` con descargas Xet, búsqueda/publicación en vivo en el Hub y telemetría en flujo continuo a 10 Hz en Monitor (`#587`).

## Presupuestos de rendimiento

Medido en un M4 Pro con 24 GB en el commit 530aa28, con Gemma 4 12B en el motor (#434):

| Presupuesto | Medido | Límite | Cómo |
|---|---|---|---|
| Ventana en un inicio en caliente | 257 ms mediana de 5 (en frío: 1.6 s) | 400 ms | `swift scripts/app_budgets.swift` |
| Despertares en inactividad una vez iniciado el motor | 0 en 10 s | 0 | mismo |
| Proceso de la aplicación | 40 MB, más 33.5 MB para una transcripción abierta de 2,000 mensajes | 150 MB | mismo, y `AppBudgetTests` |
| Un token transmitido en una transcripción de 2,000 mensajes | 7.4 ms p95, 6.0 ms mediana | 16.7 ms (60 fps) | `BASALT_APP_BUDGETS=1 swift test --filter AppBudgetTests` |
| Consulta rápida: panel y primer token sobre el tiempo propio del motor | 40 ms en caliente (panel 15 ms); la primera vista SwiftUI en un proceso 175 ms | 150 ms | mismo |

`app_budgets.swift` toma el bloqueo de la GPU, ya que el motor carga su modelo. Ejecuta ambos en un Mac sin carga, sin trabajos de CI en curso. Requieren una verificación manual no automatizada: el panel sobre una aplicación a pantalla completa (es un panel sin activación con `.fullScreenAuxiliary`), App Intents desde Atajos con la aplicación cerrada y una instalación mediante cask en un Mac limpio, lo cual necesita los secretos de firma.
