# Gem Privacy Policy

## 1. Introduction
Gem is designed from the ground up with data privacy and security as the core principle. We believe that your data belongs to you and should remain on your hardware.

## 2. 100% Local Inference
All AI model inference, prompt processing, and data handling occur **100% locally** on your Apple Silicon device. 

- Your prompts are **never** sent to any external servers, unless you turn on the cloud fallback (`basalt serve --cloud-fallback`, or `cloud_fallback` in `~/.gem/config.json`): then a request the local model cannot serve goes to the OpenAI-compatible endpoint you configured. It is off by default.
- Your responses are generated locally.
- Your context window contents exist entirely within your device's memory.

## 3. Network Calls
Gem only makes network calls in the following explicit scenarios:
1. **Model Downloads:** When a model is requested that is not present in the local cache, Gem downloads the model weights from the Hugging Face Hub (`huggingface.co`). This is a direct download.
2. **Local APIs:** Exposing the REST and MCP endpoints strictly on your local interfaces (`127.0.0.1` or configured local network interfaces).

## 4. Telemetry, Analytics & the Local Session Log
Gem **does not** collect telemetry, analytics, crash reports, or usage statistics, and nothing it records leaves your machine. We have zero visibility into how you use the application, what models you run, or what data you process.

`basalt serve` keeps a **local session log** so you can debug a client or compare models on real traffic: one JSONL file per session under `~/.basalt/sessions/<date>/`, the directory readable only by you (`0700`, files `0600`).
- **`full` (default)** stores the conversation sent to the model and its reply, besides the metrics. Your prompts are therefore on disk in this mode.
- **`meta`** stores only metrics, sizes and SHA-256 hashes: no prompt, reply, reasoning or tool arguments.
- **`off`** writes nothing.

Choose with `--session-log full|meta|off`; the server prints the mode and directory at start. HTTP headers (`Authorization`, `x-api-key`) are never recorded, and a request that fails to decode is logged only by its size. Files older than 14 days, and the oldest beyond 1 GB in total, are deleted at start (`--session-log-retention-days`, `--session-log-max-total-mb`).

## 5. Third-Party Access
No third party has access to your data through Gem. We do not use third-party tracking libraries or APIs that exfiltrate data.
