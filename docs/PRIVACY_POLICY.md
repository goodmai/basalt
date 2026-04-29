# Gem Privacy Policy

## 1. Introduction
Gem is designed from the ground up with data privacy and security as the core principle. We believe that your data belongs to you and should remain on your hardware.

## 2. 100% Local Inference
All AI model inference, prompt processing, and data handling occur **100% locally** on your Apple Silicon device. 

- Your prompts are **never** sent to any external servers.
- Your responses are generated locally.
- Your context window contents exist entirely within your device's memory.

## 3. Network Calls
Gem only makes network calls in the following explicit scenarios:
1. **Model Downloads:** When a model is requested that is not present in the local cache, Gem downloads the model weights from the Hugging Face Hub (`huggingface.co`). This is a direct download.
2. **Local APIs:** Exposing the REST and MCP endpoints strictly on your local interfaces (`127.0.0.1` or configured local network interfaces).

## 4. Telemetry & Analytics
Gem **does not** collect telemetry, analytics, crash reports, or usage statistics. We have zero visibility into how you use the application, what models you run, or what data you process.

## 5. Third-Party Access
No third party has access to your data through Gem. We do not use third-party tracking libraries or APIs that exfiltrate data.
