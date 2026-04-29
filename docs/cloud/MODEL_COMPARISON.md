# Cloud Model Comparison

Gem integrates with OpenRouter to provide access to hundreds of cloud models. The `Gem cloud models` command allows you to view currently available models.

## Local vs Cloud Models

### Local Inference
- **Privacy:** 100% data privacy. Code and prompts stay on your device.
- **Cost:** Free. No API limits or subscriptions.
- **Latency:** Dependent on your hardware. Fast for small/medium models (e.g., Qwen 2.5 7B, Gemma 2B).
- **Use Cases:** Daily tasks, simple coding queries, offline work.

### Cloud Inference
- **Performance:** Access to state-of-the-art reasoning (GPT-4, Claude 3.5, Gemini 1.5).
- **Scale:** Better suited for massive codebases, complex architectures, or large context windows.
- **Cost:** Pay per token. Configurable daily/monthly limits.
- **Use Cases:** Deep analytical tasks, complex bug fixing, long-form content generation.

## Popular Cloud Models on OpenRouter
- `anthropic/claude-3.5-sonnet` (Excellent at coding)
- `openai/gpt-4o` (Great all-rounder)
- `google/gemini-1.5-pro` (Huge context window)
- `meta-llama/llama-3-70b-instruct` (Top open-source)
