import ArgumentParser

// MARK: — Root entry point

// No @main here — entry point is Sources/GemBin/main.swift

// Добавлен атрибут @available для корректной работы swift-argument-parser
@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
public struct GemCLI: AsyncParsableCommand {
    public init() {}
    public static let configuration = CommandConfiguration(
        commandName: "gemm",
        abstract: "Local Gemma 4 inference — dual-interface server (MCP + REST/A2A)",
        discussion: """
        QUICK START
          gemm onboard                                      🆕 First-time setup wizard
          gemm fit                                          🔍 Hardware analysis & model recommendations
          gemm models list                                  List popular MLX models
          gemm models list --search Qwen3                   List Qwen3 models
          gemm models list --search Qwen2.5-Coder           List Qwen Coder models
          gemm models download                              Interactive picker
          gemm models download mlx-community/…             Download specific model
          gemm serve --model mlx-community/…               Start inference server
          gemm chat  --model mlx-community/…               Interactive chat
        RECOMMENDED MODELS (by RAM)
          mlx-community/gemma-4-e2b-it-4bit               Gemma 4 2B   ~2.7 GB
          mlx-community/Qwen3.5-4B-4bit                   Qwen3.5 4B   ~3.0 GB
          mlx-community/Qwen3.5-9B-OptiQ-4bit             Qwen3.5 9B   ~6.0 GB  ★ popular
          mlx-community/Qwen2.5-Coder-7B-Instruct-4bit    Qwen Coder   ~4.5 GB
          AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit Huihui 27B ~15.2 GB ★ abliterated
          Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX          MYTHOS 26B   ~14.5 GB ★ MoE DQ
          mlx-community/Qwen3.6-27B-4bit                  Qwen3.6 27B  ~16 GB   ★ newest
          mlx-community/Qwen3.6-35B-A3B-4bit              Qwen3.6 MoE  ~21 GB   ★ efficient
          Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4          Qwen3.6 35B  ~20.5 GB ★ NVFP4 MoE

        DENSE & ABLITERATED — one invocation, plus the flag each family needs
          gemm serve --rest --model <id>   …and whatever the right column says
            mlx-community/gemma-4-e4b-it-4bit        Gemma 4 4B, nothing extra
            mlx-community/gemma-4-12b-it-4bit        Gemma 4 12B, nothing extra
            divinetribe/gemma-4-12B-it-abliterated-4bit-mlx-text  abliterated
            AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit    abliterated
            PocketAiHub/Qwen3.8-27B-Abliterated-MLX  --quant 4bit (2/4/6bit)
            ornith-ai/Ornith-1.5-9B-MLX-4bit         --reasoning-effort none
            ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit    --reasoning-effort none
            mlx-community/Ornith-1.5-9B-OptiQ-4bit   mixed 8/4-bit, same flag
          Ornith and Qwen3.x are reasoning models: without --reasoning-effort none
          they can spend the whole token budget inside <think> and never answer.
          Tight on memory? --kv-bits 4 buys context, --dry-run checks it first.
        
        POINTING TOOLS AT IT  (127.0.0.1, not localhost — the server is IPv4 only)
          OpenCode:
            export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
            export OPENAI_API_KEY=local OPENAI_MODEL=gemm
            opencode
          Claude Code:
            export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
            export ANTHROPIC_AUTH_TOKEN=local
            claude
          Anything OpenAI-compatible:
            base URL http://127.0.0.1:8080/v1, any key, model "gemm"

        INTERFACES
          MCP  (stdio)   — Cursor, Claude Desktop, any MCP-compatible IDE
          REST (HTTP)    — Agent-to-Agent (A2A) calls, curl, microservices

        CACHE
          ~/.cache/huggingface/hub/                        Standard HF cache location
        """,
        version: HealthResponse.version,
        subcommands: [
            OnboardCommand.self,
            FitCommand.self,
            ServeCommand.self,
            ModelsCommand.self,
            ChatCommand.self,
            CloudCommand.self,
        ],
        defaultSubcommand: ChatCommand.self
    )
}
