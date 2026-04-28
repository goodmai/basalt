import ArgumentParser

// MARK: — Root entry point

// No @main here — entry point is Sources/GemmaServerBin/main.swift

// Добавлен атрибут @available для корректной работы swift-argument-parser
@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
public struct GemmaServerCLI: AsyncParsableCommand {
    public init() {}
    public static let configuration = CommandConfiguration(
        commandName: "GemmaServer",
        abstract: "Local Gemma 4 inference — dual-interface server (MCP + REST/A2A)",
        discussion: """
        QUICK START
          GemmaServer models list                                  List Gemma 4 models
          GemmaServer models list --search Qwen3                   List Qwen3 models
          GemmaServer models list --search Qwen2.5-Coder           List Qwen Coder models
          GemmaServer models download                              Interactive picker
          GemmaServer models download mlx-community/…             Download specific model
          GemmaServer serve --model mlx-community/…               Start inference server
          GemmaServer chat  --model mlx-community/…               Interactive chat
          GemmaServer agents analyze agents.md                    Analyze agent capabilities

        RECOMMENDED MODELS (by RAM)
          mlx-community/gemma-4-e2b-it-4bit               Gemma 4 2B   ~2.7 GB
          mlx-community/Qwen3.5-4B-4bit                   Qwen3.5 4B   ~3.0 GB
          mlx-community/Qwen3.5-9B-OptiQ-4bit             Qwen3.5 9B   ~6.0 GB  ★ popular
          mlx-community/Qwen2.5-Coder-7B-Instruct-4bit    Qwen Coder   ~4.5 GB
          mlx-community/Qwen3.6-27B-4bit                  Qwen3.6 27B  ~16 GB   ★ newest
          mlx-community/Qwen3.6-35B-A3B-4bit              Qwen3.6 MoE  ~21 GB   ★ efficient

        INTERFACES
          MCP  (stdio)   — Cursor, Claude Desktop, any MCP-compatible IDE
          REST (HTTP)    — Agent-to-Agent (A2A) calls, curl, microservices

        CACHE
          ~/.cache/huggingface/hub/                        Standard HF cache location
        """,
        version: HealthResponse.version,
        subcommands: [
            ServeCommand.self,
            ModelsCommand.self,
            ChatCommand.self,
            AgentsCommand.self,
        ],
        defaultSubcommand: ServeCommand.self
    )
}
