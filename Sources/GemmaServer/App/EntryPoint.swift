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
          GemmaServer models list                           Browse available models
          GemmaServer models download                       Interactive picker + download
          GemmaServer models download mlx-community/…      Download specific model
          GemmaServer serve --model mlx-community/…        Start inference server

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
        ],
        defaultSubcommand: ServeCommand.self
    )
}
