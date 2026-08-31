import Foundation
import ArgumentParser

/// Server configuration — parsed from CLI arguments.
public struct ServerConfig: Sendable {
    public let modelPath:   String
    public let modelId:     String?   // HF repo ID for display, e.g. "mlx-community/Qwen3.5-4B-4bit"
    public let restPort:    Int
    public let host:        String
    public let maxTokens:   Int
    public let logLevel:    LogLevel

    public enum LogLevel: String, Sendable, ExpressibleByArgument {
        case debug, info, warn, error
    }

    public init(
        modelPath: String,
        modelId:   String?   = nil,
        restPort:  Int       = 8080,
        host:      String    = "127.0.0.1",
        maxTokens: Int       = 65536,
        logLevel:  LogLevel  = .info
    ) {
        self.modelPath = modelPath
        self.modelId   = modelId
        self.restPort  = restPort
        self.host      = host
        self.maxTokens = maxTokens
        self.logLevel  = logLevel
    }
}

extension ServerConfig {
    /// Default path used when no --model flag is provided.
    public static let development = ServerConfig(
        modelPath: "./models/gemma-4-e2b-it-4bit",
        restPort:  8080,
        host:      "127.0.0.1",
        maxTokens: 65536,
        logLevel:  .info
    )
}
