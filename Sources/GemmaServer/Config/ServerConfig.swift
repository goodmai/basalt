import Foundation
import ArgumentParser

/// Конфигурация сервера — парсится из CLI аргументов.
public struct ServerConfig: Sendable {
    public let modelPath: String
    public let restPort: Int
    public let host: String
    public let maxConcurrentRequests: Int
    public let maxTokens: Int
    public let logLevel: LogLevel

    public enum LogLevel: String, Sendable, ExpressibleByArgument {
        case debug, info, warn, error
    }

    public init(
        modelPath: String,
        restPort: Int = 8080,
        host: String = "127.0.0.1",
        maxConcurrentRequests: Int = 4,
        maxTokens: Int = 65536,
        logLevel: LogLevel = .info
    ) {
        self.modelPath = modelPath
        self.restPort = restPort
        self.host = host
        self.maxConcurrentRequests = maxConcurrentRequests
        self.maxTokens = maxTokens
        self.logLevel = logLevel
    }
}

extension ServerConfig {
    /// Конфигурация по умолчанию для разработки.
    public static let development = ServerConfig(
        modelPath: "./models/gemma-3-1b-instruct",
        restPort: 8080,
        host: "127.0.0.1",
        maxTokens: 65536,
        logLevel: .debug
    )
}
