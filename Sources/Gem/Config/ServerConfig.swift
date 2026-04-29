import Foundation
import ArgumentParser

/// Конфигурация сервера — парсится из CLI аргументов.
public struct ServerConfig: Sendable {
    public let modelPath: String
    public let modelId: String?   // original repo ID for display (e.g. "mlx-community/Qwen3-6B-4bit")
    public let restPort: Int
    public let host: String
    public let maxConcurrentRequests: Int
    public let maxTokens: Int
    public let jwtSecret: String?  // SECURITY: Optional, must be provided via env or CLI
    public let dbPath: String
    public let logLevel: LogLevel

    public enum LogLevel: String, Sendable, ExpressibleByArgument {
        case debug, info, warn, error
    }

    public init(
        modelPath: String,
        modelId: String? = nil,
        restPort: Int = 8080,
        host: String = "127.0.0.1",
        maxConcurrentRequests: Int = 4,
        maxTokens: Int = 65536,
        jwtSecret: String? = nil,  // SECURITY: No default value
        dbPath: String = "auth.sqlite3",
        logLevel: LogLevel = .info
    ) {
        self.modelPath = modelPath
        self.modelId = modelId
        self.restPort = restPort
        self.host = host
        self.maxConcurrentRequests = maxConcurrentRequests
        self.maxTokens = maxTokens
        self.jwtSecret = jwtSecret
        self.dbPath = dbPath
        self.logLevel = logLevel
    }
}

extension ServerConfig {
    /// Конфигурация по умолчанию для разработки.
    /// SECURITY WARNING: Only use for local development, never in production!
    public static let development = ServerConfig(
        modelPath: "./models/gemma-3-1b-instruct",
        restPort: 8080,
        host: "127.0.0.1",
        maxTokens: 65536,
        jwtSecret: "dev-secret-DO-NOT-USE-IN-PRODUCTION-dev-secret-DO-NOT-USE-IN-PRODUCTION",  // 64+ chars
        dbPath: "auth.sqlite3",
        logLevel: .debug
    )
}
