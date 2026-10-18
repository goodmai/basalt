import Foundation
import ArgumentParser
import Security

/// Errors encountered while validating or loading server configuration.
public enum ServerConfigError: Error, LocalizedError, Equatable {
    case missingApiKeyForNonLoopback(host: String)
    case invalidConfigFormat(String)
    case randomGenerationFailed(status: Int32)

    public var errorDescription: String? {
        switch self {
        case .missingApiKeyForNonLoopback(let host):
            return "Server is configured to bind to non-loopback address '\(host)', but no API key is provided. When binding externally, an API key is required to prevent open access."
        case .invalidConfigFormat(let details):
            return "Invalid configuration format: \(details)"
        case .randomGenerationFailed(let status):
            return "Secure random generator failed (OSStatus \(status)); refusing to issue a predictable API key."
        }
    }
}

/// Server configuration — parsed from CLI arguments and configuration files.
public struct ServerConfig: Sendable {
    public let modelPath:   String
    public let modelId:     String?   // HF repo ID for display, e.g. "mlx-community/Qwen3.5-4B-4bit"
    public let restPort:    Int
    public let host:        String
    public let maxTokens:   Int
    public let logLevel:    LogLevel
    public let apiKey:      String?

    public enum LogLevel: String, Sendable, ExpressibleByArgument {
        case debug, info, warn, error
    }

    /// System configuration keys managed by CLI/server options.
    public static let systemKeys: Set<String> = [
        "exe", "args", "port", "host", "api_key", "model", "model_path", "max_tokens", "log_level"
    ]

    /// Default path for the persistent JSON server configuration.
    public static var defaultConfigFileURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".gem")
            .appendingPathComponent("config.json")
    }

    /// Determines whether the given host is a local loopback interface.
    public static func isLoopback(host: String) -> Bool {
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "127.0.0.1" || normalized == "localhost" || normalized == "::1" || normalized == "[::1]"
    }

    /// Returns true if the configured host is a local loopback interface.
    public var isLoopback: Bool {
        Self.isLoopback(host: host)
    }

    /// Generates a cryptographically strong pseudo-random token suitable for an API key.
    public static func generateCryptographicallyStrongToken(prefix: String = "gem_") throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        guard status == errSecSuccess else {
            throw ServerConfigError.randomGenerationFailed(status: status)
        }
        let hex = bytes.map { String(format: "%02x", $0) }.joined()
        return prefix + hex
    }

    public init(
        modelPath: String,
        modelId:   String?   = nil,
        restPort:  Int       = 8080,
        host:      String    = "127.0.0.1",
        maxTokens: Int       = 65536,
        logLevel:  LogLevel  = .info,
        apiKey:    String?   = nil,
        autoGenerateKeyIfMissing: Bool = false
    ) throws {
        self.modelPath = modelPath
        self.modelId   = modelId
        self.restPort  = restPort
        self.host      = host
        self.maxTokens = maxTokens
        self.logLevel  = logLevel

        self.apiKey = try Self.resolveApiKey(
            host: host, apiKey: apiKey, autoGenerateKeyIfMissing: autoGenerateKeyIfMissing
        ).key
    }

    /// The one place the exposure rule lives: a non-loopback bind needs a non-empty key,
    /// generated only on explicit opt-in. Callers can run it before expensive work.
    public static func resolveApiKey(
        host: String,
        apiKey: String?,
        autoGenerateKeyIfMissing: Bool
    ) throws -> (key: String?, generated: Bool) {
        if let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty {
            return (key, false)
        }
        if isLoopback(host: host) { return (nil, false) }
        guard autoGenerateKeyIfMissing else {
            throw ServerConfigError.missingApiKeyForNonLoopback(host: host)
        }
        return (try generateCryptographicallyStrongToken(), true)
    }

    // MARK: — JSON Configuration Management

    /// Non-destructively merges system updates into existing JSON data.
    /// Only `systemKeys` are written: `systemUpdates` overwrite, `defaults` fill a key only when it
    /// is absent (so a user-edited host/port survives a re-run). Every other key is left as is.
    public static func mergeConfiguration(
        existingData: Data?,
        systemUpdates: [String: Any],
        defaults: [String: Any] = [:]
    ) throws -> Data {
        var merged: [String: Any] = [:]

        if let existingData = existingData, !existingData.isEmpty {
            let jsonObject = try JSONSerialization.jsonObject(with: existingData, options: [])
            guard let dict = jsonObject as? [String: Any] else {
                throw ServerConfigError.invalidConfigFormat("Root JSON object must be a dictionary")
            }
            merged = dict
        }

        // Apply only system keys from systemUpdates
        for (key, value) in systemUpdates where systemKeys.contains(key) {
            merged[key] = value
        }
        for (key, value) in defaults where systemKeys.contains(key) && merged[key] == nil {
            merged[key] = value
        }

        return try JSONSerialization.data(withJSONObject: merged, options: [.prettyPrinted, .sortedKeys])
    }

    /// Merges system updates into the configuration file at `url`.
    /// If the file already exists, creates a `.bak` backup copy before writing.
    /// Both files are owner-only (0600): the config may carry `api_key`.
    @discardableResult
    public static func mergeAndSave(
        to url: URL,
        systemUpdates: [String: Any],
        defaults: [String: Any] = [:],
        createBackup: Bool = true
    ) throws -> URL {
        let fileManager = FileManager.default
        let dir = url.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: dir.path) {
            try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        let existingData: Data?
        if fileManager.fileExists(atPath: url.path) {
            let data = try Data(contentsOf: url)
            existingData = data
            if createBackup {
                let backupURL = url.appendingPathExtension("bak")
                try? fileManager.removeItem(at: backupURL)
                try data.write(to: backupURL, options: .atomic)
                try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backupURL.path)
            }
        } else {
            existingData = nil
        }

        let mergedData = try mergeConfiguration(existingData: existingData, systemUpdates: systemUpdates, defaults: defaults)
        try mergedData.write(to: url, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return url
    }

    /// Loads the configuration dictionary from a given file URL.
    public static func load(from url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ServerConfigError.invalidConfigFormat("Root JSON object must be a dictionary")
        }
        return dict
    }
}

extension ServerConfig {
    /// Default path used when no --model flag is provided.
    public static let development = try! ServerConfig(
        modelPath: "./models/gemma-4-e2b-it-4bit",
        restPort:  8080,
        host:      "127.0.0.1",
        maxTokens: 65536,
        logLevel:  .info
    )
}
