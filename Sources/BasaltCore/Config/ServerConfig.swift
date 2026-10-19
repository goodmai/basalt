import Foundation
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
    /// Server-wide request budget; nil means no rate limit.
    public let rateLimitPerMinute: Int?
    public let allowedOrigins: [String]

    /// Applied off loopback when no `--rate-limit` is given.
    public static let defaultRateLimitPerMinute = 120

    public enum LogLevel: String, Sendable, CaseIterable {
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
        autoGenerateKeyIfMissing: Bool = false,
        rateLimitPerMinute: Int? = nil,
        allowedOrigins: [String] = []
    ) throws {
        self.modelPath = modelPath
        self.modelId   = modelId
        self.restPort  = restPort
        self.host      = host
        self.maxTokens = maxTokens
        self.logLevel  = logLevel
        self.allowedOrigins = allowedOrigins

        self.apiKey = try Self.resolveApiKey(
            host: host, apiKey: apiKey, autoGenerateKeyIfMissing: autoGenerateKeyIfMissing
        ).key
        self.rateLimitPerMinute = Self.resolveRateLimit(host: host, requested: rateLimitPerMinute)
    }

    /// Off loopback a rate limit is on by default; an explicit 0 turns it off anywhere.
    public static func resolveRateLimit(host: String, requested: Int?) -> Int? {
        let limit = requested ?? (isLoopback(host: host) ? nil : defaultRateLimitPerMinute)
        return limit.flatMap { $0 > 0 ? $0 : nil }
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
            try fileManager.createDirectory(at: dir, withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
        }

        let existingData = fileManager.fileExists(atPath: url.path) ? try Data(contentsOf: url) : nil
        // Merge first: a malformed config must not replace the last good backup.
        let mergedData = try mergeConfiguration(existingData: existingData, systemUpdates: systemUpdates, defaults: defaults)
        if createBackup, let existingData {
            try writeOwnerOnly(existingData, to: url.appendingPathExtension("bak"))
        }
        try writeOwnerOnly(mergedData, to: url)
        return url
    }

    /// Atomic write whose file is 0600 from the moment it exists (no world-readable window):
    /// the temp file is opened O_EXCL with mode 0600, then swapped in by rename(2).
    /// (FileManager.createFile and replaceItemAt both chmod after the data is on disk.)
    static func writeOwnerOnly(_ data: Data, to url: URL) throws {
        let tmp = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        func fail(_ code: Int32) -> POSIXError {
            unlink(tmp.path)
            return POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        let fd = open(tmp.path, O_CREAT | O_EXCL | O_WRONLY, 0o600)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let written = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        guard written == data.count, fsync(fd) == 0 else {
            let code = errno
            close(fd)
            throw fail(code)
        }
        guard close(fd) == 0 else { throw fail(errno) }
        guard rename(tmp.path, url.path) == 0 else { throw fail(errno) }
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
