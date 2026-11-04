import Foundation
import ArgumentParser
import BasaltCore
import BasaltServer

struct ServeNetworkOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "REST API port (default: 8080)")
    var port: Int?

    @Option(name: .long, help: "Bind address (default: 127.0.0.1)")
    var host: String?

    @Option(
        name: .customLong("api-key"),
        help: "API key required for authenticating requests, mandatory when binding to a non-loopback address. Prefer the BASALT_API_KEY environment variable: argv is visible to every local user via ps."
    )
    var apiKey: String?

    @Flag(
        name: .customLong("auto-generate-key"),
        help: "Automatically generate a cryptographically strong API key if none is provided on non-loopback bind (a new key on every start — pass a fixed key for clients that reconnect)"
    )
    var autoGenerateKey: Bool = false

    @Option(
        name: .customLong("rate-limit"),
        help: "Server-wide request limit per minute, 429 beyond it (default: off on loopback, \(ServerConfig.defaultRateLimitPerMinute) otherwise; 0 disables)"
    )
    var rateLimit: Int?

    @Option(
        name: .customLong("rate-burst"),
        help: "Maximum burst of requests allowed before 429 (default: min(rate-limit, 10); 0 disables burst capping)"
    )
    var rateBurst: Int?

    @Option(
        name: .customLong("allow-origin"),
        help: "Browser origin allowed to call the API besides loopback ones, e.g. https://app.example:3000 (repeatable; also allowed_origins in ~/.basalt/config.json)"
    )
    var allowOrigins: [String] = []

    @Option(
        name: .customLong("config"),
        help: "Path to JSON server configuration file (default: ~/.basalt/config.json when it exists)"
    )
    var configFile: String?

    var configPath: String? { configFile ?? ServeCommand.defaultConfigFileIfPresent() }

    func validateRates() throws {
        _ = try NetworkSettings.nonNegativeRate(rateLimit, key: "--rate-limit", unit: " of requests per minute")
        _ = try NetworkSettings.nonNegativeRate(rateBurst, key: "--rate-burst")
    }
}

struct NetworkSettings: Sendable, Equatable {
    var port: Int
    var host: String
    var apiKey: String?
    var rateLimit: Int?
    var rateBurst: Int?
    var allowedOrigins: [String]
    var virtualKeys: [VirtualKey]

    struct Flags: Sendable, Equatable {
        var port: Int?
        var host: String?
        var apiKey: String?
        var rateLimit: Int?
        var rateBurst: Int?
        var allowedOrigins: [String] = []
    }

    static func resolve(_ flags: Flags, file: ConfigSection) throws -> NetworkSettings {
        let rateLimit = try nonNegativeRate(
            flags.rateLimit ?? file.value("rate_limit", as: Int.self), key: "rate_limit", unit: " of requests per minute")
        let rateBurst = try nonNegativeRate(flags.rateBurst ?? file.value("rate_burst", as: Int.self), key: "rate_burst")
        let allowedOrigins = try Self.allowedOrigins(flags.allowedOrigins, in: file)
        let virtualKeys = try Self.virtualKeys(in: file)
        return NetworkSettings(
            port: try flags.port ?? file.value("port", as: Int.self) ?? 8080,
            host: try flags.host ?? file.value("host", as: String.self) ?? "127.0.0.1",
            apiKey: try nonBlank(flags.apiKey) ?? file.value("api_key", as: String.self),
            rateLimit: rateLimit,
            rateBurst: rateBurst,
            allowedOrigins: allowedOrigins,
            virtualKeys: virtualKeys
        )
    }

    static func nonBlank(_ key: String?) -> String? {
        key.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
    }

    static func nonNegativeRate(_ value: Int?, key: String, unit: String = "") throws -> Int? {
        if let value, value < 0 {
            throw ValidationError("\(key) must be 0 (off) or a positive number\(unit), got \(value)")
        }
        return value
    }

    static func allowedOrigins(_ cli: [String], in file: ConfigSection) throws -> [String] {
        let origins = try cli + (file.value("allowed_origins", as: [String].self) ?? [])
        for origin in origins where RequestGuardMiddleware.parseOrigin(origin) == nil {
            throw ValidationError("allowed origin must be scheme://host[:port]: http or https, no path, no trailing slash, no wildcard; got \(origin)")
        }
        return origins
    }

    static func virtualKeys(in file: ConfigSection) throws -> [VirtualKey] {
        let keys = try (file.value("virtual_keys", as: [[String: Any]].self) ?? []).map(virtualKey)
        if Set(keys.map(\.id)).count != keys.count || Set(keys.map(\.key)).count != keys.count {
            throw ValidationError("virtual_keys: ids and keys must be unique")
        }
        return keys
    }

    static func virtualKey(_ entry: [String: Any]) throws -> VirtualKey {
        guard let key = entry["key"] as? String, key.count >= 16, let id = entry["id"] as? String, !id.isEmpty else {
            throw ValidationError("virtual_keys: every entry needs \"id\" and a \"key\" of at least 16 characters")
        }
        let budget = entry["max_budget"].flatMap { $0 is NSNull ? nil : $0 }
        if let budget, (budget as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() || $0.doubleValue < 0 }) ?? true {
            throw ValidationError("virtual_keys[\(id)].max_budget must be a non-negative number of USD, got \(budget)")
        }
        return VirtualKey(key: key, id: id, maxBudget: (budget as? NSNumber)?.doubleValue)
    }
}

struct ConfigSection {
    var values: [String: Any] = [:]
    var keyPrefix = ""

    static func fileURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    static func serverConfig(at path: String?) throws -> ConfigSection {
        guard let path else { return ConfigSection() }
        let url = fileURL(path)
        do {
            return ConfigSection(values: try ServerConfig.load(from: url))
        } catch {
            throw ValidationError("Cannot read server config \(url.path): \(error.localizedDescription)")
        }
    }

    static func cloudFallback(at path: String?) throws -> ConfigSection {
        var section = ConfigSection(keyPrefix: "cloud_fallback.")
        guard let path, let raw = try ServerConfig.load(from: fileURL(path))["cloud_fallback"] else { return section }
        guard let values = raw as? [String: Any] else {
            throw ValidationError("--config: \"cloud_fallback\" must be an object")
        }
        section.values = values
        return section
    }

    func value<T>(_ key: String, as _: T.Type) throws -> T? {
        guard let raw = values[key], !(raw is NSNull) else { return nil }
        guard CFGetTypeID(raw as CFTypeRef) != CFBooleanGetTypeID() || T.self == Bool.self, let typed = raw as? T else {
            throw ValidationError("--config: \"\(keyPrefix)\(key)\" must be \(T.self), got \(raw)")
        }
        return typed
    }
}

extension ServeCommand {
    var port: Int? { network.port }
    var host: String? { network.host }
    var apiKey: String? { network.apiKey }
    var allowOrigins: [String] { network.allowOrigins }

    static func defaultConfigFileIfPresent() -> String? {
        let url = ServerConfig.defaultConfigFileURL
        return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
    }

    func resolveAccess() throws -> NetworkSettings {
        var settings = try Self.resolveNetwork(
            cliPort: network.port, cliHost: network.host,
            cliApiKey: network.apiKey ?? ProcessInfo.processInfo.environment["BASALT_API_KEY"],
            cliRateLimit: network.rateLimit,
            cliRateBurst: network.rateBurst,
            cliAllowedOrigins: network.allowOrigins,
            configFile: network.configPath
        )
        let access: (key: String?, generated: Bool)
        do {
            access = try ServerConfig.resolveApiKey(
                host: settings.host, apiKey: settings.apiKey, autoGenerateKeyIfMissing: network.autoGenerateKey
            )
        } catch ServerConfigError.missingApiKeyForNonLoopback(let badHost) {
            log("\(red("Security error:")) Refusing to start on non-loopback interface '\(badHost)' without an API key.")
            log("Pass \(bold("--api-key <secret>")) or \(bold("--auto-generate-key")) to enable access.")
            throw ValidationError("Refusing to start on non-loopback interface '\(badHost)' without an API key. Pass --api-key <secret> or --auto-generate-key to protect external access.")
        }
        if access.generated, let generated = access.key {
            log("⚠️  Binding to non-loopback address '\(settings.host)' without explicit key.")
            log("🔑 Auto-generated API key: \(bold(generated))")
        }
        settings.apiKey = access.key
        return settings
    }
}
