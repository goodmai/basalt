import Testing
import Foundation
import Hummingbird
import HTTPTypes
import Logging
import NIOCore
import HummingbirdTesting
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("ServerConfig security and logic")
struct ConfigTests {

    // MARK: — Defaults & Round-trip

    @Test("ServerConfig default values on loopback")
    func testDefaults() throws {
        let config = try ServerConfig(modelPath: "path")
        #expect(config.restPort == 8080)
        #expect(config.host == "127.0.0.1")
        #expect(config.maxTokens == 65536)
        #expect(config.logLevel == .info)
        #expect(config.apiKey == nil)
        #expect(config.isLoopback == true)
    }

    @Test("ServerConfig custom values round-trip with explicit apiKey")
    func testCustomWithApiKey() throws {
        let config = try ServerConfig(
            modelPath: "/models/qwen",
            modelId:   "mlx-community/Qwen3.5-4B-4bit",
            restPort:  9090,
            host:      "0.0.0.0",
            maxTokens: 8192,
            logLevel:  .debug,
            apiKey:    "test-secret-token"
        )
        #expect(config.modelId == "mlx-community/Qwen3.5-4B-4bit")
        #expect(config.restPort == 9090)
        #expect(config.host == "0.0.0.0")
        #expect(config.maxTokens == 8192)
        #expect(config.logLevel == .debug)
        #expect(config.apiKey == "test-secret-token")
        #expect(config.isLoopback == false)
    }

    // MARK: — Loopback detection

    @Test("Loopback addresses are recognized correctly")
    func testLoopbackDetection() {
        #expect(ServerConfig.isLoopback(host: "127.0.0.1"))
        #expect(ServerConfig.isLoopback(host: "localhost"))
        #expect(ServerConfig.isLoopback(host: "LOCALHOST"))
        #expect(ServerConfig.isLoopback(host: "  127.0.0.1  "))
        #expect(ServerConfig.isLoopback(host: "::1"))
        #expect(ServerConfig.isLoopback(host: "[::1]"))

        #expect(!ServerConfig.isLoopback(host: "0.0.0.0"))
        #expect(!ServerConfig.isLoopback(host: "192.168.1.100"))
        #expect(!ServerConfig.isLoopback(host: "10.0.0.1"))
        #expect(!ServerConfig.isLoopback(host: "gemm.local"))
    }

    // MARK: — Non-loopback security policy

    @Test("Non-loopback host without apiKey throws missingApiKeyForNonLoopback")
    func testNonLoopbackRequiresApiKey() {
        #expect(throws: ServerConfigError.missingApiKeyForNonLoopback(host: "0.0.0.0")) {
            _ = try ServerConfig(modelPath: "path", host: "0.0.0.0")
        }

        #expect(throws: ServerConfigError.missingApiKeyForNonLoopback(host: "192.168.0.2")) {
            _ = try ServerConfig(modelPath: "path", host: "192.168.0.2", apiKey: nil)
        }

        // Empty string key is rejected
        #expect(throws: ServerConfigError.missingApiKeyForNonLoopback(host: "0.0.0.0")) {
            _ = try ServerConfig(modelPath: "path", host: "0.0.0.0", apiKey: "")
        }

        // Whitespace-only key is rejected
        #expect(throws: ServerConfigError.missingApiKeyForNonLoopback(host: "0.0.0.0")) {
            _ = try ServerConfig(modelPath: "path", host: "0.0.0.0", apiKey: "   \t  ")
        }
    }

    @Test("Non-loopback host with autoGenerateKeyIfMissing generates strong token")
    func testAutoGenerateToken() throws {
        let config = try ServerConfig(
            modelPath: "path",
            host: "0.0.0.0",
            apiKey: nil,
            autoGenerateKeyIfMissing: true
        )
        #expect(config.apiKey != nil)
        guard let key = config.apiKey else { return }
        #expect(key.hasPrefix("gem_"))
        #expect(key.count >= 36) // "gem_" (4) + 64 hex chars = 68
    }

    @Test("Loopback host allows optional apiKey or nil")
    func testLoopbackApiKeyOptional() throws {
        let configWithoutKey = try ServerConfig(modelPath: "path", host: "127.0.0.1")
        #expect(configWithoutKey.apiKey == nil)

        let configWithKey = try ServerConfig(modelPath: "path", host: "localhost", apiKey: "optional-key")
        #expect(configWithKey.apiKey == "optional-key")
    }

    // MARK: — Non-destructive JSON configuration merge

    @Test("Non-destructive JSON merge preserves user sections and updates system keys")
    func testNonDestructiveMerge() throws {
        let initialJSON = """
        {
          "exe": "/old/path/gemm",
          "args": ["serve", "--model", "old-model"],
          "port": 8080,
          "host": "127.0.0.1",
          "api_key": "old-key",
          "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
          },
          "mcp_servers": {
            "custom_indexer": {
              "command": "npx",
              "args": ["-y", "custom-indexer"]
            }
          },
          "cloud": {
            "daily_budget": 15.0,
            "monthly_budget": 100.0
          },
          "cors_origins": [
            "http://localhost:3000",
            "https://myagent.dev"
          ],
          "custom_user_annotation": "keep-me"
        }
        """.data(using: .utf8)!

        let systemUpdates: [String: Any] = [
            "exe": "/Applications/gemm",
            "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--rest"],
            "port": 9090,
            "host": "0.0.0.0",
            "api_key": "gem_new_generated_key_12345",
            // Attempt to update a user section via systemUpdates should be ignored
            "sampling": ["temperature": 0.0]
        ]

        let mergedData = try ServerConfig.mergeConfiguration(
            existingData: initialJSON,
            systemUpdates: systemUpdates
        )

        let merged = try JSONSerialization.jsonObject(with: mergedData) as! [String: Any]

        // System keys must be updated
        #expect(merged["exe"] as? String == "/Applications/gemm")
        #expect(merged["port"] as? Int == 9090)
        #expect(merged["host"] as? String == "0.0.0.0")
        #expect(merged["api_key"] as? String == "gem_new_generated_key_12345")
        #expect(merged["args"] as? [String] == ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--rest"])

        // User sections MUST be preserved intact
        let sampling = merged["sampling"] as? [String: Any]
        #expect(sampling?["temperature"] as? Double == 0.7)
        #expect(sampling?["top_p"] as? Double == 0.95)
        #expect(sampling?["repetition_penalty"] as? Double == 1.1)

        let mcp = merged["mcp_servers"] as? [String: Any]
        #expect(mcp?["custom_indexer"] != nil)

        let cloud = merged["cloud"] as? [String: Any]
        #expect(cloud?["daily_budget"] as? Double == 15.0)

        let cors = merged["cors_origins"] as? [String]
        #expect(cors == ["http://localhost:3000", "https://myagent.dev"])

        #expect(merged["custom_user_annotation"] as? String == "keep-me")
    }

    // MARK: — File merge and .bak backup

    @Test("mergeAndSave creates .json.bak before overwriting existing config")
    func testMergeAndSaveCreatesBackup() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GemConfigTests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let configFileURL = tempDir.appendingPathComponent("config.json")
        let backupFileURL = tempDir.appendingPathComponent("config.json.bak")

        // 1. Initial write with user custom sections
        let initialConfig: [String: Any] = [
            "exe": "gemm_initial",
            "port": 8080,
            "host": "127.0.0.1",
            "sampling": [
                "temperature": 0.2
            ],
            "cors_origins": ["*"]
        ]
        let initialData = try JSONSerialization.data(withJSONObject: initialConfig, options: [.prettyPrinted, .sortedKeys])
        try initialData.write(to: configFileURL)

        #expect(!FileManager.default.fileExists(atPath: backupFileURL.path))

        // 2. Perform mergeAndSave
        let updates: [String: Any] = [
            "exe": "gemm_updated",
            "port": 8888,
            "host": "127.0.0.1"
        ]
        try ServerConfig.mergeAndSave(to: configFileURL, systemUpdates: updates, createBackup: true)

        // 3. Verify backup file was created and contains initial content
        #expect(FileManager.default.fileExists(atPath: backupFileURL.path))
        let backupData = try Data(contentsOf: backupFileURL)
        let backupDict = try JSONSerialization.jsonObject(with: backupData) as! [String: Any]
        #expect(backupDict["exe"] as? String == "gemm_initial")
        #expect(backupDict["port"] as? Int == 8080)

        // 4. Verify updated config merged system keys and preserved user fields
        let updatedData = try Data(contentsOf: configFileURL)
        let updatedDict = try JSONSerialization.jsonObject(with: updatedData) as! [String: Any]
        #expect(updatedDict["exe"] as? String == "gemm_updated")
        #expect(updatedDict["port"] as? Int == 8888)
        let sampling = updatedDict["sampling"] as? [String: Any]
        #expect(sampling?["temperature"] as? Double == 0.2)
        #expect(updatedDict["cors_origins"] as? [String] == ["*"])
    }

    // MARK: — End-to-end Onboarding Re-run Simulation

    @Test("Re-running onboarding does not lose user fields and creates .bak")
    func testReRunningOnboardingPreservesUserFields() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("OnboardReRun_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let serverConfigFile = tempDir.appendingPathComponent("config.json")
        let backupFile = tempDir.appendingPathComponent("config.json.bak")

        // First onboarding save
        try ServerConfig.mergeAndSave(
            to: serverConfigFile,
            systemUpdates: [
                "exe": "/bin/gemm",
                "args": ["serve", "--model", "model-1"],
                "port": 8080,
                "host": "127.0.0.1"
            ],
            createBackup: true
        )

        // User edits config to add custom settings
        var currentDict = try ServerConfig.load(from: serverConfigFile)
        currentDict["sampling"] = ["temperature": 0.4, "top_p": 0.8]
        currentDict["cloud"] = ["daily_budget": 5.0]
        let modifiedData = try JSONSerialization.data(withJSONObject: currentDict, options: [.prettyPrinted, .sortedKeys])
        try modifiedData.write(to: serverConfigFile)

        // Second onboarding run (re-running onboard)
        try ServerConfig.mergeAndSave(
            to: serverConfigFile,
            systemUpdates: [
                "exe": "/usr/local/bin/gemm",
                "args": ["serve", "--model", "model-2"],
                "port": 8080,
                "host": "127.0.0.1"
            ],
            createBackup: true
        )

        // Verify backup was created
        #expect(FileManager.default.fileExists(atPath: backupFile.path))

        // Verify user fields remained intact
        let result = try ServerConfig.load(from: serverConfigFile)
        #expect(result["exe"] as? String == "/usr/local/bin/gemm")
        #expect(result["args"] as? [String] == ["serve", "--model", "model-2"])
        let sampling = result["sampling"] as? [String: Any]
        #expect(sampling?["temperature"] as? Double == 0.4)
        #expect(sampling?["top_p"] as? Double == 0.8)
        let cloud = result["cloud"] as? [String: Any]
        #expect(cloud?["daily_budget"] as? Double == 5.0)
    }

    // MARK: — APIKeyAuthenticator Middleware

    private struct DummySource: RequestContextSource {
        var logger: Logger = Logger(label: "test")
    }

    private struct TestContext: RequestContext {
        typealias Source = DummySource
        var coreContext: CoreRequestContextStorage

        init(source: DummySource = DummySource()) {
            self.coreContext = .init(source: source)
        }
    }

    @Test("APIKeyAuthenticator allows valid Bearer token and x-api-key, rejects unauthorized")
    func testAPIKeyAuthenticator() async throws {
        let auth = APIKeyAuthenticator<TestContext>(apiKey: "secret-key")

        // Helper to simulate request handling
        func runMiddleware(path: String, headers: HTTPFields) async throws -> HTTPResponse.Status {
            let request = Request(
                head: .init(method: .post, scheme: nil, authority: nil, path: path, headerFields: headers),
                body: .init(buffer: ByteBuffer())
            )
            let context = TestContext()
            let response = try await auth.handle(request, context: context) { _, _ in
                Response(status: .ok)
            }
            return response.status
        }

        // Exempt endpoints bypass auth
        let healthStatus = try await runMiddleware(path: "/api/v1/health", headers: [:])
        #expect(healthStatus == .ok)

        let swaggerStatus = try await runMiddleware(path: "/swagger", headers: [:])
        #expect(swaggerStatus == .ok)

        // Protected endpoint without token returns 401
        let noTokenStatus = try await runMiddleware(path: "/v1/chat/completions", headers: [:])
        #expect(noTokenStatus == .unauthorized)

        // Protected endpoint with invalid token returns 401
        var invalidHeaders = HTTPFields()
        invalidHeaders[.authorization] = "Bearer wrong-token"
        let wrongTokenStatus = try await runMiddleware(path: "/v1/chat/completions", headers: invalidHeaders)
        #expect(wrongTokenStatus == .unauthorized)

        // Protected endpoint with valid Bearer token succeeds
        var validHeaders = HTTPFields()
        validHeaders[.authorization] = "Bearer secret-key"
        let validStatus = try await runMiddleware(path: "/v1/chat/completions", headers: validHeaders)
        #expect(validStatus == .ok)

        // Protected endpoint with valid x-api-key succeeds
        var xApiKeyHeaders = HTTPFields()
        xApiKeyHeaders[values: HTTPField.Name("x-api-key")!] = ["secret-key"]
        let xApiKeyStatus = try await runMiddleware(path: "/v1/chat/completions", headers: xApiKeyHeaders)
        #expect(xApiKeyStatus == .ok)
    }

    // MARK: — Review round 1 (#48): precedence, permissions, constant-time compare

    private func tempConfig(_ json: String) throws -> (dir: URL, file: URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("Cfg_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("config.json")
        try Data(json.utf8).write(to: file)
        return (dir, file)
    }

    @Test("Explicit CLI values win over --config even when they equal the defaults")
    func testCliBeatsConfigFile() throws {
        let (dir, file) = try tempConfig(#"{"port": 9000, "host": "0.0.0.0", "api_key": "from-file"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }

        let explicit = try ServeCommand.resolveNetwork(cliPort: 8080, cliHost: "127.0.0.1", cliApiKey: nil, configFile: file.path)
        #expect(explicit.port == 8080)
        #expect(explicit.host == "127.0.0.1")
        #expect(explicit.apiKey == "from-file")

        let fromFile = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: file.path)
        #expect(fromFile.port == 9000)
        #expect(fromFile.host == "0.0.0.0")

        let none = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: nil)
        #expect(none.port == 8080 && none.host == "127.0.0.1" && none.apiKey == nil)
    }

    @Test("A malformed or missing --config file is an error, not a silent fallback")
    func testBrokenConfigFileIsFatal() throws {
        let (dir, file) = try tempConfig("[1, 2, 3]")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: file.path)
        }
        #expect(throws: (any Error).self) {
            _ = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: dir.appendingPathComponent("absent.json").path)
        }
    }

    @Test("resolveApiKey: loopback needs no key, external bind generates only on opt-in")
    func testResolveApiKey() throws {
        #expect(try ServerConfig.resolveApiKey(host: "127.0.0.1", apiKey: nil, autoGenerateKeyIfMissing: false).key == nil)
        #expect(throws: ServerConfigError.missingApiKeyForNonLoopback(host: "0.0.0.0")) {
            _ = try ServerConfig.resolveApiKey(host: "0.0.0.0", apiKey: "  ", autoGenerateKeyIfMissing: false)
        }
        let generated = try ServerConfig.resolveApiKey(host: "0.0.0.0", apiKey: nil, autoGenerateKeyIfMissing: true)
        #expect(generated.generated)
        #expect(generated.key?.hasPrefix("gem_") == true)
        #expect(generated.key?.count == 4 + 64)
        let supplied = try ServerConfig.resolveApiKey(host: "0.0.0.0", apiKey: " k ", autoGenerateKeyIfMissing: true)
        #expect(supplied.key == "k" && !supplied.generated)
    }

    @Test("Re-running onboarding keeps a user-edited host/port; defaults fill only missing keys")
    func testDefaultsDoNotOverrideUserEdits() throws {
        let existing = Data(#"{"host": "0.0.0.0", "port": 9001, "api_key": "k"}"#.utf8)
        let merged = try ServerConfig.mergeConfiguration(
            existingData: existing,
            systemUpdates: ["args": ["serve"]],
            defaults: ["host": "127.0.0.1", "port": 8080, "log_level": "info"]
        )
        let dict = try #require(try JSONSerialization.jsonObject(with: merged) as? [String: Any])
        #expect(dict["host"] as? String == "0.0.0.0")
        #expect(dict["port"] as? Int == 9001)
        #expect(dict["log_level"] as? String == "info")
        #expect(dict["args"] as? [String] == ["serve"])
    }

    @Test("Config and its backup are written owner-only (0600) — they may carry api_key")
    func testConfigFilePermissions() throws {
        let (dir, file) = try tempConfig(#"{"api_key": "k"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        try ServerConfig.mergeAndSave(to: file, systemUpdates: ["port": 8080], createBackup: true)
        for path in [file.path, file.path + ".bak"] {
            let perms = try FileManager.default.attributesOfItem(atPath: path)[.posixPermissions] as? NSNumber
            #expect(perms?.intValue == 0o600, "\(path)")
        }
    }

    @Test("API key match is exact: a prefix, a longer string or a case change is rejected")
    func testApiKeyMatching() {
        let auth = APIKeyAuthenticator<TestContext>(apiKey: "secret-key")
        #expect(auth.matches("secret-key"))
        #expect(!auth.matches("secret"))
        #expect(!auth.matches("secret-key2"))
        #expect(!auth.matches("SECRET-KEY"))
        #expect(!auth.matches(""))
    }

    // MARK: — Review round 2 (#48): wiring, typed config values, launch args, backup order

    @Test("The real router rejects a protected route without the key and serves it with one")
    func testRouterPerimeterWiring() async throws {
        let config = try ServerConfig(modelPath: "stub", host: "0.0.0.0", apiKey: "k-123")
        let server = RESTServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), config: config)
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .unauthorized)
                #expect(response.headers[.wwwAuthenticate] == "Bearer")
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: [.authorization: "Bearer k-123"]) { response in
                #expect(response.status != .unauthorized)
            }
            try await client.execute(uri: "/v1/models", method: .get, headers: [.authorization: "Bearer k-12"]) { response in
                #expect(response.status == .unauthorized)
            }
            try await client.execute(uri: "/api/v1/health", method: .get) { response in
                #expect(response.status == .ok)
            }
        }
    }

    @Test("A config value of the wrong type is an error, not a silent default")
    func testWrongTypeInConfigIsFatal() throws {
        let (dir, file) = try tempConfig(#"{"port": "9000"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: nil, configFile: file.path)
        }
        let (dir2, file2) = try tempConfig(#"{"api_key": "from-file"}"#)
        defer { try? FileManager.default.removeItem(at: dir2) }
        let blank = try ServeCommand.resolveNetwork(cliPort: nil, cliHost: nil, cliApiKey: "  ", configFile: file2.path)
        #expect(blank.apiKey == "from-file", "an empty CLI/env key must not mask the config key")

        // An explicit CLI value never consults the bad field.
        let ok = try ServeCommand.resolveNetwork(cliPort: 8081, cliHost: nil, cliApiKey: nil, configFile: file.path)
        #expect(ok.port == 8081)
    }

    @Test("Onboarding launch args: only --model is replaced, user flags survive")
    func testLaunchArgsPreserveUserFlags() {
        #expect(OnboardingFlow.launchArgs(existing: nil, model: "m1") == ["serve", "--model", "m1"])
        #expect(OnboardingFlow.launchArgs(existing: ["serve", "--model", "old", "--max-tokens", "8192"], model: "m2")
                == ["serve", "--model", "m2", "--max-tokens", "8192"])
        #expect(OnboardingFlow.launchArgs(existing: ["serve", "--host", "0.0.0.0"], model: "m3")
                == ["serve", "--host", "0.0.0.0", "--model", "m3"])
    }

    @Test("A malformed config does not overwrite the last good backup")
    func testMalformedConfigKeepsBackup() throws {
        let (dir, file) = try tempConfig(#"{"port": 8080}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        try ServerConfig.mergeAndSave(to: file, systemUpdates: ["host": "127.0.0.1"], createBackup: true)
        let goodBackup = try Data(contentsOf: URL(fileURLWithPath: file.path + ".bak"))

        try Data("[broken".utf8).write(to: file)
        #expect(throws: (any Error).self) {
            try ServerConfig.mergeAndSave(to: file, systemUpdates: ["host": "127.0.0.1"], createBackup: true)
        }
        #expect(try Data(contentsOf: URL(fileURLWithPath: file.path + ".bak")) == goodBackup)
    }
}
