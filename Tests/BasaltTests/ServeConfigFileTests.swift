import Foundation
import Testing
@testable import BasaltCLI

@Suite("basalt serve --config basalt.yaml sets serve options from defaults.model, defaults.serve and defaults.mcp; flags win (#344)")
struct ServeConfigFileTests {
    struct Group: Sendable, CustomTestStringConvertible {
        let yaml: String
        let flags: [String]
        let read: @Sendable (ServeCommand) -> String
        var testDescription: String { flags.joined(separator: " ") }
    }

    static let groups: [Group] = [
        Group(yaml: "model: { repo: org/model, quant: 4bit }", flags: ["--model", "org/model", "--quant", "4bit"]) { "\($0.model ?? "") \($0.quant ?? "")" },
        Group(yaml: "serve: { port: 9090, host: 0.0.0.0, max_tokens: 4096 }", flags: ["--port", "9090", "--host", "0.0.0.0", "--max-tokens", "4096"]) {
            "\($0.port ?? 0) \($0.host ?? "") \($0.maxTokens)"
        },
        Group(yaml: "serve: { kv: { bits: 8, zones: true, evict: 0.5 } }", flags: ["--kv-bits", "8", "--kv-zones", "--kv-evict", "0.5"]) {
            "\(String(describing: $0.kvBits)) \($0.kvZones) \($0.kvEvict ?? 0)"
        },
        Group(yaml: "serve: { prompt_cache: { mb: 512, hybrid_anchors: 2, disk_days: 3 } }", flags: ["--prompt-cache-mb", "512", "--hybrid-anchors", "2", "--prompt-cache-disk-days", "3"]) {
            "\($0.promptCacheMB ?? 0) \($0.hybridAnchors) \($0.promptCacheDisk.retentionDays)"
        },
        Group(yaml: "serve: { cache_blend: { enabled: true, ratio: 0.5 } }", flags: ["--cache-blend", "--cache-blend-ratio", "0.5"]) { "\($0.cacheBlend) \($0.cacheBlendRatio)" },
        Group(yaml: "serve: { speculative: { lookup: true, recycling: true, mtp_block: 2 } }", flags: ["--speculative-lookup", "--speculative-recycling", "--mtp-block", "2"]) {
            "\($0.speculativeLookup) \($0.speculativeRecycling) \($0.mtpBlock)"
        },
        Group(yaml: "serve: { sampling: { top_k: 40, min_p: 0.05, seed: 7, reasoning_effort: none } }", flags: ["--top-k", "40", "--min-p", "0.05", "--seed", "7", "--reasoning-effort", "none"]) {
            "\($0.topK ?? 0) \($0.minP ?? 0) \($0.seed ?? 0) \($0.reasoningEffort ?? "")"
        },
        Group(yaml: "serve: { experimental_adaptive_thinking_budget: true }", flags: ["--experimental-adaptive-thinking-budget"]) { "\($0.thinking.experimentalAdaptiveThinkingBudget)" },
        Group(yaml: "serve: { max_concurrent: 2, batching: { slots: 2 } }", flags: ["--max-concurrent", "2", "--batch-slots", "2"]) { "\($0.maxConcurrent) \($0.batchSlots)" },
        Group(yaml: "serve: { memory: { wired_limit: false, coordinator: false, model_probe: false } }", flags: ["--no-wired-limit", "--no-memory-coordinator", "--no-model-probe"]) {
            "\($0.wiredLimit) \($0.noMemoryCoordinator) \($0.modelProbe)"
        },
        Group(yaml: "serve: { health: { canary_minutes: 5, stats: true, stats_interval: 30 } }", flags: ["--canary-minutes", "5", "--stats", "--stats-interval", "30"]) {
            "\($0.canaryMinutes ?? 0) \($0.stats) \($0.statsInterval)"
        },
        Group(yaml: "serve: { session_log: { mode: meta, redact: off, retention_days: 3 } }", flags: ["--session-log", "meta", "--session-log-redact", "off", "--session-log-retention-days", "3"]) {
            "\($0.sessionLog.mode) \($0.sessionLog.redact) \($0.sessionLog.retentionDays)"
        },
        Group(yaml: "serve: { access: { rate_limit: 60, allow_origins: [\"https://a.example\"] } }", flags: ["--rate-limit", "60", "--allow-origin", "https://a.example"]) {
            "\($0.network.rateLimit ?? 0) \($0.allowOrigins)"
        },
        Group(yaml: "serve: { cloud: { fallback: org/cloud, cost_threshold: 0.7 } }", flags: ["--cloud-fallback", "org/cloud", "--cost-threshold", "0.7"]) {
            "\($0.cloud.model ?? "") \($0.cloud.costThreshold ?? 0)"
        },
        Group(yaml: "mcp: { enabled: true, research_tools: true, image_dirs: [/tmp/a, /tmp/b] }", flags: ["--mcp", "--research-tools", "--image-dir", "/tmp/a", "--image-dir", "/tmp/b"]) {
            "\($0.mcp) \($0.researchTools) \($0.imageDirs)"
        },
    ]

    static func serve(_ yaml: String, _ cli: [String] = []) throws -> ServeCommand {
        let url = FileManager.default.temporaryDirectory.appending(path: "basalt-\(UUID().uuidString).yaml")
        try yaml.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        return try ServeCommand.parse(Array(ServeConfigFile.expand(["serve", "--config", url.path] + cli).dropFirst()))
    }

    static func issues(_ yaml: String) -> [ServeConfigError.Issue] {
        do {
            _ = try ServeConfigFile.tokens(yaml: yaml, file: "basalt.yaml", cli: [])
            return []
        } catch let ServeConfigError.invalid(_, issues) {
            return issues
        } catch {
            return []
        }
    }

    @Test("Each option group in the file resolves to exactly what its flags give, and differs from the built-in default", arguments: groups)
    func fileEqualsFlags(_ group: Group) throws {
        let fromFile = try Self.serve("version: 1\ndefaults:\n  \(group.yaml)\n")
        let fromFlags = try ServeCommand.parse(group.flags)
        #expect(group.read(fromFile) == group.read(fromFlags))
        #expect(group.read(fromFile) != group.read(try ServeCommand.parse([])))
    }

    @Test("CLI flags beat the file, the file beats defaults, and a CLI list replaces the file's list")
    func precedence() throws {
        let yaml = "defaults:\n  serve:\n    port: 9090\n    max_queue: 3\n    memory: { wired_limit: false }\n    access: { allow_origins: [\"https://a.example\"] }\n"
        let command = try Self.serve(yaml, ["--port", "7070", "--wired-limit", "--allow-origin", "https://b.example"])
        #expect(command.port == 7070)
        #expect(command.maxQueue == 3)
        #expect(command.wiredLimit)
        #expect(command.allowOrigins == ["https://b.example"])
        #expect(command.maxConcurrent == 1)
        #expect(try Self.serve("defaults:\n  serve: { port: null }\n").port == nil)
    }

    @Test("Unknown keys and wrong types are all reported at once, each with its key path, line and column")
    func strictValidation() {
        let yaml = "version: 1\ndefaults:\n  serve:\n    max_concurrent: two\n    prompt_cahce: { mb: 1 }\n    kv: { zones: maybe }\n  profiles: {}\nextends: []\n"
        #expect(Self.issues(yaml) == [
            .init(path: "defaults.serve.max_concurrent", line: 4, column: 21, message: "expected an integer, got \"two\""),
            .init(path: "defaults.serve.prompt_cahce", line: 5, column: 5, message: "unknown key"),
            .init(path: "defaults.serve.kv.zones", line: 6, column: 18, message: "expected true or false, got \"maybe\""),
            .init(path: "defaults.profiles", line: 7, column: 3, message: "unknown key"),
            .init(path: "extends", line: 8, column: 1, message: "unknown key"),
        ])
        #expect(Self.issues("version: 2\n").map(\.path) == ["version"])
        #expect(Self.issues("defaults:\n  serve:\n    access: { allow_origins: \"https://a.example\" }\n").map(\.message) == ["expected a list of strings, got \"https://a.example\""])
        #expect(Self.issues("defaults:\n  serve: [1]\n").map(\.message) == ["must be a mapping"])
    }

    @Test("A leading ~ in a path value expands to the home directory, as the shell would expand it in a flag; other values stay literal")
    func tildeExpands() throws {
        let home = NSHomeDirectory()
        let command = try Self.serve("defaults:\n  model: { path: ~/models/q }\n  serve: { session_log: { dir: ~/logs } }\n  mcp: { image_dirs: [~/a, /b, ~c] }\n")
        #expect(command.modelPath == home + "/models/q")
        #expect(command.sessionLog.directory == home + "/logs")
        #expect(command.imageDirs == [home + "/a", "/b", "~c"])
    }

    @Test("Malformed YAML, a missing file and an oversized file are unreadable; JSON configs and other commands pass through untouched")
    func passThroughAndUnreadable() throws {
        #expect(throws: ServeConfigError.self) { try ServeConfigFile.tokens(yaml: "defaults: [", file: "x.yaml", cli: []) }
        #expect(throws: ServeConfigError.self) { try ServeConfigFile.expand(["serve", "--config=/nonexistent/basalt.yml"]) }
        let big = FileManager.default.temporaryDirectory.appending(path: "basalt-\(UUID().uuidString).yaml")
        try Data(repeating: 0x20, count: ServeConfigFile.maxBytes + 1).write(to: big)
        defer { try? FileManager.default.removeItem(at: big) }
        #expect(throws: ServeConfigError.self) { try ServeConfigFile.expand(["serve", "--config", big.path]) }
        let json = ["serve", "--config", "~/.basalt/config.json", "--port", "1"]
        #expect(try ServeConfigFile.expand(json) == json)
        #expect(try ServeConfigFile.expand(["chat", "--config", "a.yaml"]) == ["chat", "--config", "a.yaml"])
    }
}
