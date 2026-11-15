import Foundation
import CryptoKit

public struct BasaltProvenance: Sendable, Equatable, Codable {
    public var gitCommit: String
    public var composeSHA256: String
    public var slotDigests: [String: String]
    public var macOSBuild: String
    public var chip: String
    public var unifiedMemoryGB: Int
    public var gpuLockAcquired: Bool
    public var wallClockSeconds: Double
    public var cloudSpendUSD: Double
    public var timestampISO8601: String

    public init(
        gitCommit: String = "unknown",
        composeSHA256: String,
        chip: String = "Apple Silicon",
        unifiedMemoryGB: Int = 0,
        timestampISO8601: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.gitCommit = gitCommit
        self.composeSHA256 = composeSHA256
        self.slotDigests = [:]
        self.macOSBuild = ProcessInfo.processInfo.operatingSystemVersionString
        self.chip = chip
        self.unifiedMemoryGB = unifiedMemoryGB
        self.gpuLockAcquired = false
        self.wallClockSeconds = 0
        self.cloudSpendUSD = 0
        self.timestampISO8601 = timestampISO8601
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func sha256Hex(_ text: String) -> String {
        sha256Hex(Data(text.utf8))
    }

    enum CodingKeys: String, CodingKey {
        case gitCommit = "git_commit"
        case composeSHA256 = "compose_sha256"
        case slotDigests = "slot_digests"
        case macOSBuild = "macos_build"
        case chip
        case unifiedMemoryGB = "unified_memory_gb"
        case gpuLockAcquired = "gpu_lock_acquired"
        case wallClockSeconds = "wall_clock_seconds"
        case cloudSpendUSD = "cloud_spend_usd"
        case timestampISO8601 = "timestamp_iso8601"
    }
}

public struct BenchmarkManifest: Sendable, Equatable, Codable {
    public var schemaVersion: Int
    public var runID: String
    public var composeName: String
    public var suite: String
    public var seed: UInt64
    public var temperature: Double
    public var endpoint: String?
    public var adapters: [String]
    public var targets: [String]

    public init(
        runID: String, composeName: String, suite: String,
        seed: UInt64 = 42, temperature: Double = 0.0, endpoint: String? = nil
    ) {
        self.schemaVersion = 1
        self.runID = runID
        self.composeName = composeName
        self.suite = suite
        self.seed = seed
        self.temperature = temperature
        self.endpoint = endpoint
        self.adapters = []
        self.targets = []
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case runID = "run_id"
        case composeName = "compose_name"
        case suite, seed, temperature, endpoint, adapters, targets
    }
}

public struct BenchmarkToolCallRecord: Sendable, Equatable, Codable {
    public var id: String
    public var name: String
    public var arguments: String
    public var output: String?

    public init(id: String, name: String, arguments: String, output: String? = nil) {
        (self.id, self.name, self.arguments, self.output) = (id, name, arguments, output)
    }
}

public struct BenchmarkTrajectoryStep: Sendable, Equatable, Codable {
    public var role: String
    public var content: String
    public var toolCalls: [BenchmarkToolCallRecord]
    public var timestampMs: Double?

    public init(role: String, content: String, toolCalls: [BenchmarkToolCallRecord] = [], timestampMs: Double? = nil) {
        (self.role, self.content, self.toolCalls, self.timestampMs) = (role, content, toolCalls, timestampMs)
    }

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case timestampMs = "timestamp_ms"
    }
}

public struct BenchmarkSampleRecord: Sendable, Equatable, Codable {
    public var id: String
    public var adapter: String
    public var dataset: String?
    public var category: String?
    public var split: String?
    public var model: String
    public var prompt: String
    public var answer: String
    public var expected: String?
    public var reasoningTrace: String?
    public var codeCompletions: [String]
    public var modelPatch: String?
    public var gridAttempts: [[[Int]]]
    public var toolCalls: [BenchmarkToolCallRecord]
    public var trajectory: [BenchmarkTrajectoryStep]
    public var latencyMs: Double
    public var ttftMs: Double?
    public var promptTokens: Int
    public var completionTokens: Int
    public var decodeTPS: Double?
    public var costUSD: Double
    public var score: Double
    public var passed: Bool
    public var metadata: [String: String]

    public init(
        id: String, adapter: String, model: String = "pipeline",
        prompt: String = "", answer: String = "", score: Double = 0
    ) {
        self.id = id
        self.adapter = adapter
        self.dataset = nil
        self.category = nil
        self.split = nil
        self.model = model
        self.prompt = prompt
        self.answer = answer
        self.expected = nil
        self.reasoningTrace = nil
        self.codeCompletions = []
        self.modelPatch = nil
        self.gridAttempts = []
        self.toolCalls = []
        self.trajectory = []
        self.latencyMs = 0
        self.ttftMs = nil
        self.promptTokens = 0
        self.completionTokens = 0
        self.decodeTPS = nil
        self.costUSD = 0
        self.score = score
        self.passed = score >= 1.0
        self.metadata = [:]
    }

    enum CodingKeys: String, CodingKey {
        case id, adapter, dataset, category, split, model, prompt, answer, expected
        case reasoningTrace = "reasoning_trace"
        case codeCompletions = "code_completions"
        case modelPatch = "model_patch"
        case gridAttempts = "grid_attempts"
        case toolCalls = "tool_calls"
        case trajectory
        case latencyMs = "latency_ms"
        case ttftMs = "ttft_ms"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case decodeTPS = "decode_tps"
        case costUSD = "cost_usd"
        case score, passed, metadata
    }
}
