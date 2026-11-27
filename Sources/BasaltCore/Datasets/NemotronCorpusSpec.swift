import Foundation

public enum NemotronPreset: String, Sendable, Codable, Equatable, CaseIterable {
    case postV2 = "nemotron:post-v2"
    case postV1 = "nemotron:post-v1"
    case code = "nemotron:code"
    case math = "nemotron:math"
    case crossthink = "nemotron:crossthink"
    case personas = "nemotron:personas"

    public var repoID: String {
        switch self {
        case .postV2: "nvidia/Nemotron-Post-Training-Dataset-v2"
        case .postV1: "nvidia/Llama-Nemotron-Post-Training-Dataset"
        case .code: "nvidia/OpenCodeReasoning"
        case .math: "nvidia/OpenMathReasoning"
        case .crossthink: "nvidia/Nemotron-CrossThink"
        case .personas: "nvidia/Nemotron-Personas"
        }
    }

    public var defaultSubset: String? {
        switch self {
        case .postV2, .code, .math, .crossthink, .personas: nil
        case .postV1: "SFT"
        }
    }

    public var defaultSplit: String {
        switch self {
        case .postV2, .postV1: "chat"
        case .code: "split_0"
        case .math: "cot"
        case .crossthink, .personas: "train"
        }
    }

    public var isGatedByDefault: Bool {
        self == .postV2
    }

    public var licenseSummary: String {
        switch self {
        case .postV2: "CC-BY-4.0 / NVIDIA Open Model License"
        case .postV1, .code, .math, .crossthink, .personas: "CC-BY-4.0"
        }
    }
}

public enum CorpusLocation: Sendable, Codable, Equatable {
    case preset(NemotronPreset)
    case huggingFace(String)
    case localFile(String)

    public init(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let preset = NemotronPreset(rawValue: trimmed) {
            self = .preset(preset)
            return
        }
        if trimmed.hasPrefix("hf:") {
            self = .huggingFace(String(trimmed.dropFirst(3)))
            return
        }
        if trimmed.hasPrefix("/") || trimmed.hasPrefix("./") || trimmed.hasPrefix("../")
            || trimmed.hasPrefix("~") || trimmed.hasSuffix(".jsonl") || trimmed.hasSuffix(".json") {
            self = .localFile(trimmed)
            return
        }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        if parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty {
            self = .huggingFace(trimmed)
            return
        }
        self = .localFile(trimmed)
    }

    public var identifier: String {
        switch self {
        case .preset(let preset): preset.rawValue
        case .huggingFace(let repo): "hf:\(repo)"
        case .localFile(let path): path
        }
    }

    public var datasetRepo: String? {
        switch self {
        case .preset(let preset): preset.repoID
        case .huggingFace(let repo): repo
        case .localFile: nil
        }
    }
}

public struct CorpusSourceSpec: Sendable, Codable, Equatable {
    public var source: String
    public var subset: String?
    public var split: String?
    public var maxRows: Int = 512
    public var minPassRate: Double?
    public var includeReasoning: Bool = true
    public var seed: UInt64 = 42

    public static let rowCountRange = 2 ... 50_000

    public init(
        source: String,
        subset: String? = nil,
        split: String? = nil,
        maxRows: Int = 512,
        includeReasoning: Bool = true
    ) {
        self.source = source
        self.subset = subset
        self.split = split
        self.maxRows = maxRows
        self.includeReasoning = includeReasoning
    }

    public var location: CorpusLocation { CorpusLocation(source) }

    public var resolvedSubset: String? {
        if let subset, !subset.isEmpty { return subset }
        if case .preset(let preset) = location { return preset.defaultSubset }
        return nil
    }

    public var resolvedSplit: String? {
        if let split, !split.isEmpty { return split }
        if case .preset(let preset) = location { return preset.defaultSplit }
        return nil
    }

    public var licenseNotice: String {
        if case .preset(let preset) = location { return preset.licenseSummary }
        return location.datasetRepo != nil ? "Hugging Face Hub dataset terms apply" : "local-corpus"
    }

    public func validate(prefix: String = "train") throws(BasaltError) {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw .invalidRequestStructure(details: "\(prefix): corpus source cannot be empty")
        }
        if case .huggingFace(let repo) = location {
            let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
                throw .invalidRequestStructure(details: "\(prefix): Hugging Face dataset must be <org>/<repo>, got \(repo)")
            }
        }
        guard Self.rowCountRange.contains(maxRows) else {
            throw .invalidRequestStructure(details: "\(prefix): maxRows must be in \(Self.rowCountRange)")
        }
        if let rate = minPassRate, !rate.isFinite || !(0.0 ... 1.0).contains(rate) {
            throw .invalidRequestStructure(details: "\(prefix): minPassRate must lie in [0, 1]")
        }
    }
}

public struct CorpusExample: Sendable, Codable, Equatable {
    public var contextMessages: [[String: String]]
    public var reasoning: String?
    public var answer: String
    public var passRate: Double?

    public init(
        contextMessages: [[String: String]],
        reasoning: String? = nil,
        answer: String,
        passRate: Double? = nil
    ) {
        self.contextMessages = contextMessages
        self.reasoning = reasoning
        self.answer = answer
        self.passRate = passRate
    }
}

public struct CorpusTrainRequest: Sendable, Codable, Equatable {
    public static let defaultHeldOutPrompts = [
        "Explain what a hash table is in two sentences.",
        "Write a haiku about autumn.",
        "What is the capital of Australia?",
        "Summarize why the sky looks blue.",
    ]

    public var corpus: CorpusSourceSpec
    public var heldOutFraction: Float = 0.1
    public var layers: [Int]?
    public var rank: Int = 16
    public var method: AdapterMethod?
    public var iterations: Int = 200
    public var batchSize: Int = 2
    public var learningRate: Float = 1e-4
    public var optimizer: TrainingOptimizerKind = .adamw
    public var optimizerOptions: TrainingOptimizerOptions?
    public var maxSeqLen: Int = 2048
    public var checkpointEvery: Int = 50
    public var resume: Bool = false
    public var applyOverlay: Bool = false
    public var gate: GateRequest
    public var output: String?
    public var dpoPairsPath: String?
    public var quantizedBaseBits: Int?
    public var adaLoRATargetRank: Int?

    public var effectiveMethod: AdapterMethod { method ?? .lora(rank: rank) }
    public var effectiveOptimizerOptions: TrainingOptimizerOptions {
        var resolved = optimizerOptions ?? TrainingOptimizerOptions(kind: optimizer)
        resolved.kind = optimizer
        return resolved
    }
    public var effectiveLearningRate: Float {
        (try? QuantizedLRPolicy.effectiveLearningRate(baseLR: learningRate, weightBits: quantizedBaseBits)) ?? learningRate
    }

    public init(corpus: CorpusSourceSpec, gate: GateRequest = GateRequest(heldOut: Self.defaultHeldOutPrompts)) {
        self.corpus = corpus
        self.gate = gate
    }
}

public struct PreparedCorpusSplit: Sendable, Equatable {
    public var train: [CorpusExample]
    public var heldOut: [CorpusExample]
    public var rejected: [String: Int]

    public init(train: [CorpusExample], heldOut: [CorpusExample], rejected: [String: Int]) {
        self.train = train
        self.heldOut = heldOut
        self.rejected = rejected
    }
}

public struct AdapterProvenance: Sendable, Codable, Equatable {
    public var datasetSource: String
    public var datasetRepo: String?
    public var subset: String?
    public var split: String?
    public var license: String
    public var trainRows: Int
    public var heldOutRows: Int
    public var rowDigest: String
    public var baseModelFingerprint: String
    public var method: AdapterMethod
    public var iterations: Int
    public var nllBefore: Float
    public var nllAfter: Float
    public var gatePassed: Bool?

    public init(
        corpus: CorpusSourceSpec,
        datasetRepo: String?,
        baseModelFingerprint: String,
        method: AdapterMethod
    ) {
        self.datasetSource = corpus.source
        self.datasetRepo = datasetRepo
        self.subset = corpus.resolvedSubset
        self.split = corpus.resolvedSplit
        self.license = corpus.licenseNotice
        self.trainRows = 0
        self.heldOutRows = 0
        self.rowDigest = ""
        self.baseModelFingerprint = baseModelFingerprint
        self.method = method
        self.iterations = 0
        self.nllBefore = 0
        self.nllAfter = 0
        self.gatePassed = nil
    }
}

public struct CorpusTrainReport: Sendable, Codable {
    public var fetchedRows: Int = 0
    public var keptRows: Int = 0
    public var heldOutRows: Int = 0
    public var rejected: [String: Int] = [:]
    public var losses: [Float] = []
    public var nllBefore: Float?
    public var nllAfter: Float?
    public var resumedAt: Int?
    public var cacheFile: String?
    public var adapterDirectory: String?
    public var provenance: AdapterProvenance?
    public var verdict: AdapterVerdict?

    public init() {}
}
