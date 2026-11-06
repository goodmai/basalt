import Foundation

public protocol ModelLab: ModelResearch {
    func applyAdapter(_ source: AdapterSource, gate: GateRequest?) async throws(BasaltError) -> AdapterVerdict
    func gate(_ request: GateRequest) async throws(BasaltError) -> GateReport
    func pack(_ request: PackRequest) async throws(BasaltError) -> PackReport
    func contrast(_ request: ContrastRequest) async throws(BasaltError) -> ContrastReport
    func commitSteering(_ commit: SteeringCommit) async throws(BasaltError) -> AdapterVerdict
    func train(_ request: TrainRequest) async throws(BasaltError) -> TrainReport
}

extension ModelLab {
    public func applyAdapter(_ source: AdapterSource, gate: GateRequest?) async throws(BasaltError) -> AdapterVerdict { throw Self.unavailable("adapters") }
    public func gate(_ request: GateRequest) async throws(BasaltError) -> GateReport { throw Self.unavailable("quality gate") }
    public func pack(_ request: PackRequest) async throws(BasaltError) -> PackReport { throw Self.unavailable("tensor pack") }
    public func contrast(_ request: ContrastRequest) async throws(BasaltError) -> ContrastReport { throw Self.unavailable("contrastive steering") }
    public func commitSteering(_ commit: SteeringCommit) async throws(BasaltError) -> AdapterVerdict { throw Self.unavailable("steering commit") }
    public func train(_ request: TrainRequest) async throws(BasaltError) -> TrainReport { throw Self.unavailable("expert training") }

    static func unavailable(_ mode: String) -> BasaltError {
        .invalidRequestStructure(details: "\(mode) is not available on this engine")
    }
}

public enum AdapterSource: Sendable, Codable, Equatable {
    case directory(String)
    case rankOne(module: String, input: [Float], output: [Float], scale: Float)
}

public struct GateRequest: Sendable, Codable, Equatable {
    public var heldOut: [String]
    public var maxKL: Float = 0.5
    public var maxPerplexityRise: Float = 0.1
    public var target: [GateTarget] = []

    public init(heldOut: [String], maxKL: Float = 0.5, maxPerplexityRise: Float = 0.1) {
        self.heldOut = heldOut
        self.maxKL = maxKL
        self.maxPerplexityRise = maxPerplexityRise
    }
}

public struct GateTarget: Sendable, Codable, Equatable {
    public var prompt: String
    public var expected: String

    public init(prompt: String, expected: String) {
        self.prompt = prompt
        self.expected = expected
    }
}

public struct GateCheck: Sendable, Codable, Equatable {
    public var name: String
    public var passed: Bool
    public var value: Float
    public var limit: Float

    public init(name: String, passed: Bool, value: Float, limit: Float) {
        self.name = name
        self.passed = passed
        self.value = value
        self.limit = limit
    }
}

public struct GateReport: Sendable, Codable {
    public var passed: Bool { checks.allSatisfy(\.passed) }
    public var checks: [GateCheck]
    public var compare: CompareReport?

    public init(checks: [GateCheck], compare: CompareReport? = nil) {
        self.checks = checks
        self.compare = compare
    }
}

public struct AdapterVerdict: Sendable, Codable {
    public var overlay: OverlayInfo?
    public var report: GateReport?

    public init(overlay: OverlayInfo?, report: GateReport?) {
        self.overlay = overlay
        self.report = report
    }
}

public enum PackFormat: Sendable, Codable, Equatable {
    case affine(bits: Int, groupSize: Int)
    case mxfp4
    case nf4(groupSize: Int)
    case ternary
    case float16
    case bfloat16
}

public struct PackRequest: Sendable, Codable, Equatable {
    public var tensor: String
    public var format: PackFormat

    public init(tensor: String, format: PackFormat) {
        self.tensor = tensor
        self.format = format
    }
}

public struct PackReport: Sendable, Codable {
    public var overlay: OverlayInfo
    public var bytesBefore: ByteCount
    public var bytesAfter: ByteCount
    public var rmse: Float
    public var maxAbsError: Float
    public var relativeError: Float

    public init(overlay: OverlayInfo, bytesBefore: ByteCount, bytesAfter: ByteCount, rmse: Float, maxAbsError: Float, relativeError: Float) {
        self.overlay = overlay
        self.bytesBefore = bytesBefore
        self.bytesAfter = bytesAfter
        self.rmse = rmse
        self.maxAbsError = maxAbsError
        self.relativeError = relativeError
    }
}

public struct ContrastRequest: Sendable, Codable, Equatable {
    public var positive: [String]
    public var negative: [String]
    public var heldOut: [String]
    public var layers: [Int]?
    public var topHeads = 8
    public var scales: [Float] = [2, 4, 8]

    public static let classPrompts = 2...32
    public static let heldOutPrompts = 1...16
    public static let scaleCount = 1...8

    public init(positive: [String], negative: [String], heldOut: [String], layers: [Int]? = nil) {
        self.positive = positive
        self.negative = negative
        self.heldOut = heldOut
        self.layers = layers
    }
}

public struct DirectionLayer: Sendable, Codable, Equatable {
    public var index: Int
    public var norm: Float
    public var separation: Float

    public init(index: Int, norm: Float, separation: Float) {
        self.index = index
        self.norm = norm
        self.separation = separation
    }
}

public struct HeadEffect: Sendable, Codable, Equatable {
    public var layer: Int
    public var head: Int
    public var effect: Float

    public init(layer: Int, head: Int, effect: Float) {
        self.layer = layer
        self.head = head
        self.effect = effect
    }
}

public struct SteerTrial: Sendable, Codable, Equatable {
    public var scale: Float
    public var klDivergence: Float
    public var projectionShift: Float

    public init(scale: Float, klDivergence: Float, projectionShift: Float) {
        self.scale = scale
        self.klDivergence = klDivergence
        self.projectionShift = projectionShift
    }
}

public struct ContrastReport: Sendable, Codable {
    public var layers: [DirectionLayer]
    public var bestLayer: Int
    public var direction: [Float]
    public var heads: [HeadEffect]
    public var trials: [SteerTrial]

    public init(layers: [DirectionLayer], bestLayer: Int, direction: [Float], heads: [HeadEffect], trials: [SteerTrial]) {
        self.layers = layers
        self.bestLayer = bestLayer
        self.direction = direction
        self.heads = heads
        self.trials = trials
    }
}

public struct SteeringCommit: Sendable, Codable, Equatable {
    public var layer: Int
    public var direction: [Float]
    public var scale: Float
    public var gate: GateRequest
    public var positive: [String] = []

    public init(layer: Int, direction: [Float], scale: Float, gate: GateRequest) {
        self.layer = layer
        self.direction = direction
        self.scale = scale
        self.gate = gate
    }
}

public enum TrainingTasks: Sendable, Codable, Equatable {
    case arithmetic(count: Int, seed: UInt64)
    case file(String)
}

public struct TrainRequest: Sendable, Codable, Equatable {
    public var tasks: TrainingTasks
    public var teacher: String?
    public var teacherModel: String?
    public var layers: [Int]?
    public var experts: [Int]?
    public var rank = 8
    public var iterations = 200
    public var learningRate: Float = 1e-5
    public var gate: GateRequest
    public var output: String?

    public init(tasks: TrainingTasks, gate: GateRequest) {
        self.tasks = tasks
        self.gate = gate
    }
}

public struct TrainReport: Sendable, Codable {
    public var collected: Int
    public var verified: Int
    public var losses: [Float]
    public var accuracyBefore: Float
    public var accuracyAfter: Float
    public var adapterDirectory: String?
    public var verdict: AdapterVerdict

    public init(collected: Int, verified: Int, losses: [Float], accuracyBefore: Float, accuracyAfter: Float, verdict: AdapterVerdict) {
        self.collected = collected
        self.verified = verified
        self.losses = losses
        self.accuracyBefore = accuracyBefore
        self.accuracyAfter = accuracyAfter
        self.verdict = verdict
    }
}
