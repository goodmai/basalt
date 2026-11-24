import Foundation

public enum EditOp: Sendable, Codable, Equatable {
    case rank1Damp(module: String, vectorKey: String, alpha: Float)
    case rank1(module: String, outputKey: String, inputKey: String, scale: Float)
    case shift(tensor: String, start: Int, count: Int, valueKey: String)
    case clip(tensor: String, factor: Float)
    case rows(tensor: String, rowIndices: [Int], valuesKey: String)

    private enum CodingKeys: String, CodingKey {
        case op
        case module
        case vectorKey = "vector_key"
        case alpha
        case outputKey = "output_key"
        case inputKey = "input_key"
        case scale
        case tensor
        case start
        case count
        case valueKey = "value_key"
        case factor
        case rowIndices = "row_indices"
        case valuesKey = "values_key"
    }

    private enum Kind: String, Codable {
        case rank1Damp = "rank1_damp"
        case rank1
        case shift
        case clip
        case rows
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .op)
        switch kind {
        case .rank1Damp:
            self = .rank1Damp(
                module: try container.decode(String.self, forKey: .module),
                vectorKey: try container.decode(String.self, forKey: .vectorKey),
                alpha: try container.decode(Float.self, forKey: .alpha)
            )
        case .rank1:
            self = .rank1(
                module: try container.decode(String.self, forKey: .module),
                outputKey: try container.decode(String.self, forKey: .outputKey),
                inputKey: try container.decode(String.self, forKey: .inputKey),
                scale: try container.decode(Float.self, forKey: .scale)
            )
        case .shift:
            self = .shift(
                tensor: try container.decode(String.self, forKey: .tensor),
                start: try container.decode(Int.self, forKey: .start),
                count: try container.decode(Int.self, forKey: .count),
                valueKey: try container.decode(String.self, forKey: .valueKey)
            )
        case .clip:
            self = .clip(
                tensor: try container.decode(String.self, forKey: .tensor),
                factor: try container.decode(Float.self, forKey: .factor)
            )
        case .rows:
            self = .rows(
                tensor: try container.decode(String.self, forKey: .tensor),
                rowIndices: try container.decode([Int].self, forKey: .rowIndices),
                valuesKey: try container.decode(String.self, forKey: .valuesKey)
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .rank1Damp(let module, let vectorKey, let alpha):
            try container.encode(Kind.rank1Damp, forKey: .op)
            try container.encode(module, forKey: .module)
            try container.encode(vectorKey, forKey: .vectorKey)
            try container.encode(alpha, forKey: .alpha)
        case .rank1(let module, let outputKey, let inputKey, let scale):
            try container.encode(Kind.rank1, forKey: .op)
            try container.encode(module, forKey: .module)
            try container.encode(outputKey, forKey: .outputKey)
            try container.encode(inputKey, forKey: .inputKey)
            try container.encode(scale, forKey: .scale)
        case .shift(let tensor, let start, let count, let valueKey):
            try container.encode(Kind.shift, forKey: .op)
            try container.encode(tensor, forKey: .tensor)
            try container.encode(start, forKey: .start)
            try container.encode(count, forKey: .count)
            try container.encode(valueKey, forKey: .valueKey)
        case .clip(let tensor, let factor):
            try container.encode(Kind.clip, forKey: .op)
            try container.encode(tensor, forKey: .tensor)
            try container.encode(factor, forKey: .factor)
        case .rows(let tensor, let rowIndices, let valuesKey):
            try container.encode(Kind.rows, forKey: .op)
            try container.encode(tensor, forKey: .tensor)
            try container.encode(rowIndices, forKey: .rowIndices)
            try container.encode(valuesKey, forKey: .valuesKey)
        }
    }

    public var targetName: String {
        switch self {
        case .rank1Damp(let module, _, _), .rank1(let module, _, _, _):
            module
        case .shift(let tensor, _, _, _), .clip(let tensor, _), .rows(let tensor, _, _):
            tensor
        }
    }

    public var tensorKeys: [String] {
        switch self {
        case .rank1Damp(_, let vectorKey, _):
            [vectorKey]
        case .rank1(_, let outputKey, let inputKey, _):
            [outputKey, inputKey]
        case .shift(_, _, _, let valueKey):
            [valueKey]
        case .clip:
            []
        case .rows(_, _, let valuesKey):
            [valuesKey]
        }
    }
}

public enum EditApprovalStatus: String, Sendable, Codable, Equatable {
    case pending
    case approved
    case rejected
    case withdrawn
}

public struct PairedTestSummary: Sendable, Codable, Equatable {
    public var problemCount: Int
    public var baseAccuracy: Double
    public var editedAccuracy: Double
    public var delta: Double
    public var standardError: Double
    public var wrongToRight: Int
    public var rightToWrong: Int

    enum CodingKeys: String, CodingKey {
        case problemCount = "problem_count"
        case baseAccuracy = "base_accuracy"
        case editedAccuracy = "edited_accuracy"
        case delta
        case standardError = "standard_error"
        case wrongToRight = "wrong_to_right"
        case rightToWrong = "right_to_wrong"
    }

    public init(
        problemCount: Int,
        baseAccuracy: Double,
        editedAccuracy: Double,
        standardError: Double,
        wrongToRight: Int,
        rightToWrong: Int
    ) {
        self.problemCount = problemCount
        self.baseAccuracy = baseAccuracy
        self.editedAccuracy = editedAccuracy
        self.delta = editedAccuracy - baseAccuracy
        self.standardError = standardError
        self.wrongToRight = wrongToRight
        self.rightToWrong = rightToWrong
    }
}

public struct EditProgram: Sendable, Codable, Equatable {
    public static let currentFormatVersion = 1
    public static let defaultMaxOps = 64

    public var formatVersion: Int
    public var sessionId: String
    public var baseModelFingerprint: String
    public var approval: EditApprovalStatus
    public var pairedTest: PairedTestSummary?
    public var ops: [EditOp]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case sessionId = "session_id"
        case baseModelFingerprint = "base_model_fingerprint"
        case approval
        case pairedTest = "paired_test"
        case ops
    }

    public init(
        formatVersion: Int = Self.currentFormatVersion,
        sessionId: String = "",
        baseModelFingerprint: String = "",
        approval: EditApprovalStatus = .pending,
        pairedTest: PairedTestSummary? = nil,
        ops: [EditOp]
    ) {
        self.formatVersion = formatVersion
        self.sessionId = sessionId
        self.baseModelFingerprint = baseModelFingerprint
        self.approval = approval
        self.pairedTest = pairedTest
        self.ops = ops
    }

    public var referencedTensorKeys: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for op in ops {
            for key in op.tensorKeys where seen.insert(key).inserted {
                ordered.append(key)
            }
        }
        return ordered
    }
}

public struct EditOpMetrics: Sendable, Codable, Equatable {
    public var opIndex: Int
    public var target: String
    public var rmse: Float
    public var maxAbsError: Float
    public var relativeError: Float

    public init(
        opIndex: Int,
        target: String,
        rmse: Float = 0,
        maxAbsError: Float = 0,
        relativeError: Float = 0
    ) {
        self.opIndex = opIndex
        self.target = target
        self.rmse = rmse
        self.maxAbsError = maxAbsError
        self.relativeError = relativeError
    }
}
