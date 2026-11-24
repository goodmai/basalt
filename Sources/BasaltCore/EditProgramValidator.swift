import Foundation

public enum EditForbiddenTargetKind: String, Sendable, Equatable {
    case embedding
    case finalNorm = "final_norm"
    case lmHead = "lm_head"
}

public enum EditValidationError: Error, Sendable, Equatable, CustomStringConvertible {
    case unsupportedFormatVersion(Int)
    case emptyProgram
    case tooManyOps(count: Int, limit: Int)
    case invalidIdentifier(field: String, value: String)
    case forbiddenTarget(target: String, kind: EditForbiddenTargetKind)
    case wholeTensorRescale(target: String, reason: String)
    case alphaOutOfRange(module: String, alpha: Float)
    case nonFiniteValue(target: String, detail: String)
    case invalidClipFactor(tensor: String, factor: Float)
    case invalidSlice(tensor: String, start: Int, count: Int, limit: Int?)
    case invalidRowIndices(tensor: String, reason: String)
    case missingTensor(key: String)
    case unexpectedTensor(key: String)
    case shapeMismatch(target: String, expected: String, actual: String)
    case budgetExceeded(required: Int64, budget: Int64)

    public var description: String {
        switch self {
        case .unsupportedFormatVersion(let version):
            return "unsupported edit program format version \(version)"
        case .emptyProgram:
            return "edit program contains no operations"
        case .tooManyOps(let count, let limit):
            return "edit program has \(count) ops, exceeding limit \(limit)"
        case .invalidIdentifier(let field, let value):
            return "invalid \(field) identifier '\(value)'"
        case .forbiddenTarget(let target, let kind):
            return "target '\(target)' is forbidden (\(kind.rawValue))"
        case .wholeTensorRescale(let target, let reason):
            return "whole-tensor rescale refused on '\(target)': \(reason)"
        case .alphaOutOfRange(let module, let alpha):
            return "rank1Damp alpha on '\(module)' must be finite with |alpha| <= 1, got \(alpha)"
        case .nonFiniteValue(let target, let detail):
            return "non-finite value for '\(target)': \(detail)"
        case .invalidClipFactor(let tensor, let factor):
            return "clip factor on '\(tensor)' must be finite and positive, got \(factor)"
        case .invalidSlice(let tensor, let start, let count, let limit):
            return "invalid slice [\(start)..<\(start + count)] on '\(tensor)'" + (limit.map { " of size \($0)" } ?? "")
        case .invalidRowIndices(let tensor, let reason):
            return "invalid row indices on '\(tensor)': \(reason)"
        case .missingTensor(let key):
            return "missing operand tensor '\(key)'"
        case .unexpectedTensor(let key):
            return "unreferenced operand tensor '\(key)'"
        case .shapeMismatch(let target, let expected, let actual):
            return "shape mismatch on '\(target)': expected \(expected), got \(actual)"
        case .budgetExceeded(let required, let budget):
            return "weight copy extra bytes \(required) exceed budget \(budget)"
        }
    }
}

public enum EditProgramValidator {
    private static let wrapperPrefixes: Set<String> = [
        "base_model", "language_model", "model", "transformer", "backbone", "text_model", "decoder",
    ]
    private static let embeddingSegments: Set<String> = [
        "embed_tokens", "word_embeddings", "tok_embeddings", "wte", "wpe", "embed_tokens_per_layer",
    ]
    private static let finalNormSegments: Set<String> = [
        "norm", "ln_f", "norm_f", "final_layernorm", "final_norm", "embedding_norm",
    ]
    private static let lmHeadSegments: Set<String> = [
        "lm_head", "output_head", "unembed",
    ]
    private static let scaleLeafSegments: Set<String> = [
        "scales", "global_scale", "weight_scale", "layer_scalar", "per_expert_scale",
        "input_scale", "output_scale", "pre_scale", "row_scale",
    ]

    public static func validate(
        _ program: EditProgram,
        maxOps: Int = EditProgram.defaultMaxOps,
        tensorShapes: [String: [Int]]? = nil,
        tensorValues: [String: [Float]]? = nil
    ) throws(EditValidationError) {
        guard program.formatVersion == EditProgram.currentFormatVersion else {
            throw .unsupportedFormatVersion(program.formatVersion)
        }
        guard maxOps > 0 else { throw .tooManyOps(count: program.ops.count, limit: maxOps) }
        guard !program.ops.isEmpty else { throw .emptyProgram }
        guard program.ops.count <= maxOps else { throw .tooManyOps(count: program.ops.count, limit: maxOps) }
        if let paired = program.pairedTest {
            try validatePairedTest(paired)
        }
        if let tensorValues {
            let referenced = Set(program.referencedTensorKeys)
            for key in tensorValues.keys.sorted() {
                try validateIdentifier(key, field: "tensor_key")
                guard referenced.contains(key) else { throw .unexpectedTensor(key: key) }
            }
        }
        for op in program.ops {
            try validateOp(op, tensorShapes: tensorShapes, tensorValues: tensorValues)
        }
    }

    public static func validateIdentifier(_ value: String, field: String) throws(EditValidationError) {
        guard !value.isEmpty, value.count <= 512 else {
            throw .invalidIdentifier(field: field, value: value)
        }
        guard !value.contains(".."), !value.hasPrefix("."), !value.hasSuffix(".") else {
            throw .invalidIdentifier(field: field, value: value)
        }
        for scalar in value.unicodeScalars {
            let allowed = (scalar.value >= 0x30 && scalar.value <= 0x39)
                || (scalar.value >= 0x41 && scalar.value <= 0x5A)
                || (scalar.value >= 0x61 && scalar.value <= 0x7A)
                || scalar == "." || scalar == "_" || scalar == "-"
            guard allowed else {
                throw .invalidIdentifier(field: field, value: value)
            }
        }
    }

    public static func forbiddenTargetKind(for path: String) -> EditForbiddenTargetKind? {
        let segments = path.split(separator: ".").map { $0.lowercased() }
        guard !segments.isEmpty else { return nil }
        for segment in segments {
            if embeddingSegments.contains(segment) || segment.hasPrefix("embed_") || segment.hasPrefix("word_embedding") || segment.hasPrefix("tok_embedding") {
                return .embedding
            }
            if lmHeadSegments.contains(segment) {
                return .lmHead
            }
        }
        var stripped = segments[...]
        while let first = stripped.first, wrapperPrefixes.contains(first), stripped.count > 1 {
            stripped = stripped.dropFirst()
        }
        if let root = stripped.first, finalNormSegments.contains(root) {
            return .finalNorm
        }
        return nil
    }

    public static func isScaleTensor(_ path: String) -> Bool {
        guard let leaf = path.split(separator: ".").last?.lowercased() else { return false }
        return scaleLeafSegments.contains(leaf)
    }

    public static func isNormWeight(_ path: String) -> Bool {
        let segments = path.split(separator: ".").map { $0.lowercased() }
        guard segments.count >= 2, segments.last == "weight" else { return false }
        let owner = segments[segments.count - 2]
        return owner == "norm" || owner.contains("layernorm") || owner.hasSuffix("_norm")
    }
}

extension EditProgramValidator {
    private static func validatePairedTest(_ summary: PairedTestSummary) throws(EditValidationError) {
        let metrics = [summary.baseAccuracy, summary.editedAccuracy, summary.delta, summary.standardError]
        guard summary.problemCount >= 0, summary.wrongToRight >= 0, summary.rightToWrong >= 0,
              metrics.allSatisfy(\.isFinite), summary.standardError >= 0 else {
            throw .nonFiniteValue(target: "paired_test", detail: "paired test summary contains invalid or non-finite values")
        }
    }

    private static func validateOp(
        _ op: EditOp,
        tensorShapes: [String: [Int]]?,
        tensorValues: [String: [Float]]?
    ) throws(EditValidationError) {
        try validateTargetAndOperands(op, tensorValues: tensorValues)
        switch op {
        case .rank1Damp(let module, _, let alpha):
            guard alpha.isFinite, abs(alpha) <= 1.0 else {
                throw .alphaOutOfRange(module: module, alpha: alpha)
            }
        case .rank1(let module, _, _, let scale):
            guard scale.isFinite else {
                throw .nonFiniteValue(target: module, detail: "scale must be finite")
            }
        case .shift(let tensor, let start, let count, _):
            try validateShift(tensor: tensor, start: start, count: count, shape: tensorShapes?[tensor])
        case .clip(let tensor, let factor):
            guard factor.isFinite, factor > 0 else {
                throw .invalidClipFactor(tensor: tensor, factor: factor)
            }
        case .rows(let tensor, let rowIndices, _):
            try validateRows(tensor: tensor, rowIndices: rowIndices, shape: tensorShapes?[tensor])
        }
    }

    private static func validateTargetAndOperands(
        _ op: EditOp,
        tensorValues: [String: [Float]]?
    ) throws(EditValidationError) {
        try validateIdentifier(op.targetName, field: "target")
        for key in op.tensorKeys {
            try validateIdentifier(key, field: "tensor_key")
        }
        if let forbidden = forbiddenTargetKind(for: op.targetName) {
            throw .forbiddenTarget(target: op.targetName, kind: forbidden)
        }
        if isScaleTensor(op.targetName) {
            throw .wholeTensorRescale(target: op.targetName, reason: "mutating scale parameters rescales the tensor")
        }
        guard let tensorValues else { return }
        for key in op.tensorKeys {
            guard let values = tensorValues[key] else { throw .missingTensor(key: key) }
            guard values.allSatisfy(\.isFinite) else {
                throw .nonFiniteValue(target: op.targetName, detail: "operand '\(key)' contains NaN or Inf")
            }
        }
    }

    private static func validateShift(
        tensor: String,
        start: Int,
        count: Int,
        shape: [Int]?
    ) throws(EditValidationError) {
        guard start >= 0, count > 0, start <= Int.max - count else {
            throw .invalidSlice(tensor: tensor, start: start, count: count, limit: shape?.reduce(1, *))
        }
        guard let shape else { return }
        let total = shape.reduce(1, *)
        guard start + count <= total else {
            throw .invalidSlice(tensor: tensor, start: start, count: count, limit: total)
        }
        if start == 0, count == total, (isNormWeight(tensor) || (shape.count >= 2 && tensor.hasSuffix(".weight"))) {
            throw .wholeTensorRescale(target: tensor, reason: "shifting an entire norm or weight matrix rescales the layer")
        }
    }

    private static func validateRows(
        tensor: String,
        rowIndices: [Int],
        shape: [Int]?
    ) throws(EditValidationError) {
        guard !rowIndices.isEmpty else {
            throw .invalidRowIndices(tensor: tensor, reason: "rowIndices must be non-empty")
        }
        var seen = Set<Int>()
        for row in rowIndices {
            guard row >= 0 else {
                throw .invalidRowIndices(tensor: tensor, reason: "negative row index \(row)")
            }
            guard seen.insert(row).inserted else {
                throw .invalidRowIndices(tensor: tensor, reason: "duplicate row index \(row)")
            }
        }
        guard let shape else { return }
        guard shape.count == 2, let totalRows = shape.first, totalRows > 0 else {
            throw .shapeMismatch(target: tensor, expected: "2D weight matrix", actual: "\(shape)")
        }
        for row in rowIndices where row >= totalRows {
            throw .invalidRowIndices(tensor: tensor, reason: "row \(row) out of bounds 0..<\(totalRows)")
        }
        guard seen.count < totalRows else {
            throw .wholeTensorRescale(target: tensor, reason: "replacing all \(totalRows) rows overwrites the whole tensor")
        }
    }
}
