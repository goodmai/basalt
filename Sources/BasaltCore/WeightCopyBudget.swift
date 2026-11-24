import Foundation

public struct ParameterDigest: Sendable, Equatable, Codable {
    public let name: String
    public let arrayIdentity: UInt
    public let dtype: String
    public let shape: [Int]
    public let byteCount: Int
    public let contentHash: String

    public init(
        name: String,
        arrayIdentity: UInt,
        dtype: String,
        shape: [Int],
        byteCount: Int,
        contentHash: String
    ) {
        self.name = name
        self.arrayIdentity = arrayIdentity
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
        self.contentHash = contentHash
    }
}

public struct FingerprintMismatch: Sendable, Equatable {
    public let name: String
    public let reason: String

    public init(name: String, reason: String) {
        self.name = name
        self.reason = reason
    }
}

public struct TensorFootprint: Sendable, Equatable {
    public let name: String
    public let shape: [Int]
    public let byteCount: Int64
    public let moduleByteCount: Int64
    public let computeElementBytes: Int64
    public let fusedGroupKey: String?

    public init(
        name: String,
        shape: [Int],
        byteCount: Int64,
        moduleByteCount: Int64,
        computeElementBytes: Int64 = 4,
        fusedGroupKey: String? = nil
    ) {
        self.name = name
        self.shape = shape
        self.byteCount = byteCount
        self.moduleByteCount = moduleByteCount
        self.computeElementBytes = computeElementBytes
        self.fusedGroupKey = fusedGroupKey
    }
}

public struct WeightCopyBudgetState: Sendable, Equatable {
    public let footprints: [String: TensorFootprint]
    public let fusedGroupBytes: [String: Int64]
    public let existingLoraBytes: Int64
    public let editedShiftTensors: Set<String>
    public let editedLayerTensors: Set<String>
    public let disabledFusedGroups: Set<String>

    public init(
        footprints: [String: TensorFootprint],
        fusedGroupBytes: [String: Int64] = [:],
        existingLoraBytes: Int64 = 0,
        editedShiftTensors: Set<String> = [],
        editedLayerTensors: Set<String> = [],
        disabledFusedGroups: Set<String> = []
    ) {
        self.footprints = footprints
        self.fusedGroupBytes = fusedGroupBytes
        self.existingLoraBytes = existingLoraBytes
        self.editedShiftTensors = editedShiftTensors
        self.editedLayerTensors = editedLayerTensors
        self.disabledFusedGroups = disabledFusedGroups
    }
}

public struct WeightCopyBudgetPlan: Sendable, Equatable {
    public let editedBytes: Int64
    public let fusedProjectionBytes: Int64
    public let totalExtraBytes: Int64
    public let budgetBytes: Int64?
    public let editedTensors: [String]
    public let fusedGroups: [String]

    public var exceedsBudget: Bool {
        budgetBytes.map { totalExtraBytes > $0 } ?? false
    }

    public init(
        editedBytes: Int64,
        fusedProjectionBytes: Int64,
        budgetBytes: Int64?,
        editedTensors: [String],
        fusedGroups: [String]
    ) {
        self.editedBytes = editedBytes
        self.fusedProjectionBytes = fusedProjectionBytes
        self.totalExtraBytes = editedBytes + fusedProjectionBytes
        self.budgetBytes = budgetBytes
        self.editedTensors = editedTensors
        self.fusedGroups = fusedGroups
    }
}

public enum WeightCopyBudgetPlanner {
    private struct Accumulator {
        var shiftTensors: Set<String>
        var layerTensors: Set<String>
        var brokenGroups: Set<String>
        var loraBytes: Int64
        var loraTensors: Set<String> = []
    }

    public static func plan(
        program: EditProgram,
        state: WeightCopyBudgetState,
        budgetBytes: Int64? = nil
    ) throws(EditValidationError) -> WeightCopyBudgetPlan {
        let acc = accumulate(ops: program.ops, state: state)
        let editedBytes = computeEditedBytes(acc: acc, footprints: state.footprints)
        let activeGroups = state.fusedGroupBytes.keys.filter { !acc.brokenGroups.contains($0) }.sorted()
        let fusedBytes = activeGroups.reduce(Int64(0)) { $0 + (state.fusedGroupBytes[$1] ?? 0) }
        let allEdited = acc.shiftTensors.union(acc.layerTensors).union(acc.loraTensors).sorted()
        let plan = WeightCopyBudgetPlan(
            editedBytes: editedBytes,
            fusedProjectionBytes: fusedBytes,
            budgetBytes: budgetBytes,
            editedTensors: allEdited,
            fusedGroups: activeGroups
        )
        if let budgetBytes, plan.totalExtraBytes > budgetBytes {
            throw .budgetExceeded(required: plan.totalExtraBytes, budget: budgetBytes)
        }
        return plan
    }

    private static func accumulate(ops: [EditOp], state: WeightCopyBudgetState) -> Accumulator {
        var acc = Accumulator(
            shiftTensors: state.editedShiftTensors,
            layerTensors: state.editedLayerTensors,
            brokenGroups: state.disabledFusedGroups,
            loraBytes: state.existingLoraBytes
        )
        for op in ops {
            switch op {
            case .rank1Damp(let module, _, _), .rank1(let module, _, _, _):
                guard let info = state.footprints[module + ".weight"], info.shape.count == 2 else { continue }
                acc.loraBytes += Int64(info.shape[0] + info.shape[1]) * info.computeElementBytes
                acc.loraTensors.insert("\(module).lora_a")
                acc.loraTensors.insert("\(module).lora_b")
                if let group = info.fusedGroupKey { acc.brokenGroups.insert(group) }
            case .shift(let tensor, _, _, _):
                if state.footprints[tensor] != nil { acc.shiftTensors.insert(tensor) }
            case .clip(let tensor, _), .rows(let tensor, _, _):
                if state.footprints[tensor] != nil { acc.layerTensors.insert(tensor) }
            }
        }
        return acc
    }

    private static func computeEditedBytes(acc: Accumulator, footprints: [String: TensorFootprint]) -> Int64 {
        var editedBytes = acc.loraBytes
        for tensor in acc.shiftTensors where !acc.layerTensors.contains(tensor) {
            editedBytes += footprints[tensor]?.byteCount ?? 0
        }
        for tensor in acc.layerTensors {
            guard let info = footprints[tensor] else { continue }
            if let group = info.fusedGroupKey, !acc.brokenGroups.contains(group) { continue }
            editedBytes += info.moduleByteCount
        }
        return editedBytes
    }
}
