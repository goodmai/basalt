import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

public struct WeightCopyAccounting: Sendable, Equatable {
    public let editedBytes: Int64
    public let fusedProjectionBytes: Int64
    public let totalExtraBytes: Int64
    public let editedTensors: [String]
    public let fusedGroups: [String]

    public init(
        editedBytes: Int64,
        fusedProjectionBytes: Int64,
        editedTensors: [String],
        fusedGroups: [String]
    ) {
        self.editedBytes = editedBytes
        self.fusedProjectionBytes = fusedProjectionBytes
        self.totalExtraBytes = editedBytes + fusedProjectionBytes
        self.editedTensors = editedTensors
        self.fusedGroups = fusedGroups
    }
}

public struct WeightCopy: ~Copyable {
    private struct UndoFrame {
        let entries: [EditUndoEntry]
        let shifts: Set<String>
        let layers: Set<String>
        let metrics: [EditOpMetrics]
    }

    private var copyModel: (any LanguageModel)?
    public let originalFingerprint: ParameterFingerprint
    public let budgetBytes: ByteCount?
    private let protectedIdentities: [String: UInt]
    private var undoStack: [UndoFrame]
    private var loraScales: [String: Float]
    private var editedShiftTensors: Set<String>
    private var editedLayerTensors: Set<String>

    private init(
        copyModel: any LanguageModel,
        originalFingerprint: ParameterFingerprint,
        budgetBytes: ByteCount?,
        loraScales: [String: Float]
    ) {
        self.copyModel = copyModel
        self.originalFingerprint = originalFingerprint
        self.budgetBytes = budgetBytes
        self.protectedIdentities = originalFingerprint.digests.mapValues(\.arrayIdentity)
        self.undoStack = []
        self.loraScales = loraScales
        self.editedShiftTensors = []
        self.editedLayerTensors = []
    }

    static func validated(
        copy: any LanguageModel,
        fingerprint: ParameterFingerprint,
        budgetBytes: ByteCount?,
        loraScales: [String: Float]
    ) throws -> WeightCopy {
        let instance = WeightCopy(
            copyModel: copy,
            originalFingerprint: fingerprint,
            budgetBytes: budgetBytes,
            loraScales: loraScales
        )
        if let budgetBytes, instance.accounting.totalExtraBytes > budgetBytes.rawValue {
            throw WeightCopyError.budgetExceeded(required: instance.accounting.totalExtraBytes, budget: budgetBytes.rawValue)
        }
        return instance
    }

    public var model: any LanguageModel {
        get throws(WeightCopyError) {
            guard let copyModel else { throw .copyReleased }
            return copyModel
        }
    }

    public var allMetrics: [EditOpMetrics] {
        undoStack.flatMap(\.metrics)
    }

    public func verifyOriginal(_ original: any LanguageModel) throws(WeightCopyError) {
        try originalFingerprint.requireUnchanged(against: original)
    }

    public func logits(for tokens: MLXArray) throws -> MLXArray {
        let copy = try model
        let cache = try copy.newCache(parameters: nil)
        let out = copy(tokens, cache: cache)
        eval(out)
        return out
    }

    public func planBudget(for program: EditProgram) throws -> WeightCopyBudgetPlan {
        let copy = try model
        let state = Self.budgetState(
            of: copy,
            editedShiftTensors: editedShiftTensors,
            editedLayerTensors: editedLayerTensors
        )
        return try WeightCopyBudgetPlanner.plan(program: program, state: state, budgetBytes: budgetBytes?.rawValue)
    }

    public func measureRoundTrip(tensor: String) throws -> QuantizationRoundTripMetrics {
        try EditOpExecutor.measureQuantizationRoundTrip(tensorName: tensor, on: try model)
    }

    public var accounting: WeightCopyAccounting {
        guard let copyModel else {
            return WeightCopyAccounting(editedBytes: 0, fusedProjectionBytes: 0, editedTensors: [], fusedGroups: [])
        }
        return Self.measureAccounting(
            of: copyModel,
            editedShiftTensors: editedShiftTensors,
            editedLayerTensors: editedLayerTensors
        )
    }

    @discardableResult
    public mutating func apply(
        _ program: EditProgram,
        tensors: [String: MLXArray] = [:],
        verifying original: any LanguageModel
    ) throws -> [EditOpMetrics] {
        let copy = try model
        _ = try planBudget(for: program)
        var context = EditOpExecutor.ExecutionContext(protectedIdentities: protectedIdentities, loraScales: loraScales)
        let result = try EditOpExecutor.execute(program: program, tensors: tensors, on: copy, context: &context)
        try originalFingerprint.requireUnchanged(against: original)
        loraScales = context.loraScales
        let frame = UndoFrame(
            entries: result.undoEntries,
            shifts: result.editedShiftTensors,
            layers: result.editedLayerTensors,
            metrics: result.metrics
        )
        undoStack.append(frame)
        editedShiftTensors.formUnion(result.editedShiftTensors)
        editedLayerTensors.formUnion(result.editedLayerTensors)
        return result.metrics
    }

    @discardableResult
    public mutating func apply(
        _ adapter: EditAdapter,
        verifying original: any LanguageModel
    ) throws -> [EditOpMetrics] {
        var raw: [String: MLXArray] = [:]
        for (key, value) in adapter.tensors {
            raw[key] = value.reshaped(value.shape)
        }
        return try apply(adapter.program, tensors: raw, verifying: original)
    }

    public mutating func undoLast(verifying original: any LanguageModel) throws {
        let copy = try model
        guard let frame = undoStack.popLast() else { return }
        try EditOpExecutor.undo(frame.entries, in: copy, loraScales: &loraScales)
        rebuildEditedTracking()
        try originalFingerprint.requireUnchanged(against: original)
    }

    public mutating func undoAll(verifying original: any LanguageModel) throws {
        let copy = try model
        while let frame = undoStack.popLast() {
            try EditOpExecutor.undo(frame.entries, in: copy, loraScales: &loraScales)
        }
        editedShiftTensors.removeAll()
        editedLayerTensors.removeAll()
        loraScales = ModelSkeletonBuilder.extractLoRAScales(from: copy)
        try originalFingerprint.requireUnchanged(against: original)
    }

    public consuming func release(verifying original: (any LanguageModel)? = nil) throws(WeightCopyError) {
        var failure: WeightCopyError?
        if let original {
            do {
                try originalFingerprint.requireUnchanged(against: original)
            } catch {
                failure = error
            }
        }
        copyModel?.invalidateCompiledTraces()
        copyModel = nil
        undoStack.removeAll()
        loraScales.removeAll()
        editedShiftTensors.removeAll()
        editedLayerTensors.removeAll()
        Memory.clearCache()
        if let failure {
            throw failure
        }
    }
}

extension WeightCopy {
    public static func make(
        from original: any LanguageModel,
        configuration: ModelConfiguration? = nil,
        budgetBytes: ByteCount? = nil,
        makeSkeleton: (() throws -> any LanguageModel)? = nil
    ) async throws -> WeightCopy {
        eval(original)
        let fingerprint = ParameterFingerprint.capture(of: original)
        let scales = ModelSkeletonBuilder.extractLoRAScales(from: original)
        let copy = try await ModelSkeletonBuilder.buildCopy(from: original, configuration: configuration, makeSkeleton: makeSkeleton)
        try fingerprint.requireUnchanged(against: original)
        return try validated(copy: copy, fingerprint: fingerprint, budgetBytes: budgetBytes, loraScales: scales)
    }

    public static func makeSync(
        from original: any LanguageModel,
        budgetBytes: ByteCount? = nil,
        makeSkeleton: (() throws -> any LanguageModel)? = nil
    ) throws -> WeightCopy {
        eval(original)
        let fingerprint = ParameterFingerprint.capture(of: original)
        let scales = ModelSkeletonBuilder.extractLoRAScales(from: original)
        let copy = try ModelSkeletonBuilder.buildCopySync(from: original, makeSkeleton: makeSkeleton)
        try fingerprint.requireUnchanged(against: original)
        return try validated(copy: copy, fingerprint: fingerprint, budgetBytes: budgetBytes, loraScales: scales)
    }

    public static func make(
        from container: ModelContainer,
        budgetBytes: ByteCount? = nil
    ) async throws -> WeightCopy {
        let config = await container.configuration
        let diskSkeleton: (any LanguageModel)? = if let dir = ModelSkeletonBuilder.modelDirectory(from: config) {
            try await ModelSkeletonBuilder.buildFromDiskConfig(directory: dir)
        } else {
            nil
        }
        let built = try await container.performEvaluating(nonSendable: diskSkeleton) { context, prebuilt in
            eval(context.model)
            let fingerprint = ParameterFingerprint.capture(of: context.model)
            let scales = ModelSkeletonBuilder.extractLoRAScales(from: context.model)
            let factory: (() throws -> any LanguageModel)? = prebuilt.map { skeleton in { skeleton } }
            let copy = try ModelSkeletonBuilder.buildCopySync(from: context.model, makeSkeleton: factory)
            try fingerprint.requireUnchanged(against: context.model)
            return (copy: copy, fingerprint: fingerprint, scales: scales)
        }
        return try validated(copy: built.copy, fingerprint: built.fingerprint, budgetBytes: budgetBytes, loraScales: built.scales)
    }

    public func verifyOriginal(in container: ModelContainer) async throws(WeightCopyError) {
        let fingerprint = originalFingerprint
        let mismatches = await container.performEvaluating(nonSendable: ()) { context, _ in
            fingerprint.verify(against: context.model)
        }
        if !mismatches.isEmpty {
            throw .originalFingerprintMismatch(mismatches)
        }
    }
}

extension WeightCopy {
    private mutating func rebuildEditedTracking() {
        var shifts = Set<String>()
        var layers = Set<String>()
        for frame in undoStack {
            shifts.formUnion(frame.shifts)
            layers.formUnion(frame.layers)
        }
        editedShiftTensors = shifts
        editedLayerTensors = layers
    }

    static func measureAccounting(
        of copy: Module,
        editedShiftTensors: Set<String>,
        editedLayerTensors: Set<String>
    ) -> WeightCopyAccounting {
        let (fusedGroups, memberToGroup) = ModelSkeletonBuilder.fusableGroupInfo(in: copy)
        let params = Dictionary(copy.parameters().flattened(), uniquingKeysWith: { first, _ in first })
        let modules = Dictionary(copy.namedModules(), uniquingKeysWith: { first, _ in first })

        var editedBytes: Int64 = 0
        var editedNames = Set<String>()

        for (path, module) in modules where module is LoRALayer {
            for (key, array) in module.parameters().flattened() where key == "lora_a" || key == "lora_b" {
                editedBytes += Int64(array.nbytes)
                editedNames.insert("\(path).\(key)")
            }
        }

        for tensor in editedShiftTensors where !editedLayerTensors.contains(tensor) {
            if let array = params[tensor] {
                editedBytes += Int64(array.nbytes)
                editedNames.insert(tensor)
            }
        }

        for tensor in editedLayerTensors {
            let modulePath = EditOpExecutor.parentModulePath(of: tensor)
            editedNames.insert(tensor)
            if let group = memberToGroup[modulePath], fusedGroups[group] != nil { continue }
            if let module = modules[modulePath] {
                editedBytes += Int64(TensorPacker.bytes(module))
            }
        }

        let sortedGroups = fusedGroups.keys.sorted()
        let fusedBytes = sortedGroups.reduce(Int64(0)) { $0 + (fusedGroups[$1] ?? 0) }
        return WeightCopyAccounting(
            editedBytes: editedBytes,
            fusedProjectionBytes: fusedBytes,
            editedTensors: editedNames.sorted(),
            fusedGroups: sortedGroups
        )
    }

    static func budgetState(
        of copy: Module,
        editedShiftTensors: Set<String>,
        editedLayerTensors: Set<String>
    ) -> WeightCopyBudgetState {
        let (fusedGroups, memberToGroup) = ModelSkeletonBuilder.fusableGroupInfo(in: copy)
        let modules = Dictionary(copy.namedModules(), uniquingKeysWith: { first, _ in first })
        var footprints: [String: TensorFootprint] = [:]
        for view in WeightView.all(in: copy) {
            let modulePath = EditOpExecutor.parentModulePath(of: view.name)
            let module = modules[modulePath]
            let moduleBytes = module.map { Int64(TensorPacker.bytes($0)) } ?? Int64(view.array.nbytes)
            let computeBytes: Int64 = if let linear = module as? Linear {
                Int64(EditOpExecutor.computeType(of: linear).size)
            } else {
                Int64(view.array.itemSize)
            }
            footprints[view.name] = TensorFootprint(
                name: view.name,
                shape: view.shape,
                byteCount: Int64(view.array.nbytes),
                moduleByteCount: moduleBytes,
                computeElementBytes: computeBytes,
                fusedGroupKey: memberToGroup[modulePath]
            )
        }
        var existingLoraBytes: Int64 = 0
        for (path, module) in modules where module is LoRALayer {
            for (key, array) in module.parameters().flattened() where key == "lora_a" || key == "lora_b" {
                existingLoraBytes += Int64(array.nbytes)
            }
            if !path.isEmpty, let lora = module as? LoRALayer, let base = lora.reverted() as? Linear {
                let (outputs, inputs) = base.shape
                let weightName = "\(path).weight"
                footprints[weightName] = TensorFootprint(
                    name: weightName,
                    shape: [outputs, inputs],
                    byteCount: Int64(base.weight.nbytes),
                    moduleByteCount: Int64(TensorPacker.bytes(base)),
                    computeElementBytes: Int64(EditOpExecutor.computeType(of: base).size),
                    fusedGroupKey: memberToGroup[path]
                )
            }
        }
        return WeightCopyBudgetState(
            footprints: footprints,
            fusedGroupBytes: fusedGroups,
            existingLoraBytes: existingLoraBytes,
            editedShiftTensors: editedShiftTensors,
            editedLayerTensors: editedLayerTensors
        )
    }
}
