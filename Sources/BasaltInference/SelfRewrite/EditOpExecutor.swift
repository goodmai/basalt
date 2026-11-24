import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

enum EditUndoEntry: Sendable {
    case loraModule(path: String)
    case previousLoRA(path: String, loraA: MaterializedArray, loraB: MaterializedArray, scale: Float)
    case replacedLayer(PackedLayer)
    case floatTensor(name: String, original: MaterializedArray)
}

struct EditExecutionResult: Sendable {
    let metrics: [EditOpMetrics]
    let undoEntries: [EditUndoEntry]
    let editedShiftTensors: Set<String>
    let editedLayerTensors: Set<String>
    let loraModules: Set<String>
}

enum EditOpExecutor {
    struct ExecutionContext {
        var protectedIdentities: [String: UInt]
        var loraScales: [String: Float]

        init(protectedIdentities: [String: UInt] = [:], loraScales: [String: Float] = [:]) {
            self.protectedIdentities = protectedIdentities
            self.loraScales = loraScales
        }
    }

    struct SingleOpOutcome {
        let metrics: EditOpMetrics
        let undoEntries: [EditUndoEntry]
        let editedShifts: Set<String>
        let editedLayers: Set<String>
        let loraModules: Set<String>
    }

    static func tensorShapes(in model: Module) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        for view in WeightView.all(in: model) {
            shapes[view.name] = view.shape
        }
        for (path, module) in model.namedModules() where !path.isEmpty {
            if let lora = module as? LoRALayer, let base = lora.reverted() as? Linear {
                let (outputs, inputs) = base.shape
                shapes["\(path).weight"] = [outputs, inputs]
            }
        }
        return shapes
    }

    static func execute(
        program: EditProgram,
        tensors: [String: MLXArray],
        on model: Module,
        context: inout ExecutionContext
    ) throws -> EditExecutionResult {
        let shapes = tensorShapes(in: model)
        var floatValues: [String: [Float]] = [:]
        for (key, array) in tensors {
            eval(array)
            floatValues[key] = array.asType(.float32).asArray(Float.self)
        }
        try EditProgramValidator.validate(program, tensorShapes: shapes, tensorValues: floatValues)

        var metrics: [EditOpMetrics] = []
        var undoEntries: [EditUndoEntry] = []
        var editedShifts = Set<String>()
        var editedLayers = Set<String>()
        var loraModules = Set<String>()
        let savedScales = context.loraScales

        do {
            for (index, op) in program.ops.enumerated() {
                let outcome = try applySingleOp(op, index: index, tensors: tensors, on: model, context: &context)
                metrics.append(outcome.metrics)
                undoEntries.append(contentsOf: outcome.undoEntries)
                editedShifts.formUnion(outcome.editedShifts)
                editedLayers.formUnion(outcome.editedLayers)
                loraModules.formUnion(outcome.loraModules)
            }
            MLXInferenceEngine.settle(model)
            eval(model)
        } catch {
            undo(undoEntries, in: model, loraScales: &context.loraScales)
            context.loraScales = savedScales
            throw error
        }

        return EditExecutionResult(
            metrics: metrics,
            undoEntries: undoEntries,
            editedShiftTensors: editedShifts,
            editedLayerTensors: editedLayers,
            loraModules: loraModules
        )
    }

    static func undo(_ entries: [EditUndoEntry], in model: Module, loraScales: inout [String: Float]) throws {
        guard !entries.isEmpty else { return }
        var failure: (any Error)?
        for entry in entries.reversed() {
            do {
                try undoEntry(entry, in: model, loraScales: &loraScales)
            } catch {
                failure = failure ?? error
            }
        }
        MLXInferenceEngine.settle(model)
        eval(model)
        if let failure { throw failure }
    }

    private static func undoEntry(_ entry: EditUndoEntry, in model: Module, loraScales: inout [String: Float]) throws {
        switch entry {
        case .loraModule(let path):
            let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
            if let lora = modules[path] as? LoRALayer {
                try MLXInferenceEngine.replace(path, with: lora.reverted(), in: model)
            }
            loraScales.removeValue(forKey: path)
        case .previousLoRA(let path, let loraA, let loraB, let scale):
            try restorePreviousLoRA(path: path, loraA: loraA, loraB: loraB, scale: scale, in: model, loraScales: &loraScales)
        case .replacedLayer(let packed):
            try packed.restore(in: model)
            loraScales.removeValue(forKey: packed.path)
        case .floatTensor(let name, let original):
            let restored = original.reshaped(original.shape)
            _ = try model.update(parameters: ModuleParameters.unflattened([(name, restored)]), verify: .none)
            eval(restored)
        }
    }

    private static func restorePreviousLoRA(
        path: String,
        loraA: MaterializedArray,
        loraB: MaterializedArray,
        scale: Float,
        in model: Module,
        loraScales: inout [String: Float]
    ) throws {
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        let baseLinear: Linear? = if let current = modules[path] as? LoRALayer {
            current.reverted() as? Linear
        } else {
            modules[path] as? Linear
        }
        guard let base = baseLinear else { return }
        let restored = LoRALinear.from(linear: base, rank: loraA.dim(1), scale: scale)
        restored.update(parameters: ModuleParameters.unflattened([
            "lora_a": loraA.reshaped(loraA.shape),
            "lora_b": loraB.reshaped(loraB.shape),
        ]))
        eval(restored)
        try MLXInferenceEngine.replace(path, with: restored, in: model)
        loraScales[path] = scale
    }

    static func requireTensor(_ key: String, in tensors: [String: MLXArray]) throws -> MLXArray {
        guard let tensor = tensors[key] else { throw EditValidationError.missingTensor(key: key) }
        return tensor
    }

    static func parentModulePath(of tensorName: String) -> String {
        tensorName.split(separator: ".").dropLast().joined(separator: ".")
    }
}

extension EditOpExecutor {
    struct RankOneSpec {
        let modulePath: String
        let input: MLXArray
        let output: MLXArray
        let scale: Float
        let index: Int
    }

    static func applySingleOp(
        _ op: EditOp,
        index: Int,
        tensors: [String: MLXArray],
        on model: Module,
        context: inout ExecutionContext
    ) throws -> SingleOpOutcome {
        switch op {
        case .rank1Damp(let module, let vectorKey, let alpha):
            let vector = try requireTensor(vectorKey, in: tensors)
            return try applyRank1Damp(modulePath: module, vector: vector, alpha: alpha, index: index, on: model, context: &context)
        case .rank1(let module, let outputKey, let inputKey, let scale):
            let spec = RankOneSpec(
                modulePath: module,
                input: try requireTensor(inputKey, in: tensors),
                output: try requireTensor(outputKey, in: tensors),
                scale: scale,
                index: index
            )
            return try applyRank1(spec, on: model, context: &context)
        case .shift(let tensor, let start, let count, let valueKey):
            let spec = ShiftSpec(tensorName: tensor, start: start, count: count, values: try requireTensor(valueKey, in: tensors), index: index)
            return try applyShift(spec, on: model, context: context)
        case .clip(let tensor, let factor):
            return try applyClip(tensorName: tensor, factor: factor, index: index, on: model, context: &context)
        case .rows(let tensor, let rowIndices, let valuesKey):
            let spec = RowsSpec(tensorName: tensor, rowIndices: rowIndices, replacement: try requireTensor(valuesKey, in: tensors), index: index)
            return try applyRows(spec, on: model, context: &context)
        }
    }

    private static func applyRank1Damp(
        modulePath: String,
        vector: MLXArray,
        alpha: Float,
        index: Int,
        on model: Module,
        context: inout ExecutionContext
    ) throws -> SingleOpOutcome {
        let (baseLinear, existingLoRA) = try resolveLinearOrLoRA(modulePath, in: model)
        let (outputs, inputs) = baseLinear.shape
        let flatU = vector.flattened()
        guard flatU.size == outputs else {
            throw EditValidationError.shapeMismatch(target: modulePath, expected: "[\(outputs)]", actual: "\(vector.shape)")
        }
        let norm = sqrt((flatU.asType(.float32) * flatU.asType(.float32)).sum()).item(Float.self)
        guard norm.isFinite, norm > 0 else {
            throw EditValidationError.nonFiniteValue(target: modulePath, detail: "rank1Damp u has norm \(norm)")
        }
        let uRow = (flatU.asType(.float32) / norm).asType(flatU.dtype).reshaped(1, outputs)
        let baseInputVector = transposeMatVec(uRow: uRow, linear: baseLinear)
        let inputVector: MLXArray
        if let existing = existingLoRA {
            let prevScale = context.loraScales[modulePath] ?? 1.0
            let prev = prevScale * matmul(matmul(uRow.asType(.float32), existing.loraB.T.asType(.float32)), existing.loraA.T.asType(.float32)).squeezed(axis: 0)
            inputVector = baseInputVector.asType(.float32) + prev
        } else {
            inputVector = baseInputVector
        }
        eval(inputVector)
        guard inputVector.size == inputs else {
            throw EditValidationError.shapeMismatch(target: modulePath, expected: "[\(inputs)]", actual: "\(inputVector.shape)")
        }
        let spec = RankOneSpec(modulePath: modulePath, input: inputVector, output: flatU, scale: -alpha, index: index)
        return try installRankOne(spec, baseLinear: baseLinear, existingLoRA: existingLoRA, on: model, context: &context)
    }

    private static func applyRank1(
        _ spec: RankOneSpec,
        on model: Module,
        context: inout ExecutionContext
    ) throws -> SingleOpOutcome {
        let (baseLinear, existingLoRA) = try resolveLinearOrLoRA(spec.modulePath, in: model)
        let (outputs, inputs) = baseLinear.shape
        let flatOut = spec.output.flattened()
        let flatIn = spec.input.flattened()
        guard flatOut.size == outputs else {
            throw EditValidationError.shapeMismatch(target: spec.modulePath, expected: "output [\(outputs)]", actual: "\(spec.output.shape)")
        }
        guard flatIn.size == inputs else {
            throw EditValidationError.shapeMismatch(target: spec.modulePath, expected: "input [\(inputs)]", actual: "\(spec.input.shape)")
        }
        let normalized = RankOneSpec(modulePath: spec.modulePath, input: flatIn, output: flatOut, scale: spec.scale, index: spec.index)
        return try installRankOne(normalized, baseLinear: baseLinear, existingLoRA: existingLoRA, on: model, context: &context)
    }
}

extension EditOpExecutor {
    private static func installRankOne(
        _ spec: RankOneSpec,
        baseLinear: Linear,
        existingLoRA: (loraA: MLXArray, loraB: MLXArray)?,
        on model: Module,
        context: inout ExecutionContext
    ) throws -> SingleOpOutcome {
        let (outputs, inputs) = baseLinear.shape
        let dtype = PEFTLoRAReader.computeType(of: baseLinear)
        let newA = spec.input.reshaped(inputs, 1).asType(dtype)
        let newB = spec.output.reshaped(1, outputs).asType(dtype)
        let lora: LoRALayer
        let effectiveScale: Float
        let undoEntry: EditUndoEntry
        if let existing = existingLoRA {
            let prevScale = context.loraScales[spec.modulePath] ?? 1.0
            undoEntry = .previousLoRA(path: spec.modulePath, loraA: materialize(existing.loraA), loraB: materialize(existing.loraB), scale: prevScale)
            let mergedA = concatenated([existing.loraA.asType(dtype), newA], axis: 1)
            let mergedB = concatenated([(existing.loraB.asType(.float32) * prevScale).asType(dtype), (newB.asType(.float32) * spec.scale).asType(dtype)], axis: 0)
            effectiveScale = 1.0
            lora = LoRALinear.from(linear: baseLinear, rank: mergedA.dim(1), scale: effectiveScale)
            lora.update(parameters: ModuleParameters.unflattened(["lora_a": mergedA, "lora_b": mergedB]))
        } else {
            undoEntry = .loraModule(path: spec.modulePath)
            effectiveScale = spec.scale
            lora = LoRALinear.from(linear: baseLinear, rank: 1, scale: effectiveScale)
            lora.update(parameters: ModuleParameters.unflattened(["lora_a": newA, "lora_b": newB]))
        }
        eval(lora)
        try MLXInferenceEngine.replace(spec.modulePath, with: lora, in: model)
        context.loraScales[spec.modulePath] = effectiveScale
        return SingleOpOutcome(
            metrics: EditOpMetrics(opIndex: spec.index, target: spec.modulePath),
            undoEntries: [undoEntry],
            editedShifts: [],
            editedLayers: [],
            loraModules: [spec.modulePath]
        )
    }

    private static func transposeMatVec(uRow: MLXArray, linear: Linear) -> MLXArray {
        if let quantized = linear as? QuantizedLinear {
            let computeDType = PEFTLoRAReader.computeType(of: quantized)
            return quantizedMM(
                uRow.asType(computeDType),
                quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                transpose: false,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            ).squeezed(axis: 0)
        }
        return matmul(uRow.asType(.float32), linear.weight.asType(.float32)).squeezed(axis: 0).asType(linear.weight.dtype)
    }

    private static func resolveLinearOrLoRA(
        _ path: String,
        in model: Module
    ) throws -> (base: Linear, existingLoRA: (loraA: MLXArray, loraB: MLXArray)?) {
        guard let module = model.namedModules().first(where: { $0.0 == path })?.1 else {
            throw WeightCopyError.invalidOperation("module '\(path)' not found")
        }
        if let lora = module as? LoRALayer, let base = lora.reverted() as? Linear, MLXInferenceEngine.restorable(base) {
            if let reason = MLXInferenceEngine.microscaledLoRARefusal(base, path: path) {
                throw WeightCopyError.invalidOperation(reason)
            }
            let params = Dictionary(lora.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            guard let loraA = params["lora_a"], let loraB = params["lora_b"] else {
                throw WeightCopyError.invalidOperation("module '\(path)' carries a non-LoRA adapter")
            }
            return (base, (loraA, loraB))
        }
        guard let linear = module as? Linear, MLXInferenceEngine.restorable(linear) else {
            throw WeightCopyError.invalidOperation("module '\(path)' is not a plain Linear or QuantizedLinear")
        }
        if let reason = MLXInferenceEngine.microscaledLoRARefusal(linear, path: path) {
            throw WeightCopyError.invalidOperation(reason)
        }
        return (linear, nil)
    }
}
