import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

extension EditOpExecutor {
    struct ShiftSpec {
        let tensorName: String
        let start: Int
        let count: Int
        let values: MLXArray
        let index: Int
    }

    static func applyShift(_ spec: ShiftSpec, on model: Module, context: ExecutionContext) throws -> SingleOpOutcome {
        guard let existing = model.parameters().flattened().first(where: { $0.0 == spec.tensorName })?.1 else {
            throw WeightCopyError.invalidOperation("tensor '\(spec.tensorName)' not found")
        }
        let ownerPath = parentModulePath(of: spec.tensorName)
        let leafKey = String(spec.tensorName.split(separator: ".").last ?? "")
        let ownerModule = model.namedModules().first(where: { $0.0 == ownerPath })?.1
        guard MLXInferenceEngine.writableDeltaTypes.contains(existing.dtype) else {
            throw WeightCopyError.invalidOperation("shift requires a float tensor; '\(spec.tensorName)' is \(existing.dtype)")
        }
        if let ownerModule, ownerModule is Quantized, leafKey != "bias" {
            throw WeightCopyError.invalidOperation("shift cannot modify quantized parameter '\(spec.tensorName)'")
        }
        let updated = try buildShiftedArray(existing: existing, spec: spec)
        if let linear = ownerModule as? Linear, MLXInferenceEngine.restorable(linear) {
            let backup = PackedLayer(path: ownerPath, layer: linear)
            let replacement = try replacedLinear(linear, leafKey: leafKey, updated: updated, tensorName: spec.tensorName)
            try MLXInferenceEngine.replace(ownerPath, with: replacement, in: model)
            return SingleOpOutcome(
                metrics: EditOpMetrics(opIndex: spec.index, target: spec.tensorName),
                undoEntries: [.replacedLayer(backup)],
                editedShifts: [spec.tensorName],
                editedLayers: [],
                loraModules: []
            )
        }
        let identity = UInt(bitPattern: ObjectIdentifier(existing))
        if let protected = context.protectedIdentities[spec.tensorName], protected == identity {
            throw WeightCopyError.sharedHandleMutationBlocked(tensor: spec.tensorName)
        }
        let backup = materialize(existing)
        try model.update(parameters: ModuleParameters.unflattened([(spec.tensorName, updated)]), verify: .none)
        eval(model)
        return SingleOpOutcome(
            metrics: EditOpMetrics(opIndex: spec.index, target: spec.tensorName),
            undoEntries: [.floatTensor(name: spec.tensorName, original: backup)],
            editedShifts: [spec.tensorName],
            editedLayers: [],
            loraModules: []
        )
    }

    private static func buildShiftedArray(existing: MLXArray, spec: ShiftSpec) throws -> MLXArray {
        let flatValues = spec.values.asType(.float32).flattened()
        guard flatValues.size == spec.count || flatValues.size == 1 else {
            throw EditValidationError.shapeMismatch(target: spec.tensorName, expected: "\(spec.count) or 1 values", actual: "\(flatValues.size)")
        }
        guard spec.start >= 0, spec.count > 0, spec.start + spec.count <= existing.size else {
            throw EditValidationError.invalidSlice(tensor: spec.tensorName, start: spec.start, count: spec.count, limit: existing.size)
        }
        let origNative = existing.flattened()
        let delta = flatValues.size == 1 ? MLXArray.full([spec.count], values: flatValues[0]) : flatValues
        let end = spec.start + spec.count
        let shiftedMid = (origNative[spec.start ..< end].asType(.float32) + delta).asType(existing.dtype)
        var segments: [MLXArray] = []
        if spec.start > 0 { segments.append(origNative[0 ..< spec.start]) }
        segments.append(shiftedMid)
        if end < existing.size { segments.append(origNative[end ..< existing.size]) }
        let updated = (segments.count == 1 ? segments[0] : concatenated(segments, axis: 0)).reshaped(existing.shape)
        eval(updated)
        return updated
    }

    private static func replacedLinear(_ linear: Linear, leafKey: String, updated: MLXArray, tensorName: String) throws -> Linear {
        if let quantized = linear as? QuantizedLinear {
            guard leafKey == "bias" else {
                throw WeightCopyError.invalidOperation("shift cannot modify quantized parameter '\(tensorName)'")
            }
            let next = QuantizedLinear(
                weight: quantized.weight, bias: updated, scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale
            )
            next.train(quantized.training)
            next.freeze()
            return next
        }
        let next = leafKey == "bias" ? Linear(weight: linear.weight, bias: updated) : Linear(weight: updated, bias: linear.bias)
        next.train(linear.training)
        return next
    }
}

extension EditOpExecutor {
    struct RowsSpec {
        let tensorName: String
        let rowIndices: [Int]
        let replacement: MLXArray
        let index: Int
    }

    struct RepackRequest {
        let modulePath: String
        let tensorName: String
        let layer: Linear
        let editedFloat32: MLXArray
        let undoPrefix: [EditUndoEntry]
        let index: Int
    }

    static func medianAbs(_ array: MLXArray) throws -> MLXArray {
        let absFlat = sorted(abs(array.asType(.float32)).flattened())
        let count = absFlat.size
        guard count > 0 else {
            throw WeightCopyError.invalidOperation("cannot compute median of empty tensor")
        }
        return count % 2 == 1 ? absFlat[count / 2] : (absFlat[count / 2 - 1] + absFlat[count / 2]) * Float(0.5)
    }

    static func applyClip(
        tensorName: String,
        factor: Float,
        index: Int,
        on model: Module,
        context: inout ExecutionContext
    ) throws -> SingleOpOutcome {
        let resolved = try resolveEditableLayer(tensorName, in: model, context: &context)
        let median = try medianAbs(resolved.denseFloat32)
        let bound = maximum(median * factor, MLXArray(Float(0)))
        let clipped = MLX.clip(resolved.denseFloat32, min: -bound, max: bound)
        let req = RepackRequest(
            modulePath: resolved.path, tensorName: tensorName, layer: resolved.baseLayer,
            editedFloat32: clipped, undoPrefix: resolved.undoPrefix, index: index
        )
        return try repackEditedLayer(req, on: model)
    }

    static func applyRows(_ spec: RowsSpec, on model: Module, context: inout ExecutionContext) throws -> SingleOpOutcome {
        let resolved = try resolveEditableLayer(spec.tensorName, in: model, context: &context)
        let cols = resolved.denseFloat32.dim(1)
        guard spec.replacement.size == spec.rowIndices.count * cols else {
            throw EditValidationError.shapeMismatch(
                target: spec.tensorName, expected: "[\(spec.rowIndices.count), \(cols)]", actual: "\(spec.replacement.shape)"
            )
        }
        let shaped = spec.replacement.reshaped(spec.rowIndices.count, cols).asType(.float32)
        let edited = contiguous(resolved.denseFloat32) + MLXArray(Float(0))
        edited[MLXArray(spec.rowIndices.map(Int32.init))] = shaped
        let req = RepackRequest(
            modulePath: resolved.path, tensorName: spec.tensorName, layer: resolved.baseLayer,
            editedFloat32: edited, undoPrefix: resolved.undoPrefix, index: spec.index
        )
        return try repackEditedLayer(req, on: model)
    }

    private static func resolveEditableLayer(
        _ tensorName: String,
        in model: Module,
        context: inout ExecutionContext
    ) throws -> (path: String, baseLayer: Linear, denseFloat32: MLXArray, undoPrefix: [EditUndoEntry]) {
        guard tensorName.hasSuffix(".weight") else {
            throw WeightCopyError.invalidOperation("'\(tensorName)' must name a .weight tensor of a linear layer")
        }
        let path = String(tensorName.dropLast(".weight".count))
        guard let module = model.namedModules().first(where: { $0.0 == path })?.1 else {
            throw WeightCopyError.invalidOperation("module '\(path)' not found")
        }
        if let lora = module as? LoRALayer, let base = lora.reverted() as? Linear {
            let params = Dictionary(lora.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            let scale = context.loraScales[path] ?? 1.0
            let dense: MLXArray
            let backup: [EditUndoEntry]
            if let a = params["lora_a"], let b = params["lora_b"] {
                dense = TensorPacker.dense(base).asType(.float32) + scale * matmul(b.T.asType(.float32), a.T.asType(.float32))
                backup = [.previousLoRA(path: path, loraA: materialize(a), loraB: materialize(b), scale: scale), .replacedLayer(PackedLayer(path: path, layer: base))]
            } else {
                guard let fused = lora.fused() as? Linear else {
                    throw WeightCopyError.invalidOperation("cannot fuse adapter on '\(path)'")
                }
                dense = TensorPacker.dense(fused).asType(.float32)
                backup = [.replacedLayer(PackedLayer(path: path, layer: base))]
            }
            context.loraScales.removeValue(forKey: path)
            return (path, base, dense, backup)
        }
        guard let linear = module as? Linear,
              type(of: linear) == Linear.self || type(of: linear) == QuantizedLinear.self || type(of: linear) == NF4Linear.self
        else {
            throw WeightCopyError.invalidOperation("'\(tensorName)' is not a supported Linear, QuantizedLinear, or NF4Linear layer")
        }
        return (path, linear, TensorPacker.dense(linear).asType(.float32), [.replacedLayer(PackedLayer(path: path, layer: linear))])
    }

    private static func repackEditedLayer(_ req: RepackRequest, on model: Module) throws -> SingleOpOutcome {
        let repacked: Linear
        if let nf4 = req.layer as? NF4Linear {
            let target = req.editedFloat32.asType(nf4.scales.dtype.isFloatingPoint ? nf4.scales.dtype : .float32)
            repacked = NF4Linear(dense: target, bias: nf4.bias, groupSize: nf4.groupSize)
        } else if let quantized = req.layer as? QuantizedLinear,
                  ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedLinear.self) {
            let computeDType = PEFTLoRAReader.computeType(of: quantized)
            let target = req.editedFloat32.asType(computeDType)
            let next = QuantizedLinear(
                weight: target, bias: quantized.bias, groupSize: quantized.groupSize,
                bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale
            )
            next.train(quantized.training)
            next.freeze()
            repacked = next
        } else {
            let next = Linear(weight: req.editedFloat32.asType(req.layer.weight.dtype), bias: req.layer.bias)
            next.train(req.layer.training)
            repacked = next
        }
        let error = TensorPacker.measure(req.editedFloat32, repacked)
        try MLXInferenceEngine.replace(req.modulePath, with: repacked, in: model)
        return SingleOpOutcome(
            metrics: EditOpMetrics(opIndex: req.index, target: req.tensorName, rmse: error.rmse, maxAbsError: error.maxAbsError, relativeError: error.relativeError),
            undoEntries: req.undoPrefix,
            editedShifts: [],
            editedLayers: [req.tensorName],
            loraModules: []
        )
    }
}
