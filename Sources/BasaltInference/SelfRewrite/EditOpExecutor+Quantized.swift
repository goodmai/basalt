import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

extension EditOpExecutor {
    static let nvfp4ScaleDivisor: Float = 448.0 * 6.0

    static func supportsRankOne(_ linear: Linear) -> Bool {
        let kind = ObjectIdentifier(type(of: linear))
        return kind == ObjectIdentifier(Linear.self)
            || kind == ObjectIdentifier(QuantizedLinear.self)
            || kind == ObjectIdentifier(SmallMQuantizedLinear.self)
            || kind == ObjectIdentifier(MACaQQuantizedLinear.self)
            || kind == ObjectIdentifier(NF4Linear.self)
    }

    static func computeType(of linear: Linear) -> DType {
        if let nf4 = linear as? NF4Linear {
            return nf4.scales.dtype.isFloatingPoint ? nf4.scales.dtype : .bfloat16
        }
        if let mac = linear as? MACaQQuantizedLinear {
            let floats: Set<DType> = [.float16, .bfloat16, .float32]
            let candidates = [mac.biases?.dtype, mac.scales.dtype, mac.preScale.dtype].compactMap { $0 }
            return candidates.first(where: floats.contains) ?? .bfloat16
        }
        return PEFTLoRAReader.computeType(of: linear)
    }

    static func wrapRankOne(
        base: Linear,
        loraA: MLXArray,
        loraB: MLXArray,
        scale: Float
    ) -> LoRALayer {
        if MLXInferenceEngine.restorable(base) {
            let lora = LoRALinear.from(linear: base, rank: loraA.dim(1), scale: scale)
            lora.update(parameters: ModuleParameters.unflattened(["lora_a": loraA, "lora_b": loraB]))
            return lora
        }
        return EditRankOneLinear(base: base, loraA: loraA, loraB: loraB, scale: scale)
    }

    static func exactTransposeMatVec(uRow: MLXArray, linear: Linear) -> MLXArray {
        if let mac = linear as? MACaQQuantizedLinear {
            return transposeMatVecMACaQ(uRow: uRow, mac: mac)
        }
        if let nf4 = linear as? NF4Linear {
            let dense = TensorPacker.dense(nf4).asType(.float32)
            return matmul(uRow.asType(.float32), dense).squeezed(axis: 0).asType(computeType(of: nf4))
        }
        if let quantized = linear as? QuantizedLinear {
            let computeDType = computeType(of: quantized)
            var projected = quantizedMM(
                uRow.asType(computeDType),
                quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                transpose: false,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
            if let globalScale = quantized.globalScale {
                let factor = globalScale.asType(.float32) / Self.nvfp4ScaleDivisor
                projected = (projected.asType(.float32) * factor).asType(projected.dtype)
            }
            return projected.squeezed(axis: 0)
        }
        return matmul(uRow.asType(.float32), linear.weight.asType(.float32))
            .squeezed(axis: 0)
            .asType(linear.weight.dtype)
    }

    private static func transposeMatVecMACaQ(uRow: MLXArray, mac: MACaQQuantizedLinear) -> MLXArray {
        var scaledU = uRow.asType(.float32)
        if let rowScale = mac.rowScale {
            scaledU = scaledU * rowScale.asType(.float32).expandedDimensions(axis: 0)
        }
        let computeDType = computeType(of: mac)
        let projected = quantizedMM(
            scaledU.asType(computeDType),
            mac.weight,
            scales: mac.scales,
            biases: mac.biases,
            transpose: false,
            groupSize: mac.groupSize,
            bits: mac.bits,
            mode: mac.mode
        ).asType(.float32)
        let rotated = MACaQMath.rotateBlocks(projected, blockSize: mac.hadamardBlock)
        let pre = mac.preScale.asType(.float32).expandedDimensions(axis: 0)
        return (rotated * pre).squeezed(axis: 0).asType(computeDType)
    }
}

extension EditOpExecutor {
    static func repackLinear(_ layer: Linear, editedFloat32: MLXArray) -> Linear {
        if let mac = layer as? MACaQQuantizedLinear {
            return repackMACaQ(mac, editedFloat32: editedFloat32)
        }
        if let nf4 = layer as? NF4Linear {
            let target = editedFloat32.asType(computeType(of: nf4))
            let next = NF4Linear(dense: target, bias: nf4.bias, groupSize: nf4.groupSize)
            next.train(nf4.training)
            return next
        }
        if let smallM = layer as? SmallMQuantizedLinear {
            let target = editedFloat32.asType(computeType(of: smallM))
            let quantized = QuantizedLinear(
                weight: target, bias: smallM.bias, groupSize: smallM.groupSize,
                bits: smallM.bits, mode: smallM.mode, globalScale: smallM.globalScale
            )
            let next = SmallMQuantizedLinear(replacing: quantized)
            next.train(smallM.training)
            return next
        }
        if let quantized = layer as? QuantizedLinear,
           ObjectIdentifier(type(of: quantized)) == ObjectIdentifier(QuantizedLinear.self) {
            let target = editedFloat32.asType(computeType(of: quantized))
            let next = QuantizedLinear(
                weight: target, bias: quantized.bias, groupSize: quantized.groupSize,
                bits: quantized.bits, mode: quantized.mode, globalScale: quantized.globalScale
            )
            next.train(quantized.training)
            next.freeze()
            return next
        }
        let next = Linear(weight: editedFloat32.asType(layer.weight.dtype), bias: layer.bias)
        next.train(layer.training)
        return next
    }

    static func makeMACaQ(
        from mac: MACaQQuantizedLinear,
        packed: MACaQPackedWeights,
        bias: MLXArray?
    ) -> MACaQQuantizedLinear {
        var spec = MACaQLayerSpec(bits: mac.bits, groupSize: mac.groupSize, mode: mac.mode, hadamardBlock: mac.hadamardBlock)
        spec.scaleDType = "\(packed.scales.dtype)"
        spec.hasBias = bias != nil
        let rebuilt = MACaQQuantizedLinear(
            packed: packed,
            bias: bias,
            preScale: mac.preScale,
            spec: spec,
            rowScale: mac.rowScale,
            activationMode: mac.activationMode
        )
        rebuilt.useSmallMKernel = mac.useSmallMKernel
        rebuilt.useQ3Kernel = mac.useQ3Kernel
        rebuilt.train(mac.training)
        return rebuilt
    }

    static func measureQuantizationRoundTrip(
        tensorName: String,
        on model: Module
    ) throws -> QuantizationRoundTripMetrics {
        guard tensorName.hasSuffix(".weight") else {
            throw WeightCopyError.invalidOperation("'\(tensorName)' must name a .weight tensor of a linear layer")
        }
        let path = String(tensorName.dropLast(".weight".count))
        guard let module = model.namedModules().first(where: { $0.0 == path })?.1 else {
            throw WeightCopyError.invalidOperation("module '\(path)' not found")
        }
        let base: Linear
        if let lora = module as? LoRALayer, let reverted = lora.reverted() as? Linear, supportsRankOne(reverted) {
            base = reverted
        } else if let linear = module as? Linear, supportsRankOne(linear) {
            base = linear
        } else {
            throw WeightCopyError.invalidOperation("'\(tensorName)' is not a supported linear layer")
        }
        let dense = TensorPacker.dense(base).asType(.float32)
        let repacked = repackLinear(base, editedFloat32: dense)
        let error = TensorPacker.measure(dense, repacked)
        return QuantizationRoundTripMetrics(
            tensor: tensorName,
            format: TensorPacker.describe(base),
            rmse: error.rmse,
            maxAbsError: error.maxAbsError,
            relativeError: error.relativeError
        )
    }

    private static func repackMACaQ(_ mac: MACaQQuantizedLinear, editedFloat32: MLXArray) -> MACaQQuantizedLinear {
        let pre = mac.preScale.asType(.float32).expandedDimensions(axis: 0)
        let safePre = which(abs(pre) .> Float(1e-8), pre, MLXArray(Float(1)))
        var rotated = MACaQMath.rotateBlocks(editedFloat32.asType(.float32) / safePre, blockSize: mac.hadamardBlock)
        if let rowScale = mac.rowScale {
            let rs = rowScale.asType(.float32).expandedDimensions(axis: -1)
            let safeRs = which(abs(rs) .> Float(1e-8), rs, MLXArray(Float(1)))
            rotated = rotated / safeRs
        }
        let target = rotated.asType(computeType(of: mac))
        let (qWeight, qScales, qBiases) = MLX.quantized(
            target, groupSize: mac.groupSize, bits: mac.bits, mode: mac.mode
        )
        let packed = MACaQPackedWeights(weight: qWeight, scales: qScales, biases: qBiases)
        return makeMACaQ(from: mac, packed: packed, bias: mac.bias)
    }
}
