import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class PVPrologueDenseLinear: Linear {
    let rowScale: MLXArray?
    let prologue: (MLXArray) -> MLXArray

    init(denseRot: MLXArray, bias: MLXArray?, rowScale: MLXArray? = nil, prologue: @escaping (MLXArray) -> MLXArray) {
        self.rowScale = rowScale
        self.prologue = prologue
        super.init(weight: denseRot.asType(.float32), bias: bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let transformed = prologue(x.asType(.float32))
        var y = matmul(transformed, weight.T)
        if let rowScale { y = y * rowScale.asType(.float32) }
        var casted = y.asType(x.dtype)
        if let bias { casted = casted + bias.asType(x.dtype) }
        return casted
    }
}

final class PVMACaQDenseLinear: Linear {
    let preScale: MLXArray
    let hadamardBlock: Int
    let rowScale: MLXArray?

    init(denseRot: MLXArray, bias: MLXArray?, preScale: MLXArray, hadamardBlock: Int, rowScale: MLXArray? = nil) {
        self.preScale = preScale
        self.hadamardBlock = hadamardBlock
        self.rowScale = rowScale
        super.init(weight: denseRot.asType(.float32), bias: bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rotated = MACaQFusedKernels.differentiableBlockHadamard(
            x.asType(.float32), scaleVector: preScale, blockSize: hadamardBlock)
        var raw = matmul(rotated, weight.T)
        if let rowScale { raw = raw * rowScale.asType(.float32) }
        var y = raw.asType(x.dtype)
        if let bias { y = y + bias.asType(x.dtype) }
        return y
    }
}

final class PVMACaQDenseSwitchLinear: SwitchLinear {
    let preScale: MLXArray
    let hadamardBlock: Int

    init(denseRot: MLXArray, bias: MLXArray?, preScale: MLXArray, hadamardBlock: Int) {
        self.preScale = preScale
        self.hadamardBlock = hadamardBlock
        super.init(
            inputDims: denseRot.dim(-1),
            outputDims: denseRot.dim(-2),
            numExperts: denseRot.dim(0),
            weight: denseRot.asType(.float32),
            bias: bias)
    }

    override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let rotated = MACaQFusedKernels.differentiableSwitchHadamard(
            x.asType(.float32), indices: indices, preScale: preScale, blockSize: hadamardBlock)
        let params = parameters()
        guard case .value(let w)? = params["weight"] else { return x }
        var out = MLX.gatherMM(
            rotated, w.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sortedIndices
        ).asType(x.dtype)
        if case .value(let b)? = params["bias"] {
            out = out + expandedDimensions(b[indices], axis: -2).asType(x.dtype)
        }
        return out
    }
}

package enum PVMACaQGrid {
    package static func unpackCodes(_ packed: MLXArray, spec: PVCodeGrid.Spec) -> MLXArray {
        guard spec.encoding == .affine(.mxfp4) else {
            return PVCodeGrid.unpackCodes(packed, bits: spec.bits)
        }
        let nibbles = PVCodeGrid.unpackCodes(packed, bits: 4).asType(.int32)
        let mag = nibbles & Int32(7)
        let neg = ((nibbles >> Int32(3)) & Int32(1)) .== Int32(1)
        return which(
            mag .== Int32(0),
            MLXArray(Int32(7)),
            which(neg, Int32(7) - mag, Int32(7) + mag)
        ).asType(.uint8)
    }

    package static func packCodes(_ codes: MLXArray, spec: PVCodeGrid.Spec) -> MLXArray {
        guard spec.encoding == .affine(.mxfp4) else {
            return PVCodeGrid.packCodes(codes, bits: spec.bits)
        }
        let idx = codes.asType(.int32)
        let signBits = MLXArray(MACaQMath.mxfp4SignLUT).take(idx) << UInt32(3)
        let magBits = MLXArray(MACaQMath.mxfp4MagLUT).take(idx)
        return PVCodeGrid.packCodes((signBits | magBits).asType(.uint8), bits: 4)
    }

    package static func discreteStep(
        codes: MLXArray,
        scales: MLXArray,
        grad: MLXArray,
        spec: PVCodeGrid.Spec,
        fraction: Float
    ) -> (codes: MLXArray, updated: Int) {
        guard spec.encoding == .affine(.mxfp4) else {
            return PVCodeGrid.discreteStep(
                codes: codes, scales: scales, grad: grad, spec: spec, fraction: fraction)
        }
        let shape = codes.shape
        let leading = Array(shape.dropLast())
        let cols = shape.last ?? 1
        let groups = cols / MACaQMath.mxfp4BlockSize
        let qInt = codes.asType(.int32).reshaped(leading + [groups, MACaQMath.mxfp4BlockSize])
        let g = grad.asType(.float32).reshaped(leading + [groups, MACaQMath.mxfp4BlockSize])
        let s = pow(MLXArray(Float(2.0)), scales.asType(.float32) - Float(127.0)).expandedDimensions(axis: -1)
        let gs = g * s
        let stepDir = which(
            gs .> Float(0), MLXArray(Int32(-1)),
            which(gs .< Float(0), MLXArray(Int32(1)), MLXArray(Int32(0))))
        let candInt = clip(qInt + stepDir, min: 0, max: Int32(14))
        let table = MLXArray(MACaQMath.mxfp4SignedLevels)
        let dw = (table.take(candInt) - table.take(qInt)) * s
        let gain = maximum(Float(0), -(g * dw)).reshaped(-1)
        let total = gain.size
        guard total > 0 else { return (codes, 0) }
        let threshold: MLXArray
        if total == 1 {
            threshold = gain[0]
        } else {
            let k = max(1, min(total - 1, Int(Float(total) * fraction)))
            threshold = partitioned(gain, kth: total - k)[total - k]
        }
        let mask = (gain .>= threshold) .&& (gain .> Float(1e-11))
        let updated = which(mask, candInt.asType(.uint8).reshaped(-1), codes.reshaped(-1)).reshaped(shape)
        let count = mask.asType(.int32).sum()
        eval(updated, count)
        return (updated, Int(count.item(Int32.self)))
    }
}

extension PVTuner {
    static func macaqTarget(
        path: String, module: Module, maxSwitchExperts: Int
    ) -> PVQuantTarget? {
        if let mac = module as? MACaQQuantizedLinear {
            let valid = ([2, 3, 4].contains(mac.bits) && mac.mode == .affine)
                || (mac.bits == 4 && mac.mode == .mxfp4)
            guard valid else { return nil }
            let wCopy = mac.weight + MLXArray(UInt32(0))
            let sCopy = mac.scales + MLXArray(0, dtype: mac.scales.dtype)
            let bCopy = mac.biases.map { $0 + MLXArray(0, dtype: $0.dtype) }
            let pCopy = mac.preScale + MLXArray(0, dtype: mac.preScale.dtype)
            eval(wCopy, sCopy, pCopy)
            if let bCopy { eval(bCopy) }
            return PVQuantTarget(
                path: path, module: mac, packedWeight: wCopy, scales: sCopy, biases: bCopy,
                bias: mac.bias, spec: PVCodeGrid.Spec(groupSize: mac.groupSize, bits: mac.bits, mode: mac.mode),
                isSwitch: false, elementCount: mac.weight.dim(0) * (mac.weight.dim(1) * 32 / mac.bits),
                macaqPreScale: pCopy, macaqBlock: mac.hadamardBlock)
        }
        guard let macs = module as? MACaQSwitchLinear else { return nil }
        let valid = ([2, 3, 4].contains(macs.bits) && macs.mode == .affine)
            || (macs.bits == 4 && macs.mode == .mxfp4)
        guard valid else { return nil }
        let flat = Dictionary(uniqueKeysWithValues: macs.parameters().flattened())
        guard let w = flat["weight"], w.dim(0) <= maxSwitchExperts else { return nil }
        let wCopy = w + MLXArray(UInt32(0))
        let sCopy = macs.scales + MLXArray(0, dtype: macs.scales.dtype)
        let bCopy = macs.biases.map { $0 + MLXArray(0, dtype: $0.dtype) }
        let pCopy = macs.preScale + MLXArray(0, dtype: macs.preScale.dtype)
        eval(wCopy, sCopy, pCopy)
        if let bCopy { eval(bCopy) }
        return PVQuantTarget(
            path: path, module: macs, packedWeight: wCopy, scales: sCopy, biases: bCopy,
            bias: flat["bias"], spec: PVCodeGrid.Spec(groupSize: macs.groupSize, bits: macs.bits, mode: macs.mode),
            isSwitch: true, elementCount: w.dim(0) * w.dim(1) * (w.dim(2) * 32 / macs.bits),
            macaqPreScale: pCopy, macaqBlock: macs.hadamardBlock)
    }

    static func makeDenseProbe(for target: PVQuantTarget, dense: MLXArray) -> Module {
        if let fq = target.module as? FlatQuantLinear {
            let (p1, p2, clip) = (fq.p1, fq.p2, fq.clipBounds)
            return PVPrologueDenseLinear(denseRot: dense, bias: target.bias, rowScale: fq.rowScale) {
                FlatQuantKernels.differentiableTransform($0, p1: p1, p2: p2, clipBounds: clip)
            }
        }
        if let lir = target.module as? LIRQuantLinear {
            let basis = LIRBasisParameters(
                perm: lir.perm, givensPairs: lir.givensPairs, givensAngles: lir.givensAngles,
                diagScale: lir.diagScale, numLayers: lir.numLayers, pairsPerLayer: lir.pairsPerLayer)
            return PVPrologueDenseLinear(denseRot: dense, bias: target.bias) {
                LIRQuantKernels.differentiableTransform($0, basis: basis)
            }
        }
        if let pre = target.macaqPreScale, let blk = target.macaqBlock {
            if target.isSwitch {
                return PVMACaQDenseSwitchLinear(
                    denseRot: dense, bias: target.bias, preScale: pre, hadamardBlock: blk)
            }
            return PVMACaQDenseLinear(
                denseRot: dense, bias: target.bias, preScale: pre, hadamardBlock: blk,
                rowScale: (target.module as? MACaQQuantizedLinear)?.rowScale)
        }
        if target.isSwitch {
            return SwitchLinear(
                inputDims: dense.dim(-1), outputDims: dense.dim(-2),
                numExperts: dense.dim(0), weight: dense, bias: target.bias)
        }
        return Linear(weight: dense, bias: target.bias)
    }

    static func unfreezeMACaQContinuous(mod: Module, maxSwitchExperts: Int) throws -> Bool {
        if let mac = mod as? MACaQQuantizedLinear {
            let keys = mac.mode == .affine ? ["pre_scale", "scales", "biases"] : ["pre_scale"]
            mac.allowTraining(keys)
            return true
        }
        if let macs = mod as? MACaQSwitchLinear {
            let nExp = macs.parameters().flattened().first(where: { $0.0 == "weight" })?.1.dim(0) ?? 0
            let keys = (macs.mode == .affine && nExp > 0 && nExp <= maxSwitchExperts)
                ? ["pre_scale", "scales", "biases"] : ["pre_scale"]
            try macs.unfreeze(keys: keys, strict: false)
            return true
        }
        return false
    }

    static func clampContinuousUpdate(key: String, value: MLXArray) -> MLXArray {
        key.hasSuffix(".pre_scale") ? clip(value, min: Float(1e-4), max: Float(1e4)) : value
    }

    static func requantizeMACaQLinear(
        _ mac: MACaQQuantizedLinear, bits: Int, groupSize: Int, mode: QuantizationMode
    ) -> MACaQQuantizedLinear? {
        let cols = mac.weight.dim(-1) * 32 / mac.bits
        guard cols.isMultiple(of: groupSize), cols.isMultiple(of: mac.hadamardBlock) else { return nil }
        let dense = dequantized(
            mac.weight, scales: mac.scales, biases: mac.biases,
            groupSize: mac.groupSize, bits: mac.bits, mode: mac.mode)
        let q = MLX.quantized(dense, groupSize: groupSize, bits: bits, mode: mode)
        var spec = MACaQLayerSpec(
            bits: bits, groupSize: groupSize, mode: mode, hadamardBlock: mac.hadamardBlock)
        spec.scaleDType = "\(q.scales.dtype)"
        spec.hasBias = mac.bias != nil
        let pre = mac.preScale + MLXArray(0, dtype: mac.preScale.dtype)
        let rowCopy = mac.rowScale.map { $0 + MLXArray(0, dtype: $0.dtype) }
        return MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: q.wq, scales: q.scales, biases: q.biases),
            bias: mac.bias, preScale: pre, spec: spec, rowScale: rowCopy)
    }
}
