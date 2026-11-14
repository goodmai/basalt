import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

package enum PVCodeGrid {
    package struct Spec: Sendable {
        package var groupSize: Int
        package var bits: Int
        package var mode: QuantizationMode

        package init(groupSize: Int, bits: Int, mode: QuantizationMode) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
        }
    }

    package static let nf4TableValues: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ]

    package static func unpackCodes(_ packed: MLXArray, bits: Int) -> MLXArray {
        let leading = Array(packed.shape.dropLast())
        let words = packed.dim(-1)
        let flatRows = max(1, leading.reduce(1, *))
        let p = packed.reshaped(flatRows, words)
        let unpacked: MLXArray
        switch bits {
        case 2:
            let shifts = MLXArray((0 ..< 16).map { UInt32($0 * 2) }).reshaped(1, 1, 16)
            unpacked = ((p.expandedDimensions(axis: -1) >> shifts) & UInt32(3)).asType(.uint8).reshaped(flatRows, words * 16)
        case 4:
            let shifts = MLXArray((0 ..< 8).map { UInt32($0 * 4) }).reshaped(1, 1, 8)
            unpacked = ((p.expandedDimensions(axis: -1) >> shifts) & UInt32(15)).asType(.uint8).reshaped(flatRows, words * 8)
        case 3:
            let blocks = words / 3
            let t = p.reshaped(flatRows, blocks, 3)
            let (u0, u1, u2) = (t[0..., 0..., 0], t[0..., 0..., 1], t[0..., 0..., 2])
            let s0 = MLXArray((0 ..< 10).map { UInt32($0 * 3) }).reshaped(1, 1, 10)
            let s1 = MLXArray((0 ..< 10).map { UInt32(1 + $0 * 3) }).reshaped(1, 1, 10)
            let s2 = MLXArray((0 ..< 10).map { UInt32(2 + $0 * 3) }).reshaped(1, 1, 10)
            let c0 = (u0.expandedDimensions(axis: -1) >> s0) & UInt32(7)
            let c10 = (((u0 >> UInt32(30)) | ((u1 & UInt32(1)) << UInt32(2))) & UInt32(7)).expandedDimensions(axis: -1)
            let c11 = (u1.expandedDimensions(axis: -1) >> s1) & UInt32(7)
            let c21 = (((u1 >> UInt32(31)) | ((u2 & UInt32(3)) << UInt32(1))) & UInt32(7)).expandedDimensions(axis: -1)
            let c22 = (u2.expandedDimensions(axis: -1) >> s2) & UInt32(7)
            unpacked = concatenated([c0, c10, c11, c21, c22], axis: -1).asType(.uint8).reshaped(flatRows, blocks * 32)
        default:
            preconditionFailure("Unsupported PV bit-width: \(bits)")
        }
        return unpacked.reshaped(leading + [unpacked.dim(-1)])
    }

    package static func packCodes(_ codes: MLXArray, bits: Int) -> MLXArray {
        let leading = Array(codes.shape.dropLast())
        let cols = codes.dim(-1)
        let flatRows = max(1, leading.reduce(1, *))
        let u = codes.reshaped(flatRows, cols).asType(.uint32)
        let packed: MLXArray
        switch bits {
        case 2:
            let shifts = MLXArray((0 ..< 16).map { UInt32($0 * 2) }).reshaped(1, 1, 16)
            packed = (u.reshaped(flatRows, cols / 16, 16) << shifts).sum(axis: -1).asType(.uint32)
        case 4:
            let shifts = MLXArray((0 ..< 8).map { UInt32($0 * 4) }).reshaped(1, 1, 8)
            packed = (u.reshaped(flatRows, cols / 8, 8) << shifts).sum(axis: -1).asType(.uint32)
        case 3:
            let blocks = cols / 32
            let b = u.reshaped(flatRows, blocks, 32)
            let s0 = MLXArray((0 ..< 10).map { UInt32($0 * 3) }).reshaped(1, 1, 10)
            let s1 = MLXArray((0 ..< 10).map { UInt32(1 + $0 * 3) }).reshaped(1, 1, 10)
            let s2 = MLXArray((0 ..< 10).map { UInt32(2 + $0 * 3) }).reshaped(1, 1, 10)
            let u0 = (b[0..., 0..., 0 ..< 10] << s0).sum(axis: -1) | ((b[0..., 0..., 10] & UInt32(3)) << UInt32(30))
            let u1 = (b[0..., 0..., 10] >> UInt32(2)) | (b[0..., 0..., 11 ..< 21] << s1).sum(axis: -1) | ((b[0..., 0..., 21] & UInt32(1)) << UInt32(31))
            let u2 = (b[0..., 0..., 21] >> UInt32(1)) | (b[0..., 0..., 22 ..< 32] << s2).sum(axis: -1)
            packed = stacked([u0, u1, u2], axis: -1).reshaped(flatRows, blocks * 3).asType(.uint32)
        default:
            preconditionFailure("Unsupported PV bit-width: \(bits)")
        }
        return packed.reshaped(leading + [packed.dim(-1)])
    }

    package static func dequantizeDifferentiable(
        codes: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, mode: QuantizationMode, dtype: DType
    ) -> MLXArray {
        let leading = Array(codes.shape.dropLast())
        let cols = codes.dim(-1)
        let groups = cols / groupSize
        let s = scales.asType(.float32).expandedDimensions(axis: -1)
        if mode == .nf4 {
            let table = MLXArray(nf4TableValues)
            let vals = table.take(codes.asType(.int32)).reshaped(leading + [groups, groupSize])
            return (vals * s).reshaped(leading + [cols]).asType(dtype)
        }
        let q = codes.asType(.float32).reshaped(leading + [groups, groupSize])
        let b = (biases ?? MLXArray.zeros(scales.shape, dtype: .float32)).asType(.float32).expandedDimensions(axis: -1)
        return (q * s + b).reshaped(leading + [cols]).asType(dtype)
    }

    package static func discreteStep(
        codes: MLXArray, scales: MLXArray, grad: MLXArray, spec: Spec, fraction: Float
    ) -> (codes: MLXArray, updated: Int) {
        let shape = codes.shape
        let leading = Array(shape.dropLast())
        let cols = shape.last ?? 1
        let groups = cols / spec.groupSize
        let maxCode = Int32((1 << spec.bits) - 1)
        let qInt = codes.asType(.int32).reshaped(leading + [groups, spec.groupSize])
        let g = grad.asType(.float32).reshaped(leading + [groups, spec.groupSize])
        let s = scales.asType(.float32).expandedDimensions(axis: -1)
        let gs = g * s
        let stepDir = which(gs .> Float(0), MLXArray(Int32(-1)), which(gs .< Float(0), MLXArray(Int32(1)), MLXArray(Int32(0))))
        let candInt = clip(qInt + stepDir, min: 0, max: maxCode)
        let dw: MLXArray
        if spec.mode == .nf4 {
            let table = MLXArray(nf4TableValues)
            dw = (table.take(candInt) - table.take(qInt)) * s
        } else {
            dw = (candInt - qInt).asType(.float32) * s
        }
        let gain = maximum(Float(0), -(g * dw)).reshaped(-1)
        let total = gain.size
        let k = max(1, min(total - 1, Int(Float(total) * fraction)))
        let part = partitioned(gain, kth: total - k)
        let threshold = part[total - k]
        let mask = (gain .>= threshold) .&& (gain .> Float(1e-11))
        let updated = which(mask, candInt.asType(.uint8).reshaped(-1), codes.reshaped(-1)).reshaped(shape)
        let count = mask.asType(.int32).sum()
        eval(updated, count)
        return (updated, Int(count.item(Int32.self)))
    }
}

package final class PVScaleLinear: Linear {
    let codes: MLXArray
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let computeDType: DType
    @ParameterInfo var scales: MLXArray
    @ParameterInfo var biases: MLXArray?

    package init(from quantized: QuantizedLinear) {
        self.codes = PVCodeGrid.unpackCodes(quantized.weight, bits: quantized.bits)
        self.groupSize = quantized.groupSize
        self.bits = quantized.bits
        self.mode = quantized.mode
        self.computeDType = quantized.scales.dtype
        self._scales.wrappedValue = quantized.scales
        self._biases.wrappedValue = quantized.biases
        super.init(weight: quantized.weight, bias: quantized.bias)
        self.freeze()
        try? self.unfreeze(keys: ["scales", "biases"], strict: false)
    }

    override package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = PVCodeGrid.dequantizeDifferentiable(
            codes: codes, scales: scales, biases: biases, groupSize: groupSize, mode: mode, dtype: x.dtype)
        if let bias { return addMM(bias, x, w.T) }
        return matmul(x, w.T)
    }

    package func packedLayer() -> QuantizedLinear {
        let packed = PVCodeGrid.packCodes(codes, bits: bits)
        return QuantizedLinear(
            weight: packed, bias: bias, scales: scales.asType(computeDType), biases: biases?.asType(computeDType),
            groupSize: groupSize, bits: bits, mode: mode, globalScale: nil)
    }
}

package struct PVTeacherBatch: Sendable {
    package var tokens: [Int32]
    package var topIDs: [Int32]
    package var topLogprobs: [Float]
    package var seqLen: Int
    package var topK: Int

    package init(tokens: [Int32], topIDs: [Int32], topLogprobs: [Float], seqLen: Int, topK: Int) {
        self.tokens = tokens
        self.topIDs = topIDs
        self.topLogprobs = topLogprobs
        self.seqLen = seqLen
        self.topK = topK
    }
}

package struct PVTuneConfig: Sendable {
    package var globalContinuousSteps = 24
    package var continuousStepsPerGroup = 8
    package var batchesPerDiscreteStep = 6
    package var continuousLR: Float = 1.5e-5
    package var discreteFractions: [Float] = [0.0005, 0.0001, 0.00002]
    package var groupSizeBlocks = 4
    package var maxLayerGroups: Int? = 3
    package var maxActiveElements = 150_000_000
    package var maxContinuousSwitchExperts = 16

    package init() {}
}

package struct PVTuneSummary: Sendable {
    package var trainKLBefore: Float
    package var trainKLAfter: Float
    package var valKLBefore: Float
    package var valKLAfter: Float
    package var totalUpdatedCodes: Int
    package var completedGroups: Int
    package var wallSeconds: Double
    package var teacherTokens: Int
    package var rewrittenTensors = 0
}

package enum PVTuner {
    package static func collectTeacherBatches(
        _ container: ModelContainer, texts: String, count: Int, seqLen: Int, topK: Int = 64
    ) async -> [PVTeacherBatch] {
        await container.perform { context in
            let encoded = context.tokenizer.encode(text: texts, addSpecialTokens: false)
            let usable = min(count, encoded.count / (seqLen + 1))
            var batches: [PVTeacherBatch] = []
            batches.reserveCapacity(usable)
            for idx in 0 ..< usable {
                let start = idx * (seqLen + 1)
                let input = encoded[start ..< (start + seqLen)].map(Int32.init)
                let logits = context.model(LMInput.Text(tokens: MLXArray(input)[.newAxis]), cache: nil, state: nil)
                    .logits[0].asType(.float32)
                let lse = logSumExp(logits, axis: -1, keepDims: true)
                let topIdx = argPartition(-logits, kth: topK - 1, axis: -1)[0..., 0 ..< topK]
                let topVal = takeAlong(logits, topIdx, axis: -1) - lse
                eval(topIdx, topVal)
                batches.append(PVTeacherBatch(
                    tokens: input, topIDs: topIdx.asArray(Int32.self), topLogprobs: topVal.asArray(Float.self),
                    seqLen: seqLen, topK: topK))
            }
            return batches
        }
    }

    package static func klLossArrays(_ model: any LanguageModel, inputs: MLXArray, ids: MLXArray, tLogp: MLXArray) -> MLXArray {
        let logits = model(LMInput.Text(tokens: inputs), cache: nil, state: nil).logits[0].asType(.float32)
        let lse = logSumExp(logits, axis: -1, keepDims: true)
        let sTop = takeAlong(logits, ids, axis: -1) - lse
        let pTop = exp(tLogp)
        let pTail = maximum(Float(0), Float(1) - pTop.sum(axis: -1))
        let qTail = maximum(Float(1e-7), Float(1) - exp(sTop).sum(axis: -1))
        let topKL = (pTop * (tLogp - sTop)).sum(axis: -1)
        let tailKL = which(pTail .> Float(1e-7), pTail * (log(maximum(pTail, Float(1e-7))) - log(qTail)), Float(0))
        return (topKL + tailKL).mean()
    }

    package static func klLoss(_ model: any LanguageModel, _ batch: PVTeacherBatch) -> MLXArray {
        let inputs = MLXArray(batch.tokens)[.newAxis]
        let ids = MLXArray(batch.topIDs).reshaped(batch.seqLen, batch.topK)
        let tLogp = MLXArray(batch.topLogprobs).reshaped(batch.seqLen, batch.topK)
        return klLossArrays(model, inputs: inputs, ids: ids, tLogp: tLogp)
    }

    package static func meanKL(_ model: any LanguageModel, _ batches: [PVTeacherBatch]) -> Float {
        guard !batches.isEmpty else { return 0 }
        let root = model as Module
        root.train(false)
        root.invalidateCompiledTraces()
        var total: Float = 0
        for batch in batches {
            let loss = klLoss(model, batch)
            eval(loss)
            total += loss.item(Float.self)
        }
        return total / Float(batches.count)
    }

    package static func textBlockIndex(_ path: String) -> Int? {
        guard !path.contains("vision"), !path.contains("audio") else { return nil }
        let parts = path.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else { return nil }
        return Int(parts[at + 1])
    }

    package static func requantizeModel(_ model: Module, bits: Int, groupSize: Int, mode: QuantizationMode) throws {
        let entries = model.namedModules()
        var replaced = 0
        for (path, mod) in entries {
            guard textBlockIndex(path) != nil, !path.contains("router") else { continue }
            if let ql = mod as? QuantizedLinear, ql.globalScale == nil, ql.bits <= 4 {
                let cols = ql.weight.dim(-1) * 32 / ql.bits
                guard cols.isMultiple(of: groupSize) else { continue }
                let dense = dequantized(ql.weight, scales: ql.scales, biases: ql.biases, groupSize: ql.groupSize, bits: ql.bits, mode: ql.mode)
                let repacked = QuantizedLinear(weight: dense, bias: ql.bias, groupSize: groupSize, bits: bits, mode: mode)
                eval(repacked)
                try swapModule(path, with: repacked, in: model)
                replaced += 1
            } else if let qs = mod as? QuantizedSwitchLinear, qs.bits <= 4 {
                let p = Dictionary(uniqueKeysWithValues: qs.parameters().flattened())
                guard let w = p["weight"], let s = p["scales"] else { continue }
                let cols = w.dim(-1) * 32 / qs.bits
                guard cols.isMultiple(of: groupSize) else { continue }
                let dense = dequantized(w, scales: s, biases: p["biases"], groupSize: qs.groupSize, bits: qs.bits, mode: qs.mode)
                let sw = SwitchLinear(inputDims: dense.dim(-1), outputDims: dense.dim(-2), numExperts: dense.dim(0), weight: dense, bias: p["bias"])
                let repacked = QuantizedSwitchLinear(sw, groupSize: groupSize, bits: bits, mode: mode)
                eval(repacked)
                try swapModule(path, with: repacked, in: model)
                replaced += 1
            }
            if replaced > 0 && replaced.isMultiple(of: 8) { Memory.clearCache() }
        }
        MLXInferenceEngine.settle(model)
        Memory.clearCache()
    }
}
