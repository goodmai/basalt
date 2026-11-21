import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("Experimental 3-bit decode kernel matches MLX's quantized matmul and keeps TinyModel greedy tokens (#653, #228)", .serialized)
struct ThreeBitQMVTests {
    static func packed(experts: Int, outputs: Int, inputs: Int, groupSize: Int, dtype: DType, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
        let w = MLXRandom.normal([experts, outputs, inputs], scale: 0.05, key: MLXRandom.key(seed))
        let q = MLX.quantized(w, groupSize: groupSize, bits: 3, mode: .affine)
        return (q.wq, q.scales.asType(dtype), q.biases!.asType(dtype))
    }

    static func tolerance(_ dtype: DType, magnitude: Float) -> Float {
        (dtype == .float32 ? 1e-4 : 2e-2) * max(1, magnitude)
    }

    @Test("Gather matches MLX gatherQuantizedMM for every dtype and group size, with a K tail past 512 and partial output rows", arguments: [DType.bfloat16, .float16, .float32], [32, 64, 128])
    func gatherParity(dtype: DType, groupSize: Int) {
        let (experts, outputs, inputs) = (4, 37, 1152)
        let (w, scales, biases) = Self.packed(experts: experts, outputs: outputs, inputs: inputs, groupSize: groupSize, dtype: dtype, seed: 6531)
        let indices = MLXArray([Int32(3), 0, 2, 3, 1, 1]).reshaped(3, 2)
        let x = MLXRandom.normal([3, 2, 1, inputs], key: MLXRandom.key(6532)).asType(dtype)
        #expect(ThreeBitQMV.handlesGather(x, indices: indices, scales: scales, biases: biases))
        let got = ThreeBitQMV.matmul(x, indices: indices, weight: w, scales: scales, biases: biases, groupSize: groupSize)
        let ref = MLX.gatherQuantizedMM(x, w, scales: scales, biases: biases, rhsIndices: indices, transpose: true, groupSize: groupSize, bits: 3, mode: .affine)
        let diff = abs(got.asType(.float32) - ref.asType(.float32)).max()
        let magnitude = abs(ref.asType(.float32)).max()
        eval(diff, magnitude)
        #expect(got.shape == ref.shape)
        #expect(got.dtype == dtype)
        #expect(diff.item(Float.self) <= Self.tolerance(dtype, magnitude: magnitude.item(Float.self)))
    }

    @Test("Single-row dense matches MLX quantizedMM, including K = 2880 and 704 that MLX sends to its slow path", arguments: [(64, 2880, 64), (36, 704, 64), (8, 32, 32)])
    func denseParity(shape: (Int, Int, Int)) {
        let (outputs, inputs, groupSize) = shape
        let (w3, s3, b3) = Self.packed(experts: 1, outputs: outputs, inputs: inputs, groupSize: groupSize, dtype: .bfloat16, seed: 6533)
        let (w, scales, biases) = (w3[0], s3[0], b3[0])
        let x = MLXRandom.normal([1, 1, inputs], key: MLXRandom.key(6534)).asType(.bfloat16)
        #expect(ThreeBitQMV.handlesSingleRow(x, scales: scales, biases: biases))
        let got = ThreeBitQMV.matmul(x, indices: nil, weight: w, scales: scales, biases: biases, groupSize: groupSize)
        let ref = MLX.quantizedMM(x, w, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: 3, mode: .affine)
        let diff = abs(got.asType(.float32) - ref.asType(.float32)).max()
        let magnitude = abs(ref.asType(.float32)).max()
        eval(diff, magnitude)
        #expect(got.shape == ref.shape)
        #expect(diff.item(Float.self) <= Self.tolerance(.bfloat16, magnitude: magnitude.item(Float.self)))
    }

    @Test("Refuses what it does not cover: 4-bit, mxfp4, no biases, K not a multiple of 16, several rows, mixed dtypes")
    func refusals() {
        #expect(!ThreeBitQMV.fits(bits: 4, mode: .affine, groupSize: 64, inputs: 2880, hasBiases: true))
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .mxfp4, groupSize: 32, inputs: 2880, hasBiases: true))
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 64, inputs: 2880, hasBiases: false))
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 64, inputs: 72, hasBiases: true))
        let s = MLXArray.zeros([4, 2], dtype: .bfloat16)
        #expect(!ThreeBitQMV.handlesSingleRow(MLXArray.zeros([2, 128], dtype: .bfloat16), scales: s, biases: s))
        #expect(!ThreeBitQMV.handlesSingleRow(MLXArray.zeros([1, 128], dtype: .float32), scales: s, biases: s))
        #expect(!ThreeBitQMV.handlesGather(MLXArray.zeros([2, 3, 128], dtype: .bfloat16), indices: MLXArray([Int32(0), 1]), scales: s, biases: s))
    }

    @Test("MACaQ 3-bit TinyModel decode keeps the same greedy tokens with the kernel on, dense and MoE", arguments: [TinyModel.Family.qwen3, .qwen3Moe, .gemma4Moe])
    func tinyModelGreedyParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let batches: [[Int32]] = [(0 ..< 16).map { Int32(($0 * 7 + 3) % 200) }, (0 ..< 16).map { Int32(($0 * 11 + 5) % 200) }]
        var cfg = MACaQConfig()
        cfg.hadamardBlock = 32
        cfg.maxFlipIterations = 4
        _ = try MACaQRefiner.quantizeModel(model, targetMode: .affine(bits: 3, groupSize: 32), calibrationTokenBatches: batches, config: cfg)
        let halves = model.parameters().flattened().compactMap { key, value in value.dtype == .float32 ? (key, value.asType(.bfloat16)) : nil }
        model.update(parameters: ModuleParameters.unflattened(halves))
        model.train(false)
        let installed = ThreeBitQMV.install(in: model, enabled: false)
        #expect(installed > 0)
        let (offTokens, offLogits) = try Self.greedy(model)
        ThreeBitQMV.install(in: model)
        let (onTokens, onLogits) = try Self.greedy(model)
        ThreeBitQMV.install(in: model, enabled: false)
        #expect(onTokens == offTokens)
        #expect(zip(onLogits, offLogits).allSatisfy { abs($0 - $1) < 5e-2 })
    }

    @Test("With the flag on, MACaQ dense and expert layers route a one-row decode through the kernel, and stay on MLX with it off")
    func routing() {
        let (experts, outputs, inputs) = (4, 40, 1152)
        let (w, scales, biases) = Self.packed(experts: experts, outputs: outputs, inputs: inputs, groupSize: 64, dtype: .bfloat16, seed: 6535)
        let spec = MACaQLayerSpec(bits: 3, groupSize: 64, mode: .affine, hadamardBlock: 64)
        let pre = MLXArray.ones([inputs], dtype: .bfloat16)
        let dense = MACaQQuantizedLinear(packed: MACaQPackedWeights(weight: w[0], scales: scales[0], biases: biases[0]), bias: nil, preScale: pre, spec: spec)
        let sw = MACaQSwitchLinear(
            geometry: MACaQSwitchGeometry(experts: experts, outputs: outputs, inputs: inputs),
            packed: MACaQPackedWeights(weight: w, scales: scales, biases: biases), bias: nil, preScale: pre, spec: spec)
        dense.train(false)
        sw.train(false)
        #expect(ThreeBitQMV.install(in: dense) == 1 && ThreeBitQMV.install(in: sw) == 1)
        let x = MLXRandom.normal([1, 1, inputs], key: MLXRandom.key(6536)).asType(.bfloat16)
        let rotated = MACaQFusedKernels.blockHadamard(x, scaleVector: pre, blockSize: 64, isInverse: false)
        let kernelDense = ThreeBitQMV.matmul(rotated, indices: nil, weight: w[0], scales: scales[0], biases: biases[0], groupSize: 64)
        let mlxDense = MLX.quantizedMM(rotated, w[0], scales: scales[0], biases: biases[0], transpose: true, groupSize: 64, bits: 3, mode: .affine)
        let indices = MLXArray([Int32(2), 0]).reshaped(1, 2)
        let xs = expandedDimensions(x, axes: [-2, -3]).reshaped(1, 1, 1, inputs)
        let swRotated = MACaQFusedKernels.switchBlockHadamard(xs, indices: indices, preScale: pre, blockSize: 64, numExperts: experts)
        let kernelSwitch = ThreeBitQMV.matmul(swRotated, indices: indices, weight: w, scales: scales, biases: biases, groupSize: 64)
        let mlxSwitch = MLX.gatherQuantizedMM(swRotated, w, scales: scales, biases: biases, rhsIndices: indices, transpose: true, groupSize: 64, bits: 3, mode: .affine)
        let denseOn = dense(x), switchOn = sw(xs, indices)
        ThreeBitQMV.install(in: dense, enabled: false)
        ThreeBitQMV.install(in: sw, enabled: false)
        let denseOff = dense(x), switchOff = sw(xs, indices)
        func same(_ a: MLXArray, _ b: MLXArray) -> Bool { a.shape == b.shape && abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) == 0 }
        #expect(same(denseOn, kernelDense) && same(denseOff, mlxDense))
        #expect(same(switchOn, kernelSwitch) && same(switchOff, mlxSwitch))
    }

    static func greedy(_ model: some LanguageModel) throws -> ([Int32], [Float]) {
        let cache = try model.newCache(parameters: nil)
        var state: LMOutput.State?
        var token: Int32 = 5
        var tokens: [Int32] = []
        var tops: [Float] = []
        for step in 0 ..< 12 {
            let input = step < 4 ? Int32([5, 19, 42, 7][step]) : token
            let out = model(LMInput.Text(tokens: MLXArray([input])[.newAxis]), cache: cache, state: state)
            state = out.state
            let logits = out.logits[0, -1].asType(.float32)
            let next = argMax(logits, axis: -1)
            let top = logits.max()
            eval(next, top)
            token = next.item(Int32.self)
            tokens.append(token)
            tops.append(top.item(Float.self))
        }
        return (tokens, tops)
    }
}
