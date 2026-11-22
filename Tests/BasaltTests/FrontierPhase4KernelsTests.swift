import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Fused RMSNorm + QuantizedMatvec and SwiGLU/GeGLU + QuantizedMatvec Metal kernels (#490 P6)")
struct FusedTransformerKernelsTests {
    static func quantizedLayer(
        inputs: Int, outputs: Int, groupSize: Int, bits: Int, dtype: DType, bias: Bool
    ) -> QuantizedLinear {
        let seed = UInt64(inputs * 11 + outputs * 7 + groupSize * 3 + bits)
        let key = MLXRandom.key(seed)
        let dense = MLXRandom.normal([outputs, inputs], scale: 0.08, key: key).asType(dtype)
        let offset = bias ? MLXRandom.normal([outputs], scale: 0.05, key: key).asType(dtype) : nil
        let layer = QuantizedLinear(weight: dense, bias: offset, groupSize: groupSize, bits: bits)
        eval(layer)
        return layer
    }

    static func relativeError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        let ref = reference.asType(.float32)
        let denom = max(abs(ref).max().item(Float.self), 1e-3)
        return abs(actual.asType(.float32) - ref).max().item(Float.self) / denom
    }

    static func tolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .bfloat16: 1e-2
        case .float16: 1e-3
        default: 1e-4
        }
    }

    @Test(
        "Fused RMSNorm + QuantizedMatvec matches fp32 dequantized reference and stock MLX across bits, group sizes, dtypes and M=1..8",
        arguments: [4, 8], [32, 64, 128]
    )
    func rmsNormQuantizedMatmulParity(bits: Int, groupSize: Int) {
        let inputs = 384
        let outputs = 37
        for dtype in [DType.float16, .bfloat16, .float32] {
            let linear = Self.quantizedLayer(
                inputs: inputs, outputs: outputs, groupSize: groupSize, bits: bits, dtype: dtype, bias: true)
            let norm = RMSNorm(dimensions: inputs, eps: 1e-5)
            norm.update(parameters: ModuleParameters.unflattened([
                "weight": (MLXArray.ones([inputs], type: Float.self) + MLXRandom.normal([inputs], scale: 0.1, key: MLXRandom.key(17))).asType(dtype),
            ]))
            eval(norm)
            let dense = dequantized(
                linear.weight, scales: linear.scales, biases: linear.biases,
                groupSize: groupSize, bits: bits
            ).asType(.float32)
            let fusedLayer = FusedRMSNormQuantizedLinear(replacing: linear, normWeight: norm.weight, eps: norm.eps)
            for rows in [1, 2, 4, 8] {
                let x = MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(UInt64(rows * 19 + bits))).asType(dtype)
                let actual = FusedTransformerKernels.rmsNormQuantizedMatmul(x, norm: norm, linear: linear)
                let viaModule = fusedLayer(x)
                let stock = linear(norm(x))
                let xFp32 = x.asType(.float32)
                let rms = rsqrt(mean(xFp32 * xFp32, axis: -1, keepDims: true) + 1e-5)
                let normedFp32 = xFp32 * rms * norm.weight.asType(.float32)
                let fp32Ref = matmul(normedFp32, dense.T) + linear.bias!.asType(.float32)
                eval(actual, viaModule, stock, fp32Ref)
                #expect(actual.shape == [1, rows, outputs] && actual.dtype == dtype)
                #expect((actual .== viaModule).all().item(Bool.self))
                #expect(Self.relativeError(actual, fp32Ref) < Self.tolerance(dtype))
                #expect(Self.relativeError(actual, stock) < Self.tolerance(dtype))
            }
        }
    }

    @Test(
        "Fused SwiGLU and GeGLU + QuantizedMatvec match fp32 dequantized reference and stock MLX across bits, group sizes, dtypes and M=1..8",
        arguments: [4, 8], [32, 64, 128]
    )
    func gatedQuantizedMatmulParity(bits: Int, groupSize: Int) {
        let inputs = 384
        let outputs = 45
        for dtype in [DType.float16, .bfloat16, .float32] {
            let down = Self.quantizedLayer(
                inputs: inputs, outputs: outputs, groupSize: groupSize, bits: bits, dtype: dtype, bias: true)
            let dense = dequantized(
                down.weight, scales: down.scales, biases: down.biases,
                groupSize: groupSize, bits: bits
            ).asType(.float32)
            for activation in GatedActivation.allCases {
                for rows in [1, 2, 4, 8] {
                    let seed = UInt64(rows * 29 + bits * 5 + activation.rawValue)
                    let gate = MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(seed)).asType(dtype)
                    let up = MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(seed + 101)).asType(dtype)
                    let actual = FusedTransformerKernels.gatedQuantizedMatmul(
                        gate: gate, up: up, activation: activation, down: down)
                    let stockAct = activation == .silu ? silu(gate) * up : geluApproximate(gate) * up
                    let stock = down(stockAct)
                    let g32 = gate.asType(.float32)
                    let u32 = up.asType(.float32)
                    let refAct = activation == .silu ? silu(g32) * u32 : geluApproximate(g32) * u32
                    let fp32Ref = matmul(refAct, dense.T) + down.bias!.asType(.float32)
                    eval(actual, stock, fp32Ref)
                    #expect(actual.shape == [1, rows, outputs] && actual.dtype == dtype)
                    #expect(Self.relativeError(actual, fp32Ref) < Self.tolerance(dtype))
                    #expect(Self.relativeError(actual, stock) < Self.tolerance(dtype))
                }
            }
        }
    }

    @Test("Rows above 8 and unsupported widths fall back cleanly to exact stock MLX execution")
    func fallbackOutsideDecodeEnvelope() {
        let linear = Self.quantizedLayer(inputs: 128, outputs: 32, groupSize: 64, bits: 4, dtype: .float16, bias: false)
        let norm = RMSNorm(dimensions: 128, eps: 1e-5)
        eval(norm)
        let xWide = MLXRandom.normal([1, 12, 128], key: MLXRandom.key(77)).asType(.float16)
        #expect(!FusedTransformerKernels.supports(rows: 12, bits: 4, groupSize: 64, columns: 128, dtype: .float16))
        #expect(!FusedTransformerKernels.supports(rows: 1, bits: 4, groupSize: 64, columns: 96, dtype: .float16))
        let fusedNormOut = FusedTransformerKernels.rmsNormQuantizedMatmul(xWide, norm: norm, linear: linear)
        let stockNormOut = linear(norm(xWide))
        let uWide = MLXRandom.normal([1, 12, 128], key: MLXRandom.key(78)).asType(.float16)
        let fusedGateOut = FusedTransformerKernels.gatedQuantizedMatmul(
            gate: xWide, up: uWide, activation: .silu, down: linear)
        let stockGateOut = linear(silu(xWide) * uWide)
        eval(fusedNormOut, stockNormOut, fusedGateOut, stockGateOut)
        #expect((fusedNormOut .== stockNormOut).all().item(Bool.self))
        #expect((fusedGateOut .== stockGateOut).all().item(Bool.self))
    }

    @Test("Quantized TinyModel Qwen3 and Gemma4 keep greedy token parity and MLP parity when fused kernels are installed (#228)")
    func tinyModelGreedyTokenAndMLPParity() throws {
        let model = try TinyModel.make(.qwen3, adding: "\"hidden_size\": 256, \"intermediate_size\": 512, ")
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        let prompt = MLXArray([Int32(3), 141, 59, 26, 5, 35]).reshaped(1, -1)
        let stockCache = try model.newCache(parameters: nil)
        let stockPrefillLogits = model(prompt, cache: stockCache)
        eval(stockPrefillLogits)
        var stockLogits = stockPrefillLogits
        var stockTokens = [stockLogits[0, -1].argMax().item(Int32.self)]
        for _ in 0 ..< 3 {
            stockLogits = model(MLXArray([stockTokens.last!]).reshaped(1, 1), cache: stockCache)
            stockTokens.append(stockLogits[0, -1].argMax().item(Int32.self))
        }
        let installed = FusedRMSNormQuantizedLinear.install(in: model)
        #expect(installed == 4 * 5)
        let fusedCache = try model.newCache(parameters: nil)
        var fusedLogits = model(prompt, cache: fusedCache)
        eval(fusedLogits)
        #expect(Self.relativeError(fusedLogits, stockPrefillLogits) < 2e-3)
        var fusedTokens = [fusedLogits[0, -1].argMax().item(Int32.self)]
        for _ in 0 ..< 3 {
            fusedLogits = model(MLXArray([fusedTokens.last!]).reshaped(1, 1), cache: fusedCache)
            fusedTokens.append(fusedLogits[0, -1].argMax().item(Int32.self))
        }
        #expect(fusedTokens == stockTokens)
        try Self.assertTinyMLPParity(family: .qwen3, activation: .silu)
        try Self.assertTinyMLPParity(family: .gemma4, activation: .geluApprox)
        let sparse = try TinyModel.make(.qwen3, adding: "\"hidden_size\": 256, \"intermediate_size\": 512, ")
        sparse.update(parameters: sparse.parameters().mapValues { $0.asType(.float16) })
        let sparseBlocks = LeJEPABackbone.layers(of: sparse)
        quantize(model: sparseBlocks[1], groupSize: 64, bits: 4)
        quantize(model: sparseBlocks[3], groupSize: 64, bits: 4)
        eval(sparse)
        #expect(FusedRMSNormQuantizedLinear.install(in: sparse) == 2 * 5)
    }

    static func assertTinyMLPParity(family: TinyModel.Family, activation: GatedActivation) throws {
        let tiny = try TinyModel.make(family)
        tiny.update(parameters: tiny.parameters().mapValues { $0.asType(.float16) })
        quantize(model: tiny, groupSize: 32, bits: 4) { path, mod in
            path.contains(".mlp.") && mod is Linear && !(mod is QuantizedLinear)
        }
        eval(tiny)
        let map = Dictionary(tiny.namedModules(), uniquingKeysWith: { a, _ in a })
        let gate = try #require(map["model.layers.0.mlp.gate_proj"] as? Linear)
        let up = try #require(map["model.layers.0.mlp.up_proj"] as? Linear)
        let down = try #require(map["model.layers.0.mlp.down_proj"] as? Linear)
        let fusedMLP = FusedGatedQuantizedMLP(gateProj: gate, upProj: up, downProj: down, activation: activation)
        for rows in [1, 4, 8] {
            let x = MLXRandom.normal([1, rows, 64], key: MLXRandom.key(UInt64(rows + 50))).asType(.float16)
            let actual = fusedMLP(x)
            let g = gate(x)
            let u = up(x)
            let expected = down(activation == .silu ? silu(g) * u : geluApproximate(g) * u)
            eval(actual, expected)
            #expect(Self.relativeError(actual, expected) < 1e-3)
        }
    }
}

@Suite("Single-simdgroup fused MoE router TopK + Softmax Metal kernel (#490 Pillar 4.4)")
struct FusedMoERouterKernelTests {
    static func distinctLogits(rows: Int, experts: Int, dtype: DType, seed: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(rows * experts)
        let step = experts * 2 - 1
        for r in 0 ..< rows {
            for c in 0 ..< experts {
                let perm = ((c * 37 + r * 13 + seed * 7) % experts)
                let raw = Float(perm) * 0.03125 - Float(step) * 0.0078125
                let top = perm - (experts - 20)
                values.append(experts <= 256 ? raw : top >= 0 ? 1 + Float(top) * 0.0625 : Float(perm) / Float(experts) - 1)
            }
        }
        return MLXArray(values, [rows, experts]).asType(dtype)
    }

    static func weightTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .bfloat16: 1e-2
        case .float16: 1e-3
        default: 1e-5
        }
    }

    @Test(
        "Exact expert top-K indices and softmax weight parity across V in [4, 8, 64, 128, 256, 300, 512, 1024], K in [2, 4, 8, 10, 16], dtypes and modes",
        arguments: [4, 8, 64, 128, 256, 300, 512, 1024], [DType.float16, .bfloat16, .float32]
    )
    func routerTemplateMatrixParity(experts: Int, dtype: DType) {
        let ks = [2, 4, 8, 10, 16].filter { $0 <= experts }
        for k in ks {
            for rows in [1, 5, 8] {
                let logits = Self.distinctLogits(rows: rows, experts: experts, dtype: dtype, seed: k + rows)
                eval(logits)
                for (softmaxFirst, normTopk) in [(true, true), (true, false), (false, true)] {
                    let actual = FusedMoERouterKernel.route(
                        logits, k: k, normTopkProb: normTopk, softmaxBeforeTopK: softmaxFirst)
                    let expected = FusedMoERouterKernel.reference(
                        logits, k: k, normTopkProb: normTopk, softmaxBeforeTopK: softmaxFirst)
                    eval(actual.indices, actual.weights, expected.indices, expected.weights)
                    #expect(actual.indices.shape == [rows, k] && actual.indices.dtype == .int32)
                    #expect(actual.weights.shape == [rows, k] && actual.weights.dtype == dtype)
                    #expect((actual.indices .== expected.indices).all().item(Bool.self))
                    let err = abs(actual.weights.asType(.float32) - expected.weights.asType(.float32)).max().item(Float.self)
                    #expect(err < Self.weightTolerance(dtype))
                }
            }
        }
    }

    @Test("Deterministic lowest-index tie-breaking, NaN/-Inf masking, 3D preservation and V > 1024 fallback")
    func tieBreakingAndFallback() {
        let tied = MLXArray([Float(1.0), 2.5, 2.5, -Float.infinity, .nan, 2.5, 0.5, 2.5], [1, 2, 4])
        let actual = FusedMoERouterKernel.route(tied, k: 3, normTopkProb: true, softmaxBeforeTopK: true)
        let expected = FusedMoERouterKernel.reference(tied, k: 3, normTopkProb: true, softmaxBeforeTopK: true)
        eval(actual.indices, actual.weights, expected.indices, expected.weights)
        #expect(actual.indices.shape == [1, 2, 3])
        #expect((actual.indices .== expected.indices).all().item(Bool.self))
        #expect(actual.indices[0, 0].asArray(Int32.self) == [1, 2, 0])
        #expect(actual.indices[0, 1].asArray(Int32.self) == [1, 3, 2])
        #expect(abs(actual.weights - expected.weights).max().item(Float.self) < 1e-5)

        let wide = Self.distinctLogits(rows: 2, experts: 1100, dtype: .float16, seed: 9)
        #expect(!FusedMoERouterKernel.supports(wide, k: 4))
        let wideOut = FusedMoERouterKernel.route(wide, k: 4)
        let wideRef = FusedMoERouterKernel.reference(wide, k: 4)
        eval(wideOut.indices, wideOut.weights, wideRef.indices, wideRef.weights)
        #expect((wideOut.indices .== wideRef.indices).all().item(Bool.self))
        #expect((wideOut.weights .== wideRef.weights).all().item(Bool.self))
    }

    @Test("TinyModel Qwen3MoE and Gemma4MoE routers match stock expert selection, SwitchGLU outputs and greedy tokens (#228)")
    func tinyMoEModelParity() throws {
        let qwenMoe = try TinyModel.make(.qwen3Moe)
        let qMap = Dictionary(qwenMoe.namedModules(), uniquingKeysWith: { a, _ in a })
        let qBlock = try #require(qMap["model.layers.0.mlp"] as? UnaryLayer)
        let qGate = try #require(qMap["model.layers.0.mlp.gate"] as? Linear)
        let qSwitch = try #require(qMap["model.layers.0.mlp.switch_mlp"] as? SwitchGLU)
        let x = MLXRandom.normal([1, 4, 64], key: MLXRandom.key(228)).asType(.float16)
        qwenMoe.update(parameters: qwenMoe.parameters().mapValues { $0.asType(.float16) })
        eval(qwenMoe, x)
        let stockOut = qBlock(x)
        let routed = FusedMoERouterKernel.route(qGate(x), k: 2, normTopkProb: true, softmaxBeforeTopK: true)
        let fusedOut = weightedExpertSum(qSwitch(x, routed.indices), routed.weights)
        eval(stockOut, fusedOut)
        #expect(FusedTransformerKernelsTests.relativeError(fusedOut, stockOut) < 1e-3)

        quantize(model: qwenMoe, groupSize: 32, bits: 4)
        eval(qwenMoe)
        let prompt = MLXArray([Int32(4), 19, 88, 12]).reshaped(1, -1)
        let stockLogits = qwenMoe(prompt, cache: nil)
        eval(stockLogits)
        let installed = FusedRMSNormQuantizedLinear.install(in: qwenMoe)
        #expect(installed == 4 * 3)
        let fusedLogits = qwenMoe(prompt, cache: nil)
        eval(fusedLogits)
        #expect((fusedLogits.argMax(axis: -1) .== stockLogits.argMax(axis: -1)).all().item(Bool.self))

        let gemmaMoe = try TinyModel.make(.gemma4Moe)
        gemmaMoe.update(parameters: gemmaMoe.parameters().mapValues { $0.asType(.float16) })
        eval(gemmaMoe)
        let gMap = Dictionary(gemmaMoe.namedModules(), uniquingKeysWith: { a, _ in a })
        let gParams = Dictionary(gemmaMoe.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let proj = try #require(gMap["model.layers.0.router.proj"] as? Linear)
        let gSwitch = try #require(gMap["model.layers.0.experts.switch_glu"] as? SwitchGLU)
        let rScale = try #require(gParams["model.layers.0.router.scale"])
        let eScale = try #require(gParams["model.layers.0.router.per_expert_scale"])
        let rootScale = Float(1.0) / Float(64).squareRoot()
        let normed = MLXFast.rmsNorm(x, weight: (rScale * rootScale).asType(x.dtype), eps: 1e-6)
        let scores = proj(normed)
        let gRouted = FusedMoERouterKernel.route(scores, k: 2, normTopkProb: true, softmaxBeforeTopK: false)
        let gRef = FusedMoERouterKernel.reference(scores, k: 2, normTopkProb: true, softmaxBeforeTopK: false)
        let gFused = weightedExpertSum(gSwitch(x, gRouted.indices), gRouted.weights * eScale[gRouted.indices].asType(x.dtype))
        let gStock = weightedExpertSum(gSwitch(x, gRef.indices), gRef.weights * eScale[gRef.indices].asType(x.dtype))
        eval(gRouted.indices, gRef.indices, gFused, gStock)
        #expect((gRouted.indices .== gRef.indices).all().item(Bool.self))
        #expect(FusedTransformerKernelsTests.relativeError(gFused, gStock) < 1e-3)
    }

    @Test(
        "GPU microbenchmark for fused RMSNorm+QMV, SwiGLU+QMV and MoE router",
        .enabled(if: ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1")
    )
    func microbenchmark() {
        let qmv = FusedTransformerKernelsTests.quantizedLayer(
            inputs: 4096, outputs: 4096, groupSize: 64, bits: 4, dtype: .float16, bias: false)
        let down = FusedTransformerKernelsTests.quantizedLayer(
            inputs: 12288, outputs: 4096, groupSize: 64, bits: 4, dtype: .float16, bias: false)
        let norm = RMSNorm(dimensions: 4096, eps: 1e-5)
        let x = MLXRandom.normal([1, 1, 4096], key: MLXRandom.key(1)).asType(.float16)
        let g = MLXRandom.normal([1, 1, 12288], key: MLXRandom.key(2)).asType(.float16)
        let u = MLXRandom.normal([1, 1, 12288], key: MLXRandom.key(3)).asType(.float16)
        let logits = Self.distinctLogits(rows: 1, experts: 256, dtype: .float16, seed: 4)
        eval(qmv, down, norm, x, g, u, logits)
        for _ in 0 ..< 5 {
            eval(
                FusedTransformerKernels.rmsNormQuantizedMatmul(x, norm: norm, linear: qmv),
                FusedTransformerKernels.gatedQuantizedMatmul(gate: g, up: u, activation: .silu, down: down),
                FusedMoERouterKernel.route(logits, k: 8).weights)
        }
        let clock = ContinuousClock()
        let stockElapsed = clock.measure {
            for _ in 0 ..< 50 {
                let a = qmv(norm(x))
                let b = down(silu(g) * u)
                let r = FusedMoERouterKernel.reference(logits, k: 8)
                eval(a, b, r.indices, r.weights)
            }
        }
        let fusedElapsed = clock.measure {
            for _ in 0 ..< 50 {
                let a = FusedTransformerKernels.rmsNormQuantizedMatmul(x, norm: norm, linear: qmv)
                let b = FusedTransformerKernels.gatedQuantizedMatmul(gate: g, up: u, activation: .silu, down: down)
                let r = FusedMoERouterKernel.route(logits, k: 8)
                eval(a, b, r.indices, r.weights)
            }
        }
        let stockNs = Double(stockElapsed.components.seconds) * 1e9 + Double(stockElapsed.components.attoseconds) * 1e-9
        let fusedNs = Double(fusedElapsed.components.seconds) * 1e9 + Double(fusedElapsed.components.attoseconds) * 1e-9
        print(String(
            format: "[BENCH #490-P4] 50 iters (4096/12288 Q4_64 + 256-expert Top8): stock=%.2f ms fused=%.2f ms speedup=%.2fx",
            stockNs / 1e6, fusedNs / 1e6, stockNs / max(fusedNs, 1.0)))
        #expect(stockElapsed > .zero && fusedElapsed > .zero)
    }
}

@Suite("Zero-copy Span/RawSpan, InlineArray AIO, and UniqueArray ownership paths (#490 Pillars 1.3 & 2)")
struct FrontierOwnershipAndZeroCopyTests {
    @Test("AudioClip WAV little-endian framing and AVAudioPCMBuffer OutputSpan decode/resample roundtrip")
    func audioClipWavFramingAndDecodeRoundtrip() throws {
        let samples: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.125]
        let clip = try AudioClip(samples: samples)
        let wav = clip.wav
        #expect(wav.count == 44 + samples.count * 2)
        #expect(String(decoding: wav[0 ..< 4], as: UTF8.self) == "RIFF")
        #expect(String(decoding: wav[8 ..< 16], as: UTF8.self) == "WAVEfmt ")
        #expect(String(decoding: wav[36 ..< 40], as: UTF8.self) == "data")
        let chunkBytes = UInt32(wav[40]) | (UInt32(wav[41]) << 8) | (UInt32(wav[42]) << 16) | (UInt32(wav[43]) << 24)
        #expect(chunkBytes == UInt32(samples.count * 2))
        let posHalf = Int16(bitPattern: UInt16(wav[46]) | (UInt16(wav[47]) << 8))
        let negHalf = Int16(bitPattern: UInt16(wav[48]) | (UInt16(wav[49]) << 8))
        #expect(posHalf == Int16(0.5 * Float(Int16.max)) && negHalf == Int16(-0.5 * Float(Int16.max)))

        let sine = (0 ..< 1_600).map { Darwin::sin(Float($0) * 0.1) * 0.5 }
        let sineClip = try AudioClip(samples: sine)
        let decoded = try AudioClip(encoded: sineClip.wav, format: "wav")
        #expect(abs(decoded.samples.count - sine.count) <= 8)
        let targetFormat = try #require(AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 8_000, channels: 1, interleaved: false))
        let resampled = try sineClip.buffer(in: targetFormat)
        #expect(resampled.frameLength >= 790 && resampled.frameLength <= 810)
    }

    @Test("GGUFDequant RawSpan overload decodes f16, bf16 and q8_0 slices without UnsafeRawBufferPointer")
    func ggufDequantRawSpan() {
        let oneF16 = Float16(1.5).bitPattern
        let negTwoF16 = Float16(-2.25).bitPattern
        let f16Bytes: [UInt8] = [
            UInt8(oneF16 & 0xFF), UInt8(oneF16 >> 8),
            UInt8(negTwoF16 & 0xFF), UInt8(negTwoF16 >> 8),
        ]
        let decodedF16 = GGUFDequant.floats(f16Bytes.span.bytes, type: .f16, count: 2)
        #expect(decodedF16 == [1.5, -2.25])

        let scaleHalf = Float16(0.5).bitPattern
        var q8Block: [UInt8] = [UInt8(scaleHalf & 0xFF), UInt8(scaleHalf >> 8)]
        for i in 0 ..< 32 { q8Block.append(UInt8(bitPattern: Int8(i - 16))) }
        let data = Data(q8Block)
        let decodedQ8 = GGUFDequant.floats(data.bytes, type: .q8_0, count: 32)
        #expect(decodedQ8.count == 32 && decodedQ8[0] == -8.0 && decodedQ8[16] == 0.0 && decodedQ8[31] == 7.5)
    }

    @Test("ExpertReader InlineArray lio_listio batches (> 16 requests) and Span slicing read exact bytes")
    func expertReaderInlineArrayBatchesAndSpan() throws {
        let chunkSize = 64
        let totalChunks = 21
        var payload = Data(capacity: chunkSize * totalChunks)
        for chunk in 0 ..< totalChunks {
            payload.append(contentsOf: [UInt8](repeating: UInt8((chunk * 7 + 3) & 0xFF), count: chunkSize))
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("expert-aio-\(UUID().uuidString).bin")
        try payload.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let dst = UnsafeMutableRawPointer.allocate(byteCount: payload.count, alignment: 64)
        defer { dst.deallocate() }
        dst.initializeMemory(as: UInt8.self, repeating: 0, count: payload.count)
        let requests = (0 ..< totalChunks).map { idx in
            ExpertReader.Request(
                descriptor: handle.fileDescriptor,
                offset: Int64(idx * chunkSize),
                destination: dst + idx * chunkSize,
                count: chunkSize)
        }
        ExpertReader.read(requests.span)
        let readBack = Data(bytes: dst, count: payload.count)
        #expect(readBack == payload)
    }

    @Test("TokenRepetitionWindow UniqueArray borrow/mutate and ModuleSlots positions map match reference")
    func uniqueArrayRepetitionWindowAndModuleSlots() throws {
        var window = TokenRepetitionWindow(probeSize: 4, threshold: 3, minTokens: 8, retain: 12)
        let pattern = [11, 22, 33, 44]
        var stream: [Int] = []
        for step in 0 ..< 30 {
            let token = pattern[step % pattern.count]
            stream.append(token)
            let s = window.append(token)
            let suffix = Array(stream.suffix(12))
            let expected = suffix.count >= 8 ? RepetitionGuard.score(tokens: suffix.span, probeSize: 4, threshold: 3) : 0
            #expect(abs(s - expected) < 1e-12)
            window.withTokens { span in
                #expect(span.count == suffix.count)
                for idx in 0 ..< suffix.count { #expect(span[idx] == suffix[idx]) }
            }
        }
        #expect(window.storage.count <= 24 && window.score == 1.0)

        var guardText = RepetitionGuard(probeSize: 6, threshold: 3, minLength: 18, retain: 64)
        let stopped = guardText.shouldStop(after: "abcdefabcdefabcdef")
        #expect(stopped)

        let layout = ExpertLayout(
            path: "layers.0.mlp.switch_mlp.gate_proj",
            numExperts: 8,
            weight: ExpertTensor(file: 0, offset: 0, expertBytes: 64, shape: [4, 4], dtype: .float32))
        var slots = try ModuleSlots(layout: layout, slots: 0)
        #expect(slots.positions.count == 8)
        let mappedSlots = slots.mapRouted([5, 2, 5, 7], ids: [2, 5, 7], slots: [1, 0, 2])
        #expect(mappedSlots == [0, 1, 0, 2])
        let mappedRanks = slots.mapRouted([7, 2, 5], ids: [2, 5, 7], slots: nil)
        #expect(mappedRanks == [2, 0, 1])
    }
}
