import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("BailingHybrid (Ling-3.0) P1+P2: per-channel KDA Metal kernel and absorbed MLA latent cache (#672)", .serialized)
struct BailingHybridP1P2Tests {
    struct KDACase: Sendable {
        let label: String
        let batch: Int
        let steps: Int
        let hk: Int
        let hv: Int
        let dk: Int
        let dv: Int
        let dtype: DType
        let masked: Bool
        let tolerance: Float
    }

    static let kernelCases: [KDACase] = [
        KDACase(label: "tiny-f32", batch: 1, steps: 32, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .float32, masked: false, tolerance: 1e-4),
        KDACase(label: "tiny-f32-mask", batch: 2, steps: 20, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .float32, masked: true, tolerance: 1e-4),
        KDACase(label: "tiny-f16", batch: 1, steps: 24, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .float16, masked: false, tolerance: 2e-3),
        KDACase(label: "tiny-f16-mask", batch: 2, steps: 16, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .float16, masked: true, tolerance: 2e-3),
        KDACase(label: "tiny-bf16", batch: 1, steps: 24, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .bfloat16, masked: false, tolerance: 1e-2),
        KDACase(label: "tiny-bf16-mask", batch: 2, steps: 16, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .bfloat16, masked: true, tolerance: 1e-2),
        KDACase(label: "ling3-512-f32", batch: 1, steps: 512, hk: 16, hv: 16, dk: 128, dv: 128, dtype: .float32, masked: false, tolerance: 1e-4),
        KDACase(label: "ling3-decode-1-f32", batch: 1, steps: 1, hk: 16, hv: 16, dk: 128, dv: 128, dtype: .float32, masked: false, tolerance: 1e-4),
        KDACase(label: "ling3-decode-b2-mask", batch: 2, steps: 1, hk: 16, hv: 16, dk: 128, dv: 128, dtype: .float32, masked: true, tolerance: 1e-4),
        KDACase(label: "ling3-64-f32-mask", batch: 1, steps: 64, hk: 16, hv: 16, dk: 128, dv: 128, dtype: .float32, masked: true, tolerance: 1e-4),
        KDACase(label: "ling3-64-bf16", batch: 1, steps: 64, hk: 16, hv: 16, dk: 128, dv: 128, dtype: .bfloat16, masked: false, tolerance: 2e-2),
        KDACase(label: "gqa-asym-mask", batch: 2, steps: 16, hk: 2, hv: 4, dk: 64, dv: 32, dtype: .float32, masked: true, tolerance: 1e-4),
        KDACase(label: "small-dv", batch: 1, steps: 12, hk: 2, hv: 2, dk: 32, dv: 2, dtype: .float32, masked: false, tolerance: 1e-4),
        KDACase(label: "unaligned-dv", batch: 1, steps: 12, hk: 2, hv: 2, dk: 32, dv: 6, dtype: .float32, masked: true, tolerance: 1e-4),
    ]

    @Test("Per-channel KDA Metal kernel matches ops reference across Ling-3.0 [1,512,16,128], dtypes, masks, GQA, and small Dv",
          arguments: kernelCases)
    func kdaKernelMatchesOpsReference(spec: KDACase) throws {
        let (inputs, initState, mask) = Self.makeKDAFixture(spec)
        #expect(BailingHybridKDAKernel.canUseKernel(inputs))

        let (kernelOut, kernelState) = BailingHybridKDAKernel.run(inputs, state: initState, mask: mask)
        let (opsOut, opsState) = BailingHybridKDAOps.recurrentUpdateOps(inputs, state: initState, mask: mask)
        eval(kernelOut, kernelState, opsOut, opsState)

        #expect(kernelOut.shape == [spec.batch, spec.steps, spec.hv, spec.dv])
        #expect(kernelOut.dtype == spec.dtype)
        #expect(kernelState.shape == [spec.batch, spec.hv, spec.dv, spec.dk])
        #expect(kernelState.dtype == .float32)

        let outGap = abs(kernelOut.asType(.float32) - opsOut.asType(.float32)).max().item(Float.self)
        let stateGap = abs(kernelState - opsState).max().item(Float.self)
        #expect(outGap < spec.tolerance, "\(spec.label) output gap \(outGap) >= \(spec.tolerance)")
        #expect(stateGap < spec.tolerance, "\(spec.label) state gap \(stateGap) >= \(spec.tolerance)")
    }

    @Test("Per-channel KDA Metal kernel matches scalar Double CPU oracle within 1e-5 on Dk=32 masked sequence (#228)")
    func kdaKernelMatchesCPUOracle() throws {
        let (steps, heads, dim) = (8, 2, 32)
        let count = steps * heads * dim
        let sample = BailingHybridTests.CPURecurrenceCase(
            steps: steps,
            heads: heads,
            dim: dim,
            q: (0 ..< count).map { Float(($0 * 17) % 31 - 15) * 0.04 },
            k: (0 ..< count).map { Float(($0 * 13) % 29 - 14) * 0.04 },
            v: (0 ..< count).map { Float(($0 * 11) % 23 - 11) * 0.05 },
            decay: (0 ..< count).map { exp(Float(-0.2 - Double($0 % 7) * 0.25)) },
            beta: (0 ..< steps * heads).map { Float(0.2 + Double($0 % 5) * 0.15) },
            mask: [true, true, false, true, false, true, true, true]
        )
        let inputs = BailingHybridKDAInputs(
            q: MLXArray(sample.q, [1, steps, heads, dim]),
            k: MLXArray(sample.k, [1, steps, heads, dim]),
            v: MLXArray(sample.v, [1, steps, heads, dim]),
            decay: MLXArray(sample.decay, [1, steps, heads, dim]),
            beta: MLXArray(sample.beta, [1, steps, heads])
        )
        let mask = MLXArray(sample.mask, [1, steps])
        #expect(BailingHybridKDAKernel.canUseKernel(inputs, state: nil, mask: mask))
        let (outArr, stateArr) = BailingHybridKDAKernel.run(inputs, state: nil, mask: mask)
        eval(outArr, stateArr)
        let (cpuOut, cpuState) = BailingHybridTests.cpuKDARecurrence(sample)
        let maxOutErr = zip(outArr.asArray(Float.self), cpuOut).map { abs($0 - Float($1)) }.max() ?? 0
        let maxStateErr = zip(stateArr.asArray(Float.self), cpuState).map { abs($0 - Float($1)) }.max() ?? 0
        #expect(maxOutErr < 1e-5 && maxStateErr < 1e-5)
    }

    @Test("KDA Metal kernel handles non-contiguous inputs, empty slices, chunked state continuity, and falls back when Dk is not divisible by 32")
    func kdaKernelEdgeCasesAndFallback() throws {
        let full = Self.makeKDAFixture(KDACase(
            label: "chunked", batch: 1, steps: 16, hk: 4, hv: 4, dk: 32, dv: 32, dtype: .float32, masked: false, tolerance: 1e-4
        ))
        let sliced = BailingHybridKDAInputs(
            q: full.inputs.q.transposed(0, 2, 1, 3).transposed(0, 2, 1, 3),
            k: full.inputs.k.transposed(0, 2, 1, 3).transposed(0, 2, 1, 3),
            v: full.inputs.v.transposed(0, 2, 1, 3).transposed(0, 2, 1, 3),
            decay: full.inputs.decay.transposed(0, 2, 1, 3).transposed(0, 2, 1, 3),
            beta: full.inputs.beta
        )
        let (singleOut, singleState) = BailingHybridKDAKernel.run(sliced, state: full.state, mask: nil)
        var state = full.state
        var parts: [MLXArray] = []
        for start in stride(from: 0, to: 16, by: 4) {
            let chunk = BailingHybridKDAInputs(
                q: full.inputs.q[0..., start ..< (start + 4)],
                k: full.inputs.k[0..., start ..< (start + 4)],
                v: full.inputs.v[0..., start ..< (start + 4)],
                decay: full.inputs.decay[0..., start ..< (start + 4)],
                beta: full.inputs.beta[0..., start ..< (start + 4)]
            )
            let (out, next) = BailingHybridKDAKernel.run(chunk, state: state, mask: nil)
            parts.append(out)
            state = next
        }
        let steppedOut = concatenated(parts, axis: 1)
        #expect(abs(steppedOut - singleOut).max().item(Float.self) < 1e-4)
        #expect(abs(state - singleState).max().item(Float.self) < 1e-4)

        let empty = BailingHybridKDAInputs(
            q: MLXArray.zeros([1, 0, 4, 32]), k: MLXArray.zeros([1, 0, 4, 32]),
            v: MLXArray.zeros([1, 0, 4, 32]), decay: MLXArray.zeros([1, 0, 4, 32]),
            beta: MLXArray.zeros([1, 0, 4])
        )
        let (emptyOut, emptyState) = BailingHybridKDAKernel.run(empty, state: singleState, mask: nil)
        #expect(emptyOut.shape == [1, 0, 4, 32] && allClose(emptyState, singleState).item(Bool.self))

        let odd = Self.makeKDAFixture(KDACase(
            label: "odd-dk", batch: 1, steps: 6, hk: 2, hv: 2, dk: 24, dv: 24, dtype: .float32, masked: true, tolerance: 1e-5
        ))
        #expect(!BailingHybridKDAKernel.canUseKernel(odd.inputs, state: odd.state, mask: odd.mask))
        #expect(!BailingHybridKDAKernel.canUseKernel(full.inputs, state: MLXArray.zeros([1, 1, 1, 1]), mask: nil))
        #expect(!BailingHybridKDAKernel.canUseKernel(full.inputs, state: full.state, mask: MLXArray.zeros([1, 4], dtype: .bool)))
        let (fallbackOut, fallbackState) = BailingHybridKDAOps.recurrentUpdate(odd.inputs, state: odd.state, mask: odd.mask)
        let (opsOut, opsState) = BailingHybridKDAOps.recurrentUpdateOps(odd.inputs, state: odd.state, mask: odd.mask)
        #expect(abs(fallbackOut - opsOut).max().item(Float.self) < 1e-5)
        #expect(abs(fallbackState - opsState).max().item(Float.self) < 1e-5)
    }
}

extension BailingHybridP1P2Tests {
    @Test("Absorbed MLA matches non-absorbed MLA on float, 4-bit affine, SmallM, MXFP4, NVFP4 globalScale, LoRA-wrapped, and zero-nope configs")
    func absorbedMLAParityAcrossWeightModes() throws {
        let model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let mla = try #require(model.model.layers[2].attention as? BailingHybridMLAAttention)
        let hidden = Self.deterministicInput(batch: 1, length: 16, width: model.configuration.hiddenSize)
        try Self.assertMLAParity(mla, hidden: hidden, tolerance: 1e-4, expectAbsorbed: true, label: "float32")

        let affineModel = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let affineMLA = try #require(affineModel.model.layers[2].attention as? BailingHybridMLAAttention)
        quantize(model: affineMLA, groupSize: 32, bits: 4, mode: .affine) { path, module in
            path == "kv_b_proj" && module is Linear
        }
        #expect(affineMLA.kvBProj is QuantizedLinear)
        try Self.assertMLAParity(affineMLA, hidden: hidden, tolerance: 1e-4, expectAbsorbed: true, label: "affine-4bit")

        let smallMModel = try #require(try TinyModel.make(.bailingHybrid, adding: #""kv_lora_rank": 64, "#) as? BailingHybridModel)
        let smallMMLA = try #require(smallMModel.model.layers[2].attention as? BailingHybridMLAAttention)
        quantize(model: smallMMLA, groupSize: 32, bits: 4, mode: .affine) { path, module in
            path == "kv_b_proj" && module is Linear
        }
        #expect(SmallMQuantizedLinear.install(in: smallMMLA) == 1)
        #expect(smallMMLA.kvBProj is SmallMQuantizedLinear)
        try Self.assertMLAParity(smallMMLA, hidden: hidden, tolerance: 1e-4, expectAbsorbed: true, label: "smallm-4bit")

        let mxfp4Model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let mxfp4MLA = try #require(mxfp4Model.model.layers[2].attention as? BailingHybridMLAAttention)
        quantize(model: mxfp4MLA, groupSize: 32, bits: 4, mode: .mxfp4) { path, module in
            path == "kv_b_proj" && module is Linear
        }
        try Self.assertMLAParity(mxfp4MLA, hidden: hidden, tolerance: 1e-3, expectAbsorbed: true, label: "mxfp4")

        let nvfp4Model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let nvfp4MLA = try #require(nvfp4Model.model.layers[2].attention as? BailingHybridMLAAttention)
        let nvfp4Layer = QuantizedLinear(
            weight: nvfp4MLA.kvBProj.weight,
            bias: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: MLXArray(Float(1344.0))
        )
        nvfp4MLA.update(modules: ModuleChildren.unflattened([("kv_b_proj", nvfp4Layer)]))
        try Self.assertMLAParity(nvfp4MLA, hidden: hidden, tolerance: 1e-3, expectAbsorbed: true, label: "nvfp4-globalScale")

        let loraModel = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let loraMLA = try #require(loraModel.model.layers[2].attention as? BailingHybridMLAAttention)
        let baseOut = loraMLA(hidden, mask: .causal, cache: nil).asType(.float32)
        let lora = try #require(LoRALinear.from(linear: loraMLA.kvBProj, rank: 4, scale: 2.0) as? LoRALinear)
        let loraParams = Dictionary(uniqueKeysWithValues: lora.trainableParameters().flattened().enumerated().map { idx, entry in
            (entry.0, MLXRandom.normal(entry.1.shape, scale: 0.08, key: MLXRandom.key(UInt64(10 + idx))))
        })
        try lora.update(parameters: ModuleParameters.unflattened(loraParams), verify: .noUnusedKeys)
        loraMLA.update(modules: ModuleChildren.unflattened([("kv_b_proj", lora)]))
        let wrappedOut = loraMLA(hidden, mask: .causal, cache: nil).asType(.float32)
        #expect(abs(wrappedOut - baseOut).max().item(Float.self) > 1e-3)
        try Self.assertMLAParity(loraMLA, hidden: hidden, tolerance: 1e-4, expectAbsorbed: false, label: "lora-wrapped")

        let zeroNope = try #require(try TinyModel.make(.bailingHybrid, adding: #""qk_nope_head_dim": 0, "#) as? BailingHybridModel)
        let zeroNopeMLA = try #require(zeroNope.model.layers[2].attention as? BailingHybridMLAAttention)
        try Self.assertMLAParity(zeroNopeMLA, hidden: hidden, tolerance: 1e-4, expectAbsorbed: true, label: "zero-nope")
    }

    @Test("Absorbed MLA reduces cache memory ~8.9x at 128K on Ling-3.0, enables PrefillAttentionScratch, and refuses incompatible KV flags")
    func absorbedMemoryGeometryScratchAndRefusals() async throws {
        let model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let mla = try #require(model.model.layers[2].attention as? BailingHybridMLAAttention)
        let hidden = Self.deterministicInput(batch: 1, length: 20, width: model.configuration.hiddenSize)
        let absorbedCache = KVCacheSimple()
        let expandedCache = KVCacheSimple()
        _ = mla(hidden, mask: .causal, cache: absorbedCache)
        _ = mla.forwardNonAbsorbed(hidden, mask: .causal, cache: expandedCache)
        eval(absorbedCache.innerState() + expandedCache.innerState())
        let absorbedBytes = absorbedCache.state.reduce(0) { $0 + $1.nbytes }
        let expandedBytes = expandedCache.state.reduce(0) { $0 + $1.nbytes }
        #expect(absorbedBytes * 20 == expandedBytes * 3)
        #expect(model.kvHeads == [4, 4, 1, 4, 4, 1])

        let ling3JSON = try #require(
            try JSONSerialization.jsonObject(with: Data(BailingHybridTests.ling3TinyConfigJSON.utf8)) as? [String: Any]
        )
        let geometry = try #require(KVCacheGeometry(config: ling3JSON))
        #expect(geometry.bytesPerToken(nil) == 6_912)
        let kvAt128K = geometry.bytes(tokens: TokenCount(131_072)).rawValue - geometry.fixedStateBytes
        #expect(kvAt128K == 905_969_664)

        let scratchNoNAX = try #require(PrefillAttentionScratch.unfused(config: ling3JSON, nax: false))
        let scratchNAX = try #require(PrefillAttentionScratch.unfused(config: ling3JSON, nax: true))
        #expect(scratchNoNAX.heads == 16 && scratchNAX.heads == 16)
        #expect(scratchNAX.stepCap(keys: 32_768) != nil)
        let asymmetricJSON: [String: Any] = [
            "model_type": "bailing_hybrid", "num_attention_heads": 16, "head_dim": 128,
            "kv_lora_rank": 192, "qk_rope_head_dim": 64,
        ]
        #expect(PrefillAttentionScratch.unfused(config: asymmetricJSON, nax: true) == PrefillAttentionScratch(heads: 16))

        #expect(KEqVRowLayout(config: ling3JSON) == nil)
        do {
            try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: "bailing_hybrid", path: "ling3")
            Issue.record("Expected --kv-bits refusal on bailing_hybrid")
        } catch {
            #expect(error.localizedDescription.contains("absorbed MLA latent"))
        }

        let settings = GenerationPlan.Settings(
            kvPrecision: nil,
            kvZones: .standard,
            attendsThroughZones: true,
            prefillsInChunks: true,
            capabilities: try #require(ModelCapabilities.of(modelType: "bailing_hybrid")),
            kEqVRow: KEqVRowLayout(config: ling3JSON)
        )
        let plan = GenerationPlan.cache(precision: nil, settings)
        #expect(plan.kvZones == nil && plan.kEqVRow == nil)
    }

    @Test("Absorbed MLA matches non-absorbed MLA in bfloat16 across dense, 4-bit affine, MXFP4, and NVFP4 weights")
    func absorbedMLAParityInBFloat16() throws {
        let base = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let hidden = Self.deterministicInput(batch: 1, length: 16, width: base.configuration.hiddenSize).asType(.bfloat16)

        let denseMLA = try #require(base.model.layers[2].attention as? BailingHybridMLAAttention)
        denseMLA.update(parameters: denseMLA.parameters().mapValues { $0.asType(.bfloat16) })
        try Self.assertMLAParity(denseMLA, hidden: hidden, tolerance: 2e-2, expectAbsorbed: true, label: "bf16-dense")

        for (label, mode, groupSize, gs): (String, QuantizationMode, Int, MLXArray?) in [
            ("bf16-affine4", .affine, 32, nil),
            ("bf16-mxfp4", .mxfp4, 32, nil),
            ("bf16-nvfp4", .nvfp4, 16, MLXArray(Float(1344.0))),
        ] {
            let model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
            let mla = try #require(model.model.layers[2].attention as? BailingHybridMLAAttention)
            mla.update(parameters: mla.parameters().mapValues { $0.asType(.bfloat16) })
            let quantized = QuantizedLinear(
                weight: mla.kvBProj.weight,
                bias: nil,
                groupSize: groupSize,
                bits: 4,
                mode: mode,
                globalScale: gs
            )
            mla.update(modules: ModuleChildren.unflattened([("kv_b_proj", quantized)]))
            try Self.assertMLAParity(mla, hidden: hidden, tolerance: 3e-2, expectAbsorbed: true, label: label)
        }
    }
}

extension BailingHybridP1P2Tests {
    static func makeKDAFixture(_ spec: KDACase) -> (inputs: BailingHybridKDAInputs, state: MLXArray, mask: MLXArray?) {
        let qCount = spec.batch * spec.steps * spec.hk * spec.dk
        let vCount = spec.batch * spec.steps * spec.hv * spec.dv
        let gCount = spec.batch * spec.steps * spec.hv * spec.dk
        let bCount = spec.batch * spec.steps * spec.hv
        let sCount = spec.batch * spec.hv * spec.dv * spec.dk

        let q = MLXArray((0 ..< qCount).map { Float(($0 * 17) % 31 - 15) * 0.03 }, [spec.batch, spec.steps, spec.hk, spec.dk]).asType(spec.dtype)
        let k = MLXArray((0 ..< qCount).map { Float(($0 * 13) % 29 - 14) * 0.03 }, [spec.batch, spec.steps, spec.hk, spec.dk]).asType(spec.dtype)
        let v = MLXArray((0 ..< vCount).map { Float(($0 * 11) % 23 - 11) * 0.04 }, [spec.batch, spec.steps, spec.hv, spec.dv]).asType(spec.dtype)
        let decay = MLXArray((0 ..< gCount).map { exp(Float(-0.15 - Double($0 % 9) * 0.2)) }, [spec.batch, spec.steps, spec.hv, spec.dk])
        let beta = MLXArray((0 ..< bCount).map { Float(0.15 + Double($0 % 7) * 0.1) }, [spec.batch, spec.steps, spec.hv])
        let state = MLXArray((0 ..< sCount).map { Float(($0 * 7) % 19 - 9) * 0.02 }, [spec.batch, spec.hv, spec.dv, spec.dk])
        let mask: MLXArray? = spec.masked
            ? MLXArray((0 ..< spec.batch * spec.steps).map { ($0 % 5) != 2 }, [spec.batch, spec.steps])
            : nil
        return (BailingHybridKDAInputs(q: q, k: k, v: v, decay: decay, beta: beta), state, mask)
    }

    static func deterministicInput(batch: Int, length: Int, width: Int) -> MLXArray {
        MLXArray((0 ..< batch * length * width).map { Float(($0 * 19) % 37 - 18) * 0.05 }, [batch, length, width])
    }

    static func assertMLAParity(
        _ mla: BailingHybridMLAAttention,
        hidden: MLXArray,
        tolerance: Float,
        expectAbsorbed: Bool,
        label: String
    ) throws {
        #expect(BailingHybridMLAAbsorption.canAbsorb(mla.kvBProj) == expectAbsorbed)
        let fullAbsorbed = mla(hidden, mask: .causal, cache: nil).asType(.float32)
        let fullExpanded = mla.forwardNonAbsorbed(hidden, mask: .causal, cache: nil).asType(.float32)
        let fullGap = abs(fullAbsorbed - fullExpanded).max().item(Float.self)
        #expect(fullGap < tolerance, "\(label) full gap \(fullGap)")

        let absorbedCache = KVCacheSimple()
        let expandedCache = KVCacheSimple()
        var absorbedParts: [MLXArray] = []
        var expandedParts: [MLXArray] = []
        var cursor = 0
        let length = hidden.dim(1)
        for step in [5, 1, 3, 1, 6] where cursor < length {
            let end = min(length, cursor + step)
            let slice = hidden[0..., cursor ..< end, 0...]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (end - cursor) > 1 ? .causal : .none
            absorbedParts.append(mla(slice, mask: mask, cache: absorbedCache).asType(.float32))
            expandedParts.append(mla.forwardNonAbsorbed(slice, mask: mask, cache: expandedCache).asType(.float32))
            cursor = end
        }
        let chunkGap = abs(concatenated(absorbedParts, axis: 1) - concatenated(expandedParts, axis: 1)).max().item(Float.self)
        #expect(chunkGap < tolerance, "\(label) chunked gap \(chunkGap)")
        #expect(absorbedCache.state[0].shape == [1, 1, length, mla.kvLoraRank])
        #expect(absorbedCache.state[1].shape == [1, 1, length, mla.qkRopeHeadDim])
    }
}
