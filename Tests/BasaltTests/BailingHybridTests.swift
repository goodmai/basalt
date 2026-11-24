import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("BailingHybrid (Ling-3.0) P0: config, KDA math and noaux_tc router (#672)", .serialized)
struct BailingHybridTests {
    static let ling3TinyConfigJSON = """
        {"model_type": "bailing_hybrid", "hidden_size": 1536, "num_hidden_layers": 24,
         "intermediate_size": 4608, "num_attention_heads": 16, "num_key_value_heads": 16,
         "head_dim": 128, "rms_norm_eps": 1e-6, "vocab_size": 157184, "max_position_embeddings": 131072,
         "tie_word_embeddings": false, "layer_group_size": 4, "first_k_dense_replace": 1,
         "short_conv_kernel_size": 4, "no_kda_lora": true, "kda_safe_gate": true, "kda_lower_bound": -5,
         "q_lora_rank": 256, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
         "v_head_dim": 128, "rope_theta": 6000000, "rope_interleave": true,
         "gated_attention_proj_granularity_type": "head_wise", "num_experts": 128,
         "num_experts_per_tok": 8, "num_shared_experts": 1, "moe_intermediate_size": 512,
         "moe_shared_expert_intermediate_size": 512, "n_group": 8, "topk_group": 4,
         "norm_topk_prob": true, "routed_scaling_factor": 2.5, "moe_router_enable_expert_bias": true,
         "use_qk_norm": true}
        """

    @Test("Ling-3.0-tiny config decodes, computes MLA layer schedule with trailing remainders, validates, and reports exact KV geometry")
    func configurationAndGeometry() throws {
        let config = try JSONDecoder.json5().decode(BailingHybridConfiguration.self, from: Data(Self.ling3TinyConfigJSON.utf8)).validated()
        #expect(config.modelType == "bailing_hybrid")
        #expect(config.qkHeadDim == 192)
        #expect(config.convKernelSize == 4)
        #expect(config.noKDALoRA)
        #expect((0 ..< config.numHiddenLayers).filter { config.isMLALayer($0) } == [3, 7, 11, 15, 19, 23])
        #expect(!config.isMoELayer(0) && config.isMoELayer(1) && config.isMoELayer(23))

        var remainder = config
        remainder.numHiddenLayers = 7
        remainder.layerGroupSize = 3
        #expect((0 ..< 7).filter { remainder.isMLALayer($0) } == [2, 5, 6])

        let json = try #require(try JSONSerialization.jsonObject(with: Data(Self.ling3TinyConfigJSON.utf8)) as? [String: Any])
        let geometry = try #require(KVCacheGeometry(config: json))
        #expect(geometry.bytesPerToken(nil) == 61_440)
        #expect(geometry.fixedStateBytes == 19_537_920)
    }

    @Test("Configuration aliases decode conv_kernel_size, use_kda_lora and null granularity, and validation rejects invalid shapes")
    func configurationAliasesAndValidation() throws {
        let aliasJSON = """
            {"hidden_size": 64, "num_hidden_layers": 3, "intermediate_size": 128,
             "num_attention_heads": 4, "vocab_size": 256, "conv_kernel_size": 3,
             "use_kda_lora": true, "gated_attention_proj_granularity_type": null,
             "kv_lora_rank": 32, "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 16}
            """
        let alias = try JSONDecoder.json5().decode(BailingHybridConfiguration.self, from: Data(aliasJSON.utf8)).validated()
        #expect(alias.convKernelSize == 3 && !alias.noKDALoRA && alias.gatedAttentionGranularity == nil && alias.qLoraRank == nil)

        let mutations: [(inout BailingHybridConfiguration) -> Void] = [
            { $0.vocabSize = 0 },
            { $0.layerGroupSize = 0 },
            { $0.firstKDenseReplace = -1 },
            { $0.qLoraRank = 0 },
            { $0.qkRopeHeadDim = 15 },
            { $0.gatedAttentionGranularity = "unknown" },
            { $0.numExperts = -1 },
            { $0.numExperts = 8; $0.nGroup = 3; $0.topkGroup = 2; $0.numExpertsPerTok = 2 },
            { $0.numExperts = 8; $0.nGroup = 4; $0.topkGroup = 1; $0.numExpertsPerTok = 3 },
            { $0.numExperts = 8; $0.numExpertsPerTok = 0 },
        ]
        for mutate in mutations {
            var bad = alias
            mutate(&bad)
            #expect(throws: BasaltError.self) { _ = try bad.validated() }
        }
    }

    @Test("KDA per-channel safe gate, softplus fallback, and L2 norm match an explicit CPU Swift reference within 1e-5")
    func kdaSafeGateMatchesCPUReference() throws {
        let (batch, steps, heads, headDim) = (2, 3, 4, 8)
        let fValues: [Float] = (0 ..< batch * steps * heads * headDim).map { Float(($0 * 17) % 41 - 20) * 0.15 }
        let aLogValues: [Float] = [-0.8, 0.2, 1.1, -1.5]
        let dtBiasValues: [Float] = (0 ..< heads * headDim).map { Float(($0 * 7) % 19 - 9) * 0.2 }
        let lowerBound: Float = -5.0

        let fArr = MLXArray(fValues, [batch, steps, heads, headDim])
        let aLogArr = MLXArray(aLogValues, [heads])
        let dtBiasArr = MLXArray(dtBiasValues, [heads, headDim])

        let safeGot = BailingHybridKDAOps.decayLogGate(
            f: fArr, aLog: aLogArr, dtBias: dtBiasArr, safeGate: true, lowerBound: lowerBound
        ).asArray(Float.self)
        let softplusGot = BailingHybridKDAOps.decayLogGate(
            f: fArr, aLog: aLogArr, dtBias: dtBiasArr, safeGate: false, lowerBound: lowerBound
        ).asArray(Float.self)

        var (maxSafeErr, maxSoftplusErr): (Float, Float) = (0, 0)
        for idx in 0 ..< fValues.count {
            let h = (idx / headDim) % heads
            let d = idx % headDim
            let alpha = exp(Double(aLogValues[h]))
            let pre = Double(fValues[idx]) + Double(dtBiasValues[h * headDim + d])
            let expectedSafe = Float(Double(lowerBound) / (1.0 + exp(-(alpha * pre))))
            let expectedSoft = Float(-alpha * (pre > 20 ? pre : log(1.0 + exp(pre))))
            #expect(safeGot[idx] > lowerBound && safeGot[idx] < 0)
            maxSafeErr = max(maxSafeErr, abs(safeGot[idx] - expectedSafe))
            maxSoftplusErr = max(maxSoftplusErr, abs(softplusGot[idx] - expectedSoft))
        }
        #expect(maxSafeErr < 1e-5 && maxSoftplusErr < 1e-5)

        let normGot = BailingHybridKDAOps.l2Normalize(fArr).asArray(Float.self)
        var maxNormErr: Float = 0
        for row in 0 ..< (batch * steps * heads) {
            let base = row * headDim
            let sq = (0 ..< headDim).reduce(0.0) { $0 + Double(fValues[base + $1]) * Double(fValues[base + $1]) }
            let denom = sq.squareRoot() + 1e-6
            for d in 0 ..< headDim {
                maxNormErr = max(maxNormErr, abs(normGot[base + d] - Float(Double(fValues[base + d]) / denom)))
            }
        }
        #expect(maxNormErr < 1e-5)
    }

    @Test("KDA per-channel recurrent update and short conv match CPU reference, zero masked outputs, and handle empty slices")
    func kdaRecurrenceMatchesCPUReference() throws {
        let (batch, steps, heads, dim) = (1, 4, 2, 4)
        let sample = CPURecurrenceCase(
            steps: steps,
            heads: heads,
            dim: dim,
            q: (0 ..< batch * steps * heads * dim).map { Float(($0 * 13) % 23 - 11) * 0.1 },
            k: (0 ..< batch * steps * heads * dim).map { Float(($0 * 11) % 19 - 9) * 0.1 },
            v: (0 ..< batch * steps * heads * dim).map { Float(($0 * 7) % 17 - 8) * 0.15 },
            decay: (0 ..< batch * steps * heads * dim).map { exp(Float(-0.2 - Double($0 % 7) * 0.3)) },
            beta: (0 ..< batch * steps * heads).map { Float(0.2 + Double($0 % 5) * 0.15) },
            mask: [true, false, true, true]
        )
        let inputs = BailingHybridKDAInputs(
            q: MLXArray(sample.q, [batch, steps, heads, dim]),
            k: MLXArray(sample.k, [batch, steps, heads, dim]),
            v: MLXArray(sample.v, [batch, steps, heads, dim]),
            decay: MLXArray(sample.decay, [batch, steps, heads, dim]),
            beta: MLXArray(sample.beta, [batch, steps, heads])
        )
        let (outArr, stateArr) = BailingHybridKDAOps.recurrentUpdate(
            inputs, state: nil, mask: MLXArray(sample.mask, [batch, steps])
        )
        let (cpuOut, cpuState) = Self.cpuKDARecurrence(sample)
        let maxOutErr = zip(outArr.asArray(Float.self), cpuOut).map { abs($0 - Float($1)) }.max() ?? 0
        let maxStateErr = zip(stateArr.asArray(Float.self), cpuState).map { abs($0 - Float($1)) }.max() ?? 0
        #expect(maxOutErr < 1e-5 && maxStateErr < 1e-5)

        let emptyInputs = BailingHybridKDAInputs(
            q: MLXArray.zeros([1, 0, heads, dim]), k: MLXArray.zeros([1, 0, heads, dim]),
            v: MLXArray.zeros([1, 0, heads, dim]), decay: MLXArray.zeros([1, 0, heads, dim]),
            beta: MLXArray.zeros([1, 0, heads])
        )
        let (emptyOut, emptyState) = BailingHybridKDAOps.recurrentUpdate(emptyInputs, state: stateArr)
        #expect(emptyOut.shape == [1, 0, heads, dim])
        #expect(allClose(emptyState, stateArr).item(Bool.self))

        let shortConv = BailingHybridShortConv1d(channels: 8, kernelSize: 4)
        let (convEmpty, convState) = shortConv(MLXArray.zeros([1, 0, 8]), state: nil)
        #expect(convEmpty.shape == [1, 0, 8] && convState.shape == [1, 3, 8])
    }

    @Test("noaux_tc group router matches CPU reference for exact indices, unbiased weights, and expertMask under positive bias")
    func groupRouterMatchesCPUReference() throws {
        let logitsRows: [[Float]] = [
            [2.0, 1.5, -0.4, 0.2, 1.1, 0.9, -1.2, 0.5],
            [-3.0, -2.5, 4.0, 3.5, -0.5, -0.2, -0.1, -0.3],
            [0.4, -0.8, 1.2, 0.3, -1.5, 2.1, 0.7, -0.6],
        ]
        let bias: [Float] = [-2.0, -2.0, -5.0, -5.0, 0.1, 0.3, 0.2, 0.4]
        let config = BailingHybridRouter.Config(
            topK: 2, nGroup: 4, topkGroup: 2, normTopkProb: true, routedScalingFactor: 2.5, enableExpertBias: true
        )
        let (idxArr, weightArr) = BailingHybridRouter.route(
            logits: MLXArray(logitsRows.flatMap { $0 }, [1, logitsRows.count, 8]),
            expertBias: MLXArray(bias, [8]),
            config: config
        )
        let gotIdx = idxArr.asArray(Int32.self).map(Int.init)
        let gotWeights = weightArr.asArray(Float.self)
        for (token, row) in logitsRows.enumerated() {
            let (expectedIdx, expectedWeights) = Self.cpuNoauxTC(logits: row, bias: bias, config: config)
            let tokenIdx = Array(gotIdx[(token * 2) ..< ((token + 1) * 2)])
            let tokenWeights = Array(gotWeights[(token * 2) ..< ((token + 1) * 2)])
            #expect(tokenIdx == expectedIdx)
            #expect((zip(tokenWeights, expectedWeights).map { abs($0 - $1) }.max() ?? 0) < 1e-5)
            #expect(abs(tokenWeights.reduce(0, +) - 2.5) < 1e-5)
        }

        let noDropConfig = BailingHybridRouter.Config(
            topK: 2, nGroup: 4, topkGroup: 4, normTopkProb: true, routedScalingFactor: 2.5, enableExpertBias: false
        )
        let (noDropIdx, noDropW) = BailingHybridRouter.route(
            logits: MLXArray(logitsRows[0], [1, 1, 8]), expertBias: nil, config: noDropConfig
        )
        let (expIdx, expW) = Self.cpuNoauxTC(logits: logitsRows[0], bias: nil, config: noDropConfig)
        #expect(noDropIdx.asArray(Int32.self).map(Int.init) == expIdx)
        #expect((zip(noDropW.asArray(Float.self), expW).map { abs($0 - $1) }.max() ?? 0) < 1e-5)

        let hugeBias: [Float] = [50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let mask = MLXArray([Float](arrayLiteral: 0, 0, 1, 1, 1, 1, 1, 1), [8])
        let (maskedIdx, _) = BailingHybridRouter.route(
            logits: MLXArray(logitsRows[0], [1, 1, 8]),
            expertBias: MLXArray(hugeBias, [8]),
            config: config,
            expertMask: mask
        )
        let chosen = maskedIdx.asArray(Int32.self).map(Int.init)
        #expect(!chosen.contains(0) && !chosen.contains(1))
    }
}

extension BailingHybridTests {
    struct CPURecurrenceCase {
        let steps: Int
        let heads: Int
        let dim: Int
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let decay: [Float]
        let beta: [Float]
        let mask: [Bool]
    }

    static func cpuKDARecurrence(_ sample: CPURecurrenceCase) -> ([Double], [Double]) {
        let (steps, heads, dim) = (sample.steps, sample.heads, sample.dim)
        var state = Array(repeating: 0.0, count: heads * dim * dim)
        var out = Array(repeating: 0.0, count: steps * heads * dim)
        for t in 0 ..< steps where sample.mask[t] {
            for h in 0 ..< heads {
                let stateSlice = (h * dim * dim) ..< ((h + 1) * dim * dim)
                var updated = Array(state[stateSlice])
                for dv in 0 ..< dim {
                    var kvMem = 0.0
                    for dk in 0 ..< dim {
                        let idx = (t * heads + h) * dim + dk
                        updated[dv * dim + dk] *= Double(sample.decay[idx])
                        kvMem += updated[dv * dim + dk] * Double(sample.k[idx])
                    }
                    let delta = (Double(sample.v[(t * heads + h) * dim + dv]) - kvMem) * Double(sample.beta[t * heads + h])
                    for dk in 0 ..< dim {
                        updated[dv * dim + dk] += Double(sample.k[(t * heads + h) * dim + dk]) * delta
                    }
                    out[(t * heads + h) * dim + dv] = (0 ..< dim).reduce(0.0) { acc, dk in
                        acc + updated[dv * dim + dk] * Double(sample.q[(t * heads + h) * dim + dk])
                    }
                }
                state.replaceSubrange(stateSlice, with: updated)
            }
        }
        return (out, state)
    }

    static func cpuNoauxTC(
        logits: [Float],
        bias: [Float]?,
        config: BailingHybridRouter.Config
    ) -> ([Int], [Float]) {
        let rawScores = logits.map { Float(1.0 / (1.0 + exp(-Double($0)))) }
        let biased = (0 ..< logits.count).map { idx in
            config.enableExpertBias ? rawScores[idx] + (bias?[idx] ?? 0) : rawScores[idx]
        }
        let perGroup = logits.count / config.nGroup
        let groupScores = (0 ..< config.nGroup).map { group -> (Int, Float) in
            let slice = Array(biased[(group * perGroup) ..< ((group + 1) * perGroup)]).sorted(by: >)
            return (group, slice.prefix(min(2, perGroup)).reduce(0, +))
        }
        let keptGroups = Set(groupScores.sorted { $0.1 > $1.1 }.prefix(config.topkGroup).map(\.0))
        let candidates = (0 ..< logits.count).map { idx -> (Int, Float) in
            (idx, keptGroups.contains(idx / perGroup) ? biased[idx] : -Float.infinity)
        }
        let chosen = candidates.sorted { $0.1 > $1.1 }.prefix(config.topK).map(\.0)
        var weights = chosen.map { rawScores[$0] }
        if config.topK > 1, config.normTopkProb {
            let sum = weights.reduce(0, +) + 1e-20
            weights = weights.map { $0 / sum }
        }
        weights = weights.map { $0 * config.routedScalingFactor }
        return (chosen, weights)
    }
}
