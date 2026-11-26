import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Mamba-3 MIMO RoPE-SSM & exponential-trapezoidal discretization (#847)", .serialized)
struct Mamba3ModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 120, 4]

    struct CPUParitySpec: Sendable {
        let label: String
        let batch: Int
        let steps: Int
        let heads: Int
        let headDim: Int
        let dState: Int
        let mimoRank: Int
        let chunkSize: Int
        let heavyTail: Bool
        let masked: Bool
        let ropeFraction: Double
    }

    static let cpuParitySpecs: [CPUParitySpec] = [
        CPUParitySpec(label: "siso-heavy-chunked", batch: 1, steps: 10, heads: 2, headDim: 8, dState: 8, mimoRank: 1, chunkSize: 4, heavyTail: true, masked: false, ropeFraction: 1.0),
        CPUParitySpec(label: "siso-softplus-chunked", batch: 1, steps: 9, heads: 2, headDim: 8, dState: 8, mimoRank: 1, chunkSize: 3, heavyTail: false, masked: false, ropeFraction: 1.0),
        CPUParitySpec(label: "mimo-r4-heavy-chunked", batch: 2, steps: 10, heads: 2, headDim: 8, dState: 8, mimoRank: 4, chunkSize: 4, heavyTail: true, masked: false, ropeFraction: 0.5),
        CPUParitySpec(label: "mimo-r2-softplus-chunked", batch: 1, steps: 11, heads: 3, headDim: 6, dState: 6, mimoRank: 2, chunkSize: 5, heavyTail: false, masked: false, ropeFraction: 1.0),
        CPUParitySpec(label: "mimo-r2-masked", batch: 2, steps: 7, heads: 2, headDim: 6, dState: 8, mimoRank: 2, chunkSize: 4, heavyTail: true, masked: true, ropeFraction: 0.5),
    ]

    @Test("Mamba-3 chunked SSD and recurrent scan match the plain-Swift Double CPU oracle across SISO, MIMO, heavy_tail, and masks (#228)",
          arguments: cpuParitySpecs)
    func opsMatchCPUOracle(spec: CPUParitySpec) throws {
        let fixture = Self.makeCPUFixture(spec)
        let cpu = Mamba3CPUReference.forward(fixture.cpu)

        let (recOut, recState) = Mamba3Ops.recurrentScan(fixture.mlx, state: fixture.initState, mask: fixture.mask)
        let recGated = Self.applySkipGateAndReduce(recOut, fixture: fixture)
        eval(recGated, recState.angleState, recState.ssmState, recState.kState, recState.vState)
        Self.assertClose(recGated.asArray(Float.self), cpu.output, tolerance: 1e-4, label: "\(spec.label)-rec-out")
        Self.assertClose(recState.angleState.asArray(Float.self), cpu.state.angleState, tolerance: 1e-4, label: "\(spec.label)-rec-angle")
        Self.assertClose(recState.ssmState.asArray(Float.self), cpu.state.ssmState, tolerance: 1e-4, label: "\(spec.label)-rec-ssm")
        Self.assertClose(recState.kState.asArray(Float.self), cpu.state.kState, tolerance: 1e-4, label: "\(spec.label)-rec-k")
        Self.assertClose(recState.vState.asArray(Float.self), cpu.state.vState, tolerance: 1e-4, label: "\(spec.label)-rec-v")

        if !spec.masked {
            let (ssdOut, ssdState) = Mamba3Ops.chunkedSSD(fixture.mlx, state: fixture.initState, chunkSize: spec.chunkSize)
            let ssdGated = Self.applySkipGateAndReduce(ssdOut, fixture: fixture)
            eval(ssdGated, ssdState.angleState, ssdState.ssmState, ssdState.kState, ssdState.vState)
            Self.assertClose(ssdGated.asArray(Float.self), cpu.output, tolerance: 1e-4, label: "\(spec.label)-ssd-out")
            Self.assertClose(ssdState.angleState.asArray(Float.self), cpu.state.angleState, tolerance: 1e-4, label: "\(spec.label)-ssd-angle")
            Self.assertClose(ssdState.ssmState.asArray(Float.self), cpu.state.ssmState, tolerance: 1e-4, label: "\(spec.label)-ssd-ssm")
            Self.assertClose(ssdState.kState.asArray(Float.self), cpu.state.kState, tolerance: 1e-4, label: "\(spec.label)-ssd-k")
            Self.assertClose(ssdState.vState.asArray(Float.self), cpu.state.vState, tolerance: 1e-4, label: "\(spec.label)-ssd-v")
        }
    }

    @Test("Cumulative-angle RoPE is orthogonal within 1e-6 for both pairwise (SISO) and split-half (MIMO) pairings",
          arguments: [false, true])
    func ropeIsOrthogonal(splitHalf: Bool) throws {
        let dState = 16
        let half = dState / 2
        let u = (0 ..< dState).map { Double(($0 * 7) % 19 - 9) * 0.13 }
        let v = (0 ..< dState).map { Double(($0 * 11) % 23 - 11) * 0.11 }
        let phi = (0 ..< half).map { Double($0 + 1) * 0.37 - 1.1 }
        let cosV = phi.map(Foundation.cos)
        let sinV = phi.map(Foundation.sin)

        let uRot = Mamba3CPUReference.applyRoPE(u, cos: cosV, sin: sinV, splitHalf: splitHalf)
        let vRot = Mamba3CPUReference.applyRoPE(v, cos: cosV, sin: sinV, splitHalf: splitHalf)
        let uBack = Mamba3CPUReference.applyRoPE(uRot, cos: cosV, sin: sinV.map { -$0 }, splitHalf: splitHalf)

        let normBefore = sqrt(u.reduce(0) { $0 + $1 * $1 })
        let normAfter = sqrt(uRot.reduce(0) { $0 + $1 * $1 })
        let dotBefore = zip(u, v).reduce(0) { $0 + $1.0 * $1.1 }
        let dotAfter = zip(uRot, vRot).reduce(0) { $0 + $1.0 * $1.1 }
        let maxInvErr = zip(u, uBack).map { abs($0 - $1) }.max() ?? 0
        #expect(abs(normBefore - normAfter) < 1e-6)
        #expect(abs(dotBefore - dotAfter) < 1e-6)
        #expect(maxInvErr < 1e-6)

        let mlxU = MLXArray(u.map(Float.init), [1, 1, 1, 1, dState])
        let mlxCos = MLXArray(cosV.map(Float.init), [1, 1, 1, 1, half])
        let mlxSin = MLXArray(sinV.map(Float.init), [1, 1, 1, 1, half])
        let mlxRot = Mamba3Ops.applyRoPE(mlxU, cos: mlxCos, sin: mlxSin, splitHalf: splitHalf)
        let mlxBack = Mamba3Ops.applyRoPE(mlxRot, cos: mlxCos, sin: -mlxSin, splitHalf: splitHalf)
        eval(mlxRot, mlxBack)
        let mlxInvErr = MLX.abs(mlxBack - mlxU).max().item(Float.self)
        let cpuMlxGap = zip(mlxRot.asArray(Float.self), uRot).map { abs($0 - Float($1)) }.max() ?? 0
        #expect(mlxInvErr < 1e-6 && cpuMlxGap < 1e-6)

        if splitHalf {
            let d128 = (0 ..< 128).map { Double($0 + 1) * 0.03 }
            let phi32 = (0 ..< 32).map { Double($0 + 1) * 0.17 }
            let c32 = phi32.map(Foundation.cos)
            let s32 = phi32.map(Foundation.sin)
            let cpuPartial = Mamba3CPUReference.applyRoPE(d128, cos: c32, sin: s32, splitHalf: true)
            let mlxPartial = Mamba3Ops.applyRoPE(
                MLXArray(d128.map(Float.init), [1, 1, 1, 1, 128]),
                cos: MLXArray(c32.map(Float.init), [1, 1, 1, 1, 32]),
                sin: MLXArray(s32.map(Float.init), [1, 1, 1, 1, 32]),
                splitHalf: true
            ).asArray(Float.self)
            for i in 0 ..< 32 {
                let expLo = Float(d128[i] * c32[i] - d128[i + 64] * s32[i])
                let expHi = Float(d128[i] * s32[i] + d128[i + 64] * c32[i])
                #expect(abs(mlxPartial[i] - expLo) < 1e-5 && abs(mlxPartial[i + 64] - expHi) < 1e-5)
            }
            for i in 32 ..< 64 {
                #expect(mlxPartial[i] == Float(d128[i]) && mlxPartial[i + 64] == Float(d128[i + 64]))
            }
            let partialGap = zip(mlxPartial, cpuPartial).map { abs($0 - Float($1)) }.max() ?? 0
            #expect(partialGap < 1e-5)
        }
    }

    @Test("Full chunked SSD prefill matches 1-token recurrent decode and split prefill (5 + 7) within 1e-4 with exact argmax ids",
          arguments: [TinyModel.Family.mamba3Siso, .mamba3Mimo])
    func fullModelChunkedVsStepAndSplitPrefill(family: TinyModel.Family) throws {
        let model = try #require(try TinyModel.make(family) as? Mamba3Model)
        let tokenArray = MLXArray(Self.tokens)[.newAxis]

        let chunkedCache = model.newCache(parameters: nil)
        let fullChunked = model.forward(tokenArray, cache: chunkedCache, useChunked: true).asType(.float32)
        let fullRecurrent = model.forward(tokenArray, cache: nil, useChunked: false).asType(.float32)
        #expect(MLX.abs(fullChunked - fullRecurrent).max().item(Float.self) < 1e-4)
        #expect(argMax(fullChunked, axis: -1).asArray(Int32.self) == argMax(fullRecurrent, axis: -1).asArray(Int32.self))

        let stepCache = model.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: stepCache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - fullChunked).max().item(Float.self) < 1e-4)
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(fullChunked, axis: -1).asArray(Int32.self))

        let splitCache = model.newCache(parameters: nil)
        let partA = model(MLXArray(Array(Self.tokens[..<5]))[.newAxis], cache: splitCache).asType(.float32)
        let partB = model(MLXArray(Array(Self.tokens[5...]))[.newAxis], cache: splitCache).asType(.float32)
        let split = concatenated([partA, partB], axis: 1)
        #expect(MLX.abs(split - fullChunked).max().item(Float.self) < 1e-4)
        #expect(argMax(split, axis: -1).asArray(Int32.self) == argMax(fullChunked, axis: -1).asArray(Int32.self))

        for (a, b) in zip(chunkedCache, stepCache) {
            let ca = try #require(a as? ArraysCache)
            let cb = try #require(b as? ArraysCache)
            #expect(ca.offset == Self.tokens.count && cb.offset == Self.tokens.count)
            for slot in 0 ..< 4 {
                let sa = try #require(ca[slot])
                let sb = try #require(cb[slot])
                #expect(MLX.abs(sa - sb).max().item(Float.self) < 1e-4)
            }
        }
    }

    @Test("ArraysCache snapshot, PromptCacheDiskStore round-trip, and exact O(1) RecurrentSnapshot bytes invariance between T=4 and T=64",
          arguments: [TinyModel.Family.mamba3Siso, .mamba3Mimo])
    func promptCacheDiskRoundTripAndConstantFootprint(family: TinyModel.Family) throws {
        let model = try #require(try TinyModel.make(family) as? Mamba3Model)
        let expectedFootprint = model.configuration.recurrentStateFootprint()
        #expect(expectedFootprint.bytes == model.configuration.recurrentStateBytes)

        let shortPrompt = (0 ..< 5).map { ($0 * 7 + 3) % 200 }
        let longPrompt = (0 ..< 65).map { ($0 * 11 + 5) % 200 }
        let shortSnap = try TinyHybrid.snapshotted(model, shortPrompt)
        let longSnap = try TinyHybrid.snapshotted(model, longPrompt)
        let shortRec = try #require(RecurrentSnapshot(of: shortSnap))
        let longRec = try #require(RecurrentSnapshot(of: longSnap))
        #expect(shortRec.bytes == expectedFootprint.bytes)
        #expect(longRec.bytes == expectedFootprint.bytes)
        #expect(shortRec.bytes == longRec.bytes)

        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.afterToolCall)
        let boundary = first.count - 1
        let snap = try TinyHybrid.snapshotted(model, first)
        #expect(snap.isHybrid && snap.isSnapshot(of: boundary))

        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.checkedStore("ns", tokens: Array(first.prefix(boundary)), cache: snap)
        let lease = try tree.required("ns", for: second)
        let tail = model(TinyHybrid.row(Array(second[boundary...])), cache: lease.cache.layers)
        eval(tail)
        _ = consume lease
        let expected = model(TinyHybrid.row(second), cache: model.newCache(parameters: nil))[0..., boundary..., 0...]
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = Array(first.prefix(boundary))
        let memLease = try tree.required("ns", for: first)
        var writer = DiskFixture.store(directory)
        #expect(writer.store(fingerprint: "fp", model: "m3", namespace: "ns", tokens: prefix, cache: memLease.cache) != nil)
        var reader = DiskFixture.store(directory)
        let layout = model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName)
        let entry = try #require(reader.longestPrefix(fingerprint: "fp", namespace: "ns", of: prefix, longerThan: 0, maxBytes: ByteCount(1 << 26))?.entry)
        let loaded = try reader.load(entry, layout: layout)
        #expect(loaded.isSnapshot(of: prefix.count))
        let fromDisk = DiskFixture.nextLogits(model, first[boundary], loaded)
        let fromMemory = DiskFixture.nextLogits(model, first[boundary], memLease.cache)
        #expect(DiskFixture.identical(fromDisk, fromMemory))
        _ = consume memLease
    }
}

extension Mamba3ModelTests {
    @Test("Verbatim state-spaces/mamba3-mimo-187m config.json decodes exact shapes and sanitizes 128-rounded fc1/fc2")
    func upstream187mConfigAndSanitizeLayout() throws {
        let upstream187mJSON = """
            {"d_model": 768, "d_intermediate": 1264, "n_layer": 12, "vocab_size": 128256,
             "ssm_cfg": {"layer": "Mamba3", "d_state": 128, "expand": 2, "headdim": 64, "ngroups": 1,
                         "rope_fraction": 0.5, "dt_min": 0.001, "dt_max": 0.1, "dt_init_floor": 0.0001,
                         "A_floor": 0.0001, "chunk_size": 16, "is_mimo": true, "mimo_rank": 4, "is_outproj_norm": false},
             "attn_layer_idx": [], "attn_cfg": {}, "rms_norm": true, "residual_in_fp32": true,
             "fused_add_norm": false, "pad_vocab_size_multiple": 16, "tie_embeddings": true}
            """
        let upstream187m = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: Data(upstream187mJSON.utf8)).validated()
        #expect(upstream187m.numHiddenLayers == 12 && upstream187m.vocabSize == 128_256 && upstream187m.tieWordEmbeddings)
        #expect(upstream187m.inProjDim == 4200 && upstream187m.numRopeAngles == 32 && upstream187m.intermediateSize == 1280)
        #expect(upstream187m.useGatedMLP && upstream187m.isMIMO && !upstream187m.isOutprojNorm)

        var singleLayer = upstream187m
        singleLayer.numHiddenLayers = 1
        singleLayer.vocabSize = 64
        let model = Mamba3Model(singleLayer)
        let sanitized = model.sanitize(weights: [
            "backbone.embedding.weight": MLXArray.zeros([64, 768]),
            "backbone.norm_f.weight": MLXArray.ones([768]),
            "backbone.layers.0.norm.weight": MLXArray.ones([768]),
            "backbone.layers.0.mixer.in_proj.weight": MLXArray.zeros([4200, 768]),
            "backbone.layers.0.mixer.out_proj.weight": MLXArray.zeros([768, 1536]),
            "backbone.layers.0.mixer.B_norm.weight": MLXArray.ones([128]),
            "backbone.layers.0.mixer.C_norm.weight": MLXArray.ones([128]),
            "backbone.layers.0.mixer.dt_bias": MLXArray.zeros([24]),
            "backbone.layers.0.mixer.D": MLXArray.ones([24]),
            "backbone.layers.0.mixer.B_bias": MLXArray.ones([24, 4, 128]),
            "backbone.layers.0.mixer.C_bias": MLXArray.ones([24, 4, 128]),
            "backbone.layers.0.mixer.mimo_x": MLXArray.ones([24, 4, 64]),
            "backbone.layers.0.mixer.mimo_z": MLXArray.ones([24, 4, 64]),
            "backbone.layers.0.mixer.mimo_o": MLXArray.ones([24, 4, 64]),
            "backbone.layers.0.norm2.weight": MLXArray.ones([768]),
            "backbone.layers.0.mlp.fc1.weight": MLXArray.zeros([2560, 768]),
            "backbone.layers.0.mlp.fc2.weight": MLXArray.zeros([768, 1280]),
        ])
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        let logits = model(MLXArray([Int32(1), 2])[.newAxis], cache: nil)
        #expect(logits.shape == [1, 2, 64])
    }

    @Test("Config decoding handles HF/FLA, state-spaces ssm_cfg, KVCacheGeometry, optional no-MLP/tied variants, and rejects invalid configs")
    func configGeometryVariantsAndSanitize() throws {
        let nativeJSON = """
            {"model_type": "mamba_3", "vocab_size": 512, "d_model": 64, "n_layer": 2, "d_intermediate": 0,
             "ssm_cfg": {"expand": 2, "d_state": 16, "headdim": 16, "ngroups": 2, "mimo_rank": 2, "chunk_size": 16, "heavy_tail": false, "rope_fraction": 0.5, "A_floor": 1e-4, "is_outproj_norm": true},
             "norm_eps": 1e-6, "tie_embeddings": true}
            """
        let native = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: Data(nativeJSON.utf8)).validated()
        #expect(native.hiddenSize == 64 && native.numHiddenLayers == 2 && native.numHeads == 8)
        #expect(native.mimoRank == 2 && !native.heavyTail && !native.useGatedMLP && native.tieWordEmbeddings)
        #expect(native.ropeDim == 8 && native.numRopeAngles == 4 && native.isOutprojNorm)

        let sisoForcedJSON = #"{"model_type": "mamba3", "vocab_size": 256, "d_model": 64, "n_layer": 2, "ssm_cfg": {"d_state": 16, "headdim": 16, "mimo_rank": 4, "is_mimo": false}}"#
        let sisoForced = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: Data(sisoForcedJSON.utf8)).validated()
        #expect(sisoForced.mimoRank == 1 && !sisoForced.isMIMO)

        let modelNoMLP = Mamba3Model(native)
        #expect(modelNoMLP.lmHead == nil && modelNoMLP.model.layers[0].mlp == nil && modelNoMLP.model.layers[0].mixer.norm != nil)
        let full = modelNoMLP(MLXArray(Array(Self.tokens.prefix(6)))[.newAxis], cache: nil).asType(.float32)
        let cache = modelNoMLP.newCache(parameters: nil)
        let stepped = concatenated(Self.tokens.prefix(6).map { modelNoMLP(MLXArray([$0])[.newAxis], cache: cache).asType(.float32) }, axis: 1)
        #expect(MLX.abs(stepped - full).max().item(Float.self) < 1e-4)

        let jsonObj = try #require(try JSONSerialization.jsonObject(with: Data(TinyModel.Family.mamba3Mimo.config.utf8)) as? [String: Any])
        let geom = try #require(KVCacheGeometry(config: jsonObj))
        let mimoCfg = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: Data(TinyModel.Family.mamba3Mimo.config.utf8)).validated()
        #expect(geom.elementsPerToken == 0 && geom.fixedStateBytes == mimoCfg.recurrentStateFootprint().bytes.rawValue)

        let badConfigs = [
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "d_state": 15, "head_dim": 16}"#,
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "d_state": 16, "head_dim": 12}"#,
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "d_state": 16, "head_dim": 16, "n_groups": 3}"#,
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "d_state": 16, "head_dim": 16, "mimo_rank": 0}"#,
        ]
        for bad in badConfigs {
            #expect(throws: BasaltError.self) {
                _ = try Mamba3Model(configurationData: Data(bad.utf8))
            }
        }

        let siso = try #require(try TinyModel.make(.mamba3Siso, adding: #""tie_word_embeddings": true, "#) as? Mamba3Model)
        let raw: [String: MLXArray] = [
            "backbone.embedding.weight": MLXArray.zeros([256, 64]),
            "backbone.norm_f.weight": MLXArray.ones([64]),
            "lm_head.weight": MLXArray.ones([256, 64]),
            "backbone.layers.0.mlp.fc1.weight": concatenated([MLXArray.ones([128, 64]), MLXArray.zeros([128, 64])], axis: 0),
            "backbone.layers.0.mlp.fc2.weight": MLXArray.ones([64, 128]),
            "backbone.layers.0.mixer.B_bias": MLXArray.ones([8, 16]),
            "backbone.layers.0.mixer.mimo_x": MLXArray.ones([8, 1, 16]),
            "backbone.layers.9.mixer.in_proj.weight": MLXArray.zeros([16, 64]),
        ]
        let sanitized = siso.sanitize(weights: raw)
        #expect(sanitized["model.embed_tokens.weight"] != nil && sanitized["model.norm.weight"] != nil)
        #expect(sanitized["lm_head.weight"] == nil && sanitized["model.layers.0.mixer.mimo_x"] == nil)
        #expect(sanitized["model.layers.9.mixer.in_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.mlp.up_proj.weight"]?.sum().item(Float.self) == Float(128 * 64))
        #expect(sanitized["model.layers.0.mlp.gate_proj.weight"]?.sum().item(Float.self) == 0)
        #expect(sanitized["model.layers.0.mlp.down_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.mixer.B_bias"]?.shape == [8, 1, 16])
    }

    @Test("LLMTypeRegistry registers mamba3 aliases and ModelCapabilities classifies them as recurrent")
    func registryAndCapabilitiesSupportMamba3() async throws {
        await MLXInferenceEngine.registerAliases()
        for alias in ["mamba3", "mamba_3", "mamba-3"] {
            #expect(await LLMTypeRegistry.shared.contains(alias))
            let created = try await LLMTypeRegistry.shared.createModel(
                configuration: Data(TinyModel.Family.mamba3Siso.config.utf8),
                modelType: alias
            )
            #expect(created is Mamba3Model)
            let cap = try #require(ModelCapabilities.of(modelType: alias))
            #expect(cap.allows(.promptCacheReuse) && cap.allows(.grammarJumpForward) && !cap.allows(.kvQuantization))
            #expect(throws: BasaltError.self) {
                try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: alias, path: "tiny-mamba3")
            }
        }
        let engine = try await TinyModel.engine(.mamba3Siso)
        let container = try #require(await engine.container)
        let epoch = await engine.modelEpoch
        #expect(await engine.cacheIsHybrid(container, epoch: epoch))
        #expect(await engine.modelHasRecurrentState == true)
    }
}

extension Mamba3ModelTests {
    struct CPUFixtureBundle {
        let cpu: Mamba3CPUInputs
        let mlx: Mamba3RecurrenceInputs
        let initState: Mamba3State
        let mask: MLXArray?
        let z: MLXArray
        let dSkip: MLXArray
        let mimoZ: MLXArray?
        let mimoO: MLXArray?
    }

    static func makeCPUFixture(_ spec: CPUParitySpec) -> CPUFixtureBundle {
        let (b, t, h, p, n, r) = (spec.batch, spec.steps, spec.heads, spec.headDim, spec.dState, spec.mimoRank)
        let shape = Mamba3CPUShape(
            batch: b, steps: t, heads: h, headDim: p, dState: n, mimoRank: r, ropeFraction: spec.ropeFraction)
        let half = shape.numRopeAngles
        let xArr = (0 ..< b * t * h * p).map { Float(($0 * 17) % 31 - 15) * 0.04 }
        let zArr = (0 ..< b * t * h * p).map { Float(($0 * 13) % 29 - 14) * 0.05 }
        let bArr = (0 ..< b * t * h * r * n).map { Float(($0 * 11) % 23 - 11) * 0.05 }
        let cArr = (0 ..< b * t * h * r * n).map { Float(($0 * 7) % 19 - 9) * 0.05 }
        let ddDtArr = (0 ..< b * t * h).map { Float(($0 * 5) % 13 - 6) * 0.1 }
        let ddAArr = (0 ..< b * t * h).map { Float(($0 * 3) % 11 - 5) * 0.15 }
        let trapArr = (0 ..< b * t * h).map { Float(($0 * 7) % 17 - 8) * 0.12 }
        let angArr = (0 ..< b * t * h * half).map { Float(($0 * 19) % 29 - 14) * 0.06 }

        let dtBias = (0 ..< h).map { Float($0) * 0.1 - 0.1 }
        let aFloor: Float = 1e-4
        let dSkip = (0 ..< h).map { 0.4 + Float($0) * 0.15 }
        let mimoX: [Float]? = r > 1 ? (0 ..< h * r * p).map { Float(($0 * 7) % 13 + 1) * 0.06 } : nil
        let mimoZ: [Float]? = r > 1 ? (0 ..< h * r * p).map { Float(($0 * 5) % 11 + 1) * 0.07 } : nil
        let mimoO: [Float]? = r > 1 ? (0 ..< h * r * p).map { Float(($0 * 3) % 9 + 1) * 0.08 } : nil

        let cpuState = Mamba3CPUState(
            angleState: (0 ..< b * h * half).map { Float(($0 * 3) % 11 - 5) * 0.05 },
            ssmState: (0 ..< b * h * p * n).map { Float(($0 * 5) % 13 - 6) * 0.02 },
            kState: (0 ..< b * h * r * n).map { Float(($0 * 7) % 17 - 8) * 0.03 },
            vState: (0 ..< b * h * r * p).map { Float(($0 * 11) % 19 - 9) * 0.03 }
        )
        let maskArr: [Bool]? = spec.masked ? (0 ..< b * t).map { ($0 % 4) != 1 } : nil
        let params = Mamba3CPUParameters(dtBias: dtBias, aFloor: aFloor, dSkip: dSkip, mimoX: mimoX, mimoZ: mimoZ, mimoO: mimoO)
            .withHeavyTail(spec.heavyTail)
        let cpuInputs = Mamba3CPUInputs(shape: shape, x: xArr, z: zArr, b: bArr, c: cArr, parameters: params)
            .withControls(ddDt: ddDtArr, ddA: ddAArr, trap: trapArr, angles: angArr, initialState: cpuState, mask: maskArr)
        return buildMLXBundle(spec: spec, cpu: cpuInputs, cpuState: cpuState)
    }

    private static func buildMLXBundle(
        spec: CPUParitySpec,
        cpu: Mamba3CPUInputs,
        cpuState: Mamba3CPUState
    ) -> CPUFixtureBundle {
        let (b, t, h, p, n, r) = (spec.batch, spec.steps, spec.heads, spec.headDim, spec.dState, spec.mimoRank)
        let half = cpu.shape.numRopeAngles
        let xMLX = MLXArray(cpu.x, [b, t, h, p])
        let vMLX: MLXArray
        if let mx = cpu.parameters.mimoX {
            vMLX = xMLX.expandedDimensions(axis: 3) * MLXArray(mx, [1, 1, h, r, p])
        } else {
            vMLX = xMLX.expandedDimensions(axis: 3)
        }
        let dtMLX = softplus(MLXArray(cpu.ddDt, [b, t, h]) + MLXArray(cpu.parameters.dtBias, [1, 1, h]))
        let ddAMLX = MLXArray(cpu.ddA, [b, t, h])
        let aScale = spec.heavyTail ? Mamba3Ops.heavyTail(ddAMLX) : softplus(ddAMLX)
        let logAlphaMLX = minimum(-aScale, MLXArray(-cpu.parameters.aFloor, dtype: .float32)) * dtMLX
        let trapMLX = sigmoid(MLXArray(cpu.trap, [b, t, h]))
        let mlxInputs = Mamba3RecurrenceInputs(
            q: MLXArray(cpu.c, [b, t, h, r, n]),
            k: MLXArray(cpu.b, [b, t, h, r, n]),
            v: vMLX,
            dt: dtMLX,
            logAlpha: logAlphaMLX,
            trap: trapMLX
        ).withAngles(MLXArray(cpu.angles, [b, t, h, half]), splitHalfRoPE: r > 1)
        let mlxState = Mamba3State(
            angleState: MLXArray(cpuState.angleState, [b, h, half]),
            ssmState: MLXArray(cpuState.ssmState, [b, h, p, n]),
            kState: MLXArray(cpuState.kState, [b, h, r, n]),
            vState: MLXArray(cpuState.vState, [b, h, r, p])
        )
        return CPUFixtureBundle(
            cpu: cpu,
            mlx: mlxInputs,
            initState: mlxState,
            mask: cpu.mask.map { MLXArray($0, [b, t]) },
            z: MLXArray(cpu.z, [b, t, h, p]),
            dSkip: MLXArray(cpu.parameters.dSkip, [1, 1, h, 1, 1]),
            mimoZ: cpu.parameters.mimoZ.map { MLXArray($0, [1, 1, h, r, p]) },
            mimoO: cpu.parameters.mimoO.map { MLXArray($0, [1, 1, h, r, p]) }
        )
    }

    static func applySkipGateAndReduce(_ y: MLXArray, fixture: CPUFixtureBundle) -> MLXArray {
        var withSkip = y + fixture.dSkip * fixture.mlx.v
        if let mask = fixture.mask {
            withSkip = MLX.where(mask.expandedDimensions(axes: [2, 3, 4]), withSkip, MLXArray.zeros(like: withSkip))
        }
        if let mz = fixture.mimoZ, let mo = fixture.mimoO {
            let gated = withSkip * silu(fixture.z.expandedDimensions(axis: 3) * mz)
            return (gated * mo).sum(axis: 3)
        }
        return withSkip.squeezed(axis: 3) * silu(fixture.z)
    }

    static func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float, label: String) {
        #expect(actual.count == expected.count, "\(label) count mismatch \(actual.count) vs \(expected.count)")
        let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDiff < tolerance, "\(label) max diff \(maxDiff) >= \(tolerance)")
    }
}
