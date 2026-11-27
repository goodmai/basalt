import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Gated DeltaNet-2 decoupled erase/write, non-expansive beta_guard, chunked WY, and fused Metal kernel (#847)", .serialized)
struct GatedDeltaNet2ModelTests {
    static let tokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 120, 4, 55, 11]

    struct KernelParitySpec: Sendable {
        let label: String
        let batch: Int
        let steps: Int
        let hk: Int
        let hv: Int
        let dk: Int
        let dv: Int
        let dtype: DType
        let allowNegEigval: Bool
        let betaGuard: Bool
        let masked: Bool
        let tolerance: Float
    }

    static let paritySpecs: [KernelParitySpec] = [
        KernelParitySpec(label: "f32-gva-guard-pos", batch: 1, steps: 12, hk: 2, hv: 4, dk: 32, dv: 32, dtype: .float32, allowNegEigval: false, betaGuard: true, masked: false, tolerance: 1e-4),
        KernelParitySpec(label: "f32-gva-guard-neg", batch: 2, steps: 10, hk: 2, hv: 4, dk: 32, dv: 32, dtype: .float32, allowNegEigval: true, betaGuard: true, masked: false, tolerance: 1e-4),
        KernelParitySpec(label: "f32-gva-noguard-neg", batch: 1, steps: 10, hk: 2, hv: 4, dk: 64, dv: 32, dtype: .float32, allowNegEigval: true, betaGuard: false, masked: false, tolerance: 1e-4),
        KernelParitySpec(label: "f32-masked-guard", batch: 2, steps: 8, hk: 2, hv: 4, dk: 32, dv: 32, dtype: .float32, allowNegEigval: true, betaGuard: true, masked: true, tolerance: 1e-4),
        KernelParitySpec(label: "f32-unaligned-dv", batch: 1, steps: 9, hk: 2, hv: 2, dk: 32, dv: 6, dtype: .float32, allowNegEigval: true, betaGuard: true, masked: true, tolerance: 1e-4),
        KernelParitySpec(label: "f16-gva-guard", batch: 1, steps: 10, hk: 2, hv: 4, dk: 32, dv: 32, dtype: .float16, allowNegEigval: true, betaGuard: true, masked: false, tolerance: 5e-3),
        KernelParitySpec(label: "bf16-gva-guard", batch: 1, steps: 10, hk: 2, hv: 4, dk: 32, dv: 32, dtype: .bfloat16, allowNegEigval: true, betaGuard: true, masked: false, tolerance: 2e-2),
        KernelParitySpec(label: "bf16-real-d128-guard", batch: 1, steps: 6, hk: 2, hv: 4, dk: 128, dv: 128, dtype: .bfloat16, allowNegEigval: true, betaGuard: true, masked: false, tolerance: 2e-2),
        KernelParitySpec(label: "f32-real-d128-noguard", batch: 1, steps: 6, hk: 2, hv: 4, dk: 128, dv: 128, dtype: .float32, allowNegEigval: true, betaGuard: false, masked: false, tolerance: 1e-4),
    ]

    @Test("GDN-2 Metal kernel, pure-MLX recurrent, chunked WY, and plain-Swift CPU oracle agree across beta_guard, allow_neg_eigval, GVA, masks, and dtypes (#228)",
          arguments: paritySpecs)
    func fourWayKernelParity(spec: KernelParitySpec) throws {
        let fixture = Self.makeFixture(spec)
        #expect(GatedDeltaNet2Kernel.canUseKernel(fixture.mlx, state: fixture.initState, mask: fixture.mask))

        let (metalOut, metalState) = GatedDeltaNet2Kernel.run(fixture.mlx, state: fixture.initState, mask: fixture.mask)
        let (opsOut, opsState) = GatedDeltaNet2Ops.recurrentOps(fixture.mlx, state: fixture.initState, mask: fixture.mask)
        eval(metalOut, metalState, opsOut, opsState)

        #expect(metalOut.shape == [spec.batch, spec.steps, spec.hv, spec.dv] && metalOut.dtype == spec.dtype)
        #expect(metalState.shape == [spec.batch, spec.hv, spec.dv, spec.dk] && metalState.dtype == .float32)

        let mOpsOutGap = MLX.abs(metalOut.asType(.float32) - opsOut.asType(.float32)).max().item(Float.self)
        let mOpsStateGap = MLX.abs(metalState - opsState).max().item(Float.self)
        #expect(mOpsOutGap < spec.tolerance, "\(spec.label) metal vs ops out gap \(mOpsOutGap)")
        #expect(mOpsStateGap < spec.tolerance, "\(spec.label) metal vs ops state gap \(mOpsStateGap)")

        if !spec.masked {
            for chunkSize in [4, 64] {
                let (wyOut, wyState) = GatedDeltaNet2Ops.chunkedWY(fixture.mlx, state: fixture.initState, chunkSize: chunkSize)
                eval(wyOut, wyState)
                let wyOutGap = MLX.abs(wyOut.asType(.float32) - metalOut.asType(.float32)).max().item(Float.self)
                let wyStateGap = MLX.abs(wyState - metalState).max().item(Float.self)
                #expect(wyOutGap < spec.tolerance, "\(spec.label) WY(\(chunkSize)) vs metal out gap \(wyOutGap)")
                #expect(wyStateGap < spec.tolerance, "\(spec.label) WY(\(chunkSize)) vs metal state gap \(wyStateGap)")
            }
        }

        if spec.dtype == .float32 {
            let cpu = GatedDeltaNet2CPUReference.forward(fixture.cpu)
            let cpuOutGap = zip(metalOut.asArray(Float.self), cpu.output).map { abs($0 - $1) }.max() ?? 0
            let cpuStateGap = zip(metalState.asArray(Float.self), cpu.state).map { abs($0 - $1) }.max() ?? 0
            #expect(cpuOutGap < 1e-4, "\(spec.label) metal vs CPU out gap \(cpuOutGap)")
            #expect(cpuStateGap < 1e-4, "\(spec.label) metal vs CPU state gap \(cpuStateGap)")
        }
    }

    @Test("beta_guard enforces ||(I - k (b * k)^T) Diag(e^g)||_2 <= 1 when expansive and leaves contractive or uniform rows unchanged")
    func betaGuardEnforcesNonExpansiveSpectralNorm() throws {
        let dk = 32
        let invSqrt = 1.0 / Double(dk).squareRoot()
        let key = [Double](repeating: invSqrt, count: dk)
        let expansiveErase = (0 ..< dk).map { $0 < dk / 2 ? 2.0 : 0.0 }
        let weakDecay = [Double](repeating: 0.95, count: dk)

        let rawNorm = GatedDeltaNet2CPUReference.transitionSpectralNorm(erase: expansiveErase, key: key, decay: weakDecay)
        #expect(rawNorm > 1.2)
        let guarded = GatedDeltaNet2CPUReference.applyBetaGuard(erase: expansiveErase, key: key, decay: weakDecay)
        let guardedNorm = GatedDeltaNet2CPUReference.transitionSpectralNorm(erase: guarded, key: key, decay: weakDecay)
        #expect(guardedNorm <= 1.0 + 1e-5)

        let mlxGuarded = GatedDeltaNet2Ops.applyBetaGuard(
            erase: MLXArray(expansiveErase.map(Float.init), [1, 1, 1, dk]),
            key: MLXArray(key.map(Float.init), [1, 1, 1, dk]),
            decay: MLXArray(weakDecay.map(Float.init), [1, 1, 1, dk])
        )
        eval(mlxGuarded)
        let guardGap = zip(mlxGuarded.asArray(Float.self), guarded).map { abs($0 - Float($1)) }.max() ?? 0
        #expect(guardGap < 1e-5)

        let strongDecay = [Double](repeating: 0.5, count: dk)
        let untouched = GatedDeltaNet2CPUReference.applyBetaGuard(erase: expansiveErase, key: key, decay: strongDecay)
        #expect(untouched == expansiveErase)

        let uniform = [Double](repeating: 1.75, count: dk)
        let uniformGuarded = GatedDeltaNet2CPUReference.applyBetaGuard(erase: uniform, key: key, decay: [Double](repeating: 1.0, count: dk))
        #expect(zip(uniform, uniformGuarded).allSatisfy { abs($0 - $1) < 1e-12 })
    }

    @Test("GDN-2 reduces algebraically within 1e-5 to BailingHybridKDAKernel (tied erase=write) and GatedDeltaFastPath (scalar decay)")
    func exactAlgebraicReductionsToKDAAndGDN() throws {
        let (b, t, hk, hv, dk, dv) = (1, 10, 2, 4, 32, 32)
        let qRaw = MLXArray((0 ..< b * t * hk * dk).map { Float(($0 * 17) % 31 - 15) * 0.04 }, [b, t, hk, dk])
        let kRaw = MLXArray((0 ..< b * t * hk * dk).map { Float(($0 * 13) % 29 - 14) * 0.04 }, [b, t, hk, dk])
        let q = BailingHybridKDAOps.l2Normalize(qRaw)
        let k = BailingHybridKDAOps.l2Normalize(kRaw)
        let v = MLXArray((0 ..< b * t * hv * dv).map { Float(($0 * 11) % 23 - 11) * 0.05 }, [b, t, hv, dv])
        let gLogPerChannel = MLXArray((0 ..< b * t * hv * dk).map { Float(-0.15 - Double($0 % 9) * 0.2) }, [b, t, hv, dk])
        let betaScalar = MLXArray((0 ..< b * t * hv).map { Float(0.2 + Double($0 % 7) * 0.1) }, [b, t, hv])
        let eraseBcast = broadcast(betaScalar.expandedDimensions(axis: -1), to: [b, t, hv, dk])
        let writeBcast = broadcast(betaScalar.expandedDimensions(axis: -1), to: [b, t, hv, dv])

        let kdaInputs = BailingHybridKDAInputs(q: q, k: k, v: v, decay: MLX.exp(gLogPerChannel), beta: betaScalar)
        let (kdaOut, kdaState) = BailingHybridKDAKernel.run(kdaInputs, state: nil)
        let gdn2ForKDA = GatedDeltaNet2Inputs(q: q, k: k, v: v, gLog: gLogPerChannel, erase: eraseBcast, write: writeBcast)
            .withBetaGuard(true)
        let (gdn2KDAOut, gdn2KDAState) = GatedDeltaNet2Kernel.run(gdn2ForKDA, state: nil)
        let (gdn2WYOut, gdn2WYState) = GatedDeltaNet2Ops.chunkedWY(gdn2ForKDA, state: nil, chunkSize: 4)
        eval(kdaOut, kdaState, gdn2KDAOut, gdn2KDAState, gdn2WYOut, gdn2WYState)
        #expect(MLX.abs(gdn2KDAOut - kdaOut).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(gdn2KDAState - kdaState).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(gdn2WYOut - kdaOut).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(gdn2WYState - kdaState).max().item(Float.self) < 1e-5)

        let gLogScalar = MLXArray((0 ..< b * t * hv).map { Float(-0.12 - Double($0 % 5) * 0.18) }, [b, t, hv])
        let gLogScalarBcast = broadcast(gLogScalar.expandedDimensions(axis: -1), to: [b, t, hv, dk])
        let gdnInputs = GatedDeltaInputs(queries: q, keys: k, values: v, gamma: MLX.exp(gLogScalar), beta: betaScalar)
        let (gdnOut, gdnState) = try GatedDeltaFastPath.update(gdnInputs)
        let gdn2ForGDN = GatedDeltaNet2Inputs(q: q, k: k, v: v, gLog: gLogScalarBcast, erase: eraseBcast, write: writeBcast)
            .withBetaGuard(true)
        let (gdn2Out, gdn2State) = GatedDeltaNet2Kernel.run(gdn2ForGDN, state: nil)
        eval(gdnOut, gdnState, gdn2Out, gdn2State)
        #expect(MLX.abs(gdn2Out - gdnOut).max().item(Float.self) < 1e-5)
        #expect(MLX.abs(gdn2State - gdnState).max().item(Float.self) < 1e-5)
    }

    @Test("Kernel handles empty steps, non-contiguous slices, and falls back to pure MLX when Dk is not divisible by 32")
    func kernelEdgeCasesAndFallback() throws {
        let oddSpec = KernelParitySpec(
            label: "odd-dk", batch: 1, steps: 6, hk: 2, hv: 4, dk: 24, dv: 16,
            dtype: .float32, allowNegEigval: true, betaGuard: true, masked: true, tolerance: 1e-4
        )
        let odd = Self.makeFixture(oddSpec)
        #expect(!GatedDeltaNet2Kernel.canUseKernel(odd.mlx, state: odd.initState, mask: odd.mask))
        let (fbOut, fbState) = GatedDeltaNet2Ops.forward(odd.mlx, state: odd.initState, mask: odd.mask, chunkSize: 1, useMetal: true)
        let cpu = GatedDeltaNet2CPUReference.forward(odd.cpu)
        eval(fbOut, fbState)
        let outGap = zip(fbOut.asArray(Float.self), cpu.output).map { abs($0 - $1) }.max() ?? 0
        let stateGap = zip(fbState.asArray(Float.self), cpu.state).map { abs($0 - $1) }.max() ?? 0
        #expect(outGap < 1e-4 && stateGap < 1e-4)

        let empty = GatedDeltaNet2Inputs(
            q: MLXArray.zeros([1, 0, 2, 32]), k: MLXArray.zeros([1, 0, 2, 32]),
            v: MLXArray.zeros([1, 0, 4, 32]), gLog: MLXArray.zeros([1, 0, 4, 32]),
            erase: MLXArray.zeros([1, 0, 4, 32]), write: MLXArray.zeros([1, 0, 4, 32])
        )
        let initS = MLXArray.ones([1, 4, 32, 32], dtype: .float32)
        let (emptyOut, emptyState) = GatedDeltaNet2Kernel.run(empty, state: initS)
        #expect(emptyOut.shape == [1, 0, 4, 32] && allClose(emptyState, initS).item(Bool.self))

        let ncSpec = KernelParitySpec(
            label: "non-contiguous", batch: 2, steps: 4, hk: 2, hv: 4, dk: 32, dv: 16,
            dtype: .float32, allowNegEigval: true, betaGuard: true, masked: false, tolerance: 1e-4
        )
        let nc = Self.makeFixture(ncSpec)
        let qRealNC = concatenated([nc.mlx.q, nc.mlx.q], axis: 2)[0..., 0..., 0 ..< 2, 0...]
        let kRealNC = concatenated([nc.mlx.k, nc.mlx.k], axis: 2)[0..., 0..., 0 ..< 2, 0...]
        let vRealNC = concatenated([nc.mlx.v, nc.mlx.v], axis: 2)[0..., 0..., 0 ..< 4, 0...]
        let realStrided = GatedDeltaNet2Inputs(
            q: qRealNC, k: kRealNC, v: vRealNC,
            gLog: nc.mlx.gLog, erase: nc.mlx.erase, write: nc.mlx.write
        ).withBetaGuard(nc.mlx.betaGuard)
        let (ncOut, ncState) = GatedDeltaNet2Kernel.run(realStrided, state: nc.initState)
        let (refOut, refState) = GatedDeltaNet2Ops.forward(nc.mlx, state: nc.initState, chunkSize: 1, useMetal: false)
        eval(ncOut, ncState, refOut, refState)
        #expect(MLX.abs(ncOut - refOut).max().item(Float.self) < 1e-4)
        #expect(MLX.abs(ncState - refState).max().item(Float.self) < 1e-4)

        let c64Inputs = GatedDeltaNet2Inputs(
            q: MLXArray.zeros([1, 2, 2, 32], dtype: .complex64),
            k: MLXArray.zeros([1, 2, 2, 32], dtype: .complex64),
            v: MLXArray.zeros([1, 2, 4, 32], dtype: .complex64),
            gLog: nc.mlx.gLog[0 ..< 1, 0 ..< 2], erase: nc.mlx.erase[0 ..< 1, 0 ..< 2],
            write: MLXArray.zeros([1, 2, 4, 32], dtype: .float32)
        )
        #expect(!GatedDeltaNet2Kernel.canUseKernel(c64Inputs))
        let cpuAllowed = Device.withDefaultDevice(.cpu) { GatedDeltaNet2Kernel.canUseKernel(nc.mlx, state: nc.initState) }
        #expect(!cpuAllowed)

        let trainable = try #require(try TinyModel.make(.gdn2) as? GatedDeltaNet2Model)
        trainable.train(true)
        let tok = MLXArray([Int32(1), 2, 3, 4])[.newAxis]
        let (loss, _) = valueAndGrad(model: trainable) { _, arrays in
            [trainable.forward(arrays[0], cache: nil, useChunked: false, useMetal: true).sum()]
        }(trainable, [tok])
        eval(loss)
        #expect(loss[0].item(Float.self).isFinite)
    }
}

extension GatedDeltaNet2ModelTests {
    @Test("Hybrid, pure (no short conv), and sliding-window GDN-2 models match full WY prefill vs Metal decode, non-Metal decode, and split prefill")
    func fullModelParityAcrossTopologies() throws {
        let extras = [
            "",
            #""attention_layer_indices": [], "use_short_conv": false, "tie_word_embeddings": true, "#,
            #""layer_types": ["linear_attention", "sliding_attention", "linear_attention", "full_attention"], "sliding_window": 8, "#,
        ]
        for extra in extras {
            let model = try #require(try TinyModel.make(.gdn2, adding: extra) as? GatedDeltaNet2Model)
            let tokenArray = MLXArray(Self.tokens)[.newAxis]
            let fullWY = model.forward(tokenArray, cache: nil, useChunked: true, useMetal: true).asType(.float32)
            let fullNoMetal = model.forward(tokenArray, cache: nil, useChunked: false, useMetal: false).asType(.float32)
            #expect(MLX.abs(fullWY - fullNoMetal).max().item(Float.self) < 1e-4)
            #expect(argMax(fullWY, axis: -1).asArray(Int32.self) == argMax(fullNoMetal, axis: -1).asArray(Int32.self))

            let stepCache = try model.newCache(parameters: nil)
            let stepped = concatenated(Self.tokens.map { model(MLXArray([$0])[.newAxis], cache: stepCache).asType(.float32) }, axis: 1)
            #expect(MLX.abs(stepped - fullWY).max().item(Float.self) < 1e-4)
            #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(fullWY, axis: -1).asArray(Int32.self))

            let splitCache = try model.newCache(parameters: nil)
            let partA = model(MLXArray(Array(Self.tokens[..<6]))[.newAxis], cache: splitCache).asType(.float32)
            let partB = model(MLXArray(Array(Self.tokens[6...]))[.newAxis], cache: splitCache).asType(.float32)
            let split = concatenated([partA, partB], axis: 1)
            #expect(MLX.abs(split - fullWY).max().item(Float.self) < 1e-4)
            #expect(argMax(split, axis: -1).asArray(Int32.self) == argMax(fullWY, axis: -1).asArray(Int32.self))
        }
    }

    @Test("GDN-2 PromptCacheTree, PromptCacheDiskStore safetensors round-trip, and exact O(1) RecurrentSnapshot bytes invariance between T=4 and T=64")
    func promptCacheDiskRoundTripAndConstantFootprint() throws {
        let model = try #require(try TinyModel.make(.gdn2) as? GatedDeltaNet2Model)
        let expectedFootprint = model.configuration.recurrentStateFootprint(batchSize: 1, activationBytes: 4)
        #expect(expectedFootprint.bytes == model.configuration.recurrentStateBytes)

        let shortPrompt = (0 ..< 5).map { ($0 * 7 + 3) % 200 }
        let longPrompt = (0 ..< 65).map { ($0 * 11 + 5) % 200 }
        let shortRec = try #require(RecurrentSnapshot(of: try TinyHybrid.snapshotted(model, shortPrompt)))
        let longRec = try #require(RecurrentSnapshot(of: try TinyHybrid.snapshotted(model, longPrompt)))
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
        let expected = model(TinyHybrid.row(second), cache: try model.newCache(parameters: nil))[0..., boundary..., 0...]
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = Array(first.prefix(boundary))
        let memLease = try tree.required("ns", for: first)
        var writer = DiskFixture.store(directory)
        #expect(writer.store(fingerprint: "fp", model: "gdn2", namespace: "ns", tokens: prefix, cache: memLease.cache) != nil)
        var reader = DiskFixture.store(directory)
        let layout = try model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName)
        let entry = try #require(reader.longestPrefix(fingerprint: "fp", namespace: "ns", of: prefix, longerThan: 0, maxBytes: ByteCount(1 << 26))?.entry)
        let loaded = try reader.load(entry, layout: layout)
        #expect(loaded.isSnapshot(of: prefix.count))
        let fromDisk = DiskFixture.nextLogits(model, first[boundary], loaded)
        let fromMemory = DiskFixture.nextLogits(model, first[boundary], memLease.cache)
        #expect(DiskFixture.identical(fromDisk, fromMemory))
        _ = consume memLease
    }

    @Test("Config aliases, KVCacheGeometry, AnatomyScan, weight sanitizer, registry, and ModelCapabilities work for gdn2")
    func configGeometrySanitizeAndRegistration() async throws {
        let litJSON = """
            {"model_type": "gated-deltanet-2", "vocab_size": 512, "n_embd": 64, "n_layer": 4,
             "n_head": 2, "num_value_heads": 4, "head_dim": 32, "expand_v": 1.0,
             "full_attention_interval": 4, "conv_size": 4, "allow_neg_eigval": true, "beta_guard": true}
            """
        let defaultCfg = try JSONDecoder.json5().decode(
            GatedDeltaNet2Configuration.self,
            from: Data(#"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 2, "head_dim": 32}"#.utf8)
        ).validated()
        #expect(!defaultCfg.betaGuard)
        let lit = try JSONDecoder.json5().decode(GatedDeltaNet2Configuration.self, from: Data(litJSON.utf8)).validated()
        #expect(lit.numHeads == 2 && lit.numVHeads == 4 && lit.attentionLayerIndices == [3] && lit.linearLayerCount == 3 && lit.betaGuard)

        let cfgObj = try #require(try JSONSerialization.jsonObject(with: Data(TinyModel.Family.gdn2.config.utf8)) as? [String: Any])
        let geom = try #require(KVCacheGeometry(config: cfgObj))
        #expect(geom.elementsPerToken > 0 && geom.fixedStateBytes > 0)

        let model = try #require(try TinyModel.make(.gdn2) as? GatedDeltaNet2Model)
        let scan = AnatomyScan(model: model, caches: try model.newCache(parameters: nil))
        let anatomy = scan.anatomy(config: cfgObj)
        #expect(anatomy.layers.map(\.kind) == ["linear attention", "linear attention", "linear attention", "full attention"])

        let badConfigs = [
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 3, "num_v_heads": 4, "head_dim": 32}"#,
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 2, "head_dim": 32, "attention_layer_indices": [5]}"#,
            #"{"vocab_size": 256, "hidden_size": 64, "num_hidden_layers": 2, "num_heads": 2, "head_dim": 32, "layer_types": ["linear_attention", "sliding_attention"]}"#,
        ]
        for bad in badConfigs {
            #expect(throws: BasaltError.self) {
                _ = try GatedDeltaNet2Model(configurationData: Data(bad.utf8))
            }
        }

        let tied = try #require(try TinyModel.make(.gdn2, adding: #""tie_word_embeddings": true, "#) as? GatedDeltaNet2Model)
        let raw: [String: MLXArray] = [
            "transformer.wte.weight": MLXArray.zeros([256, 64]),
            "transformer.ln_f.weight": MLXArray.ones([64]),
            "lm_head.weight": MLXArray.ones([256, 64]),
            "transformer.h.0.linear_attn.erase_proj.weight": MLXArray.ones([128, 64]),
            "transformer.h.0.linear_attn.write_proj.weight": MLXArray.ones([128, 64]),
            "transformer.h.0.linear_attn.decay_proj.weight": MLXArray.ones([128, 64]),
            "transformer.h.0.linear_attn.q_conv1d.weight": MLXArray.ones([64, 4]),
            "transformer.h.0.linear_attn.k_conv1d.conv.weight": MLXArray.ones([64, 1, 4]),
            "transformer.h.0.linear_attn.dt_bias": MLXArray.zeros([128]),
            "transformer.h.9.linear_attn.q_proj.weight": MLXArray.zeros([64, 64]),
        ]
        let sanitized = tied.sanitize(weights: raw)
        #expect(sanitized["model.embed_tokens.weight"] != nil && sanitized["model.norm.weight"] != nil)
        #expect(sanitized["lm_head.weight"] == nil && sanitized["model.layers.9.attn.q_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.attn.b_proj.weight"] != nil && sanitized["model.layers.0.attn.w_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.attn.a_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.attn.q_conv1d.conv.weight"]?.shape == [64, 4, 1])
        #expect(sanitized["model.layers.0.attn.k_conv1d.conv.weight"]?.shape == [64, 4, 1])
        #expect(sanitized["model.layers.0.attn.dt_bias"]?.shape == [4, 32])

        await MLXInferenceEngine.registerAliases()
        for alias in ["gdn2", "gated_deltanet_2", "gated-deltanet-2"] {
            #expect(await LLMTypeRegistry.shared.contains(alias))
            let created = try await LLMTypeRegistry.shared.createModel(
                configuration: Data(TinyModel.Family.gdn2.config.utf8),
                modelType: alias
            )
            #expect(created is GatedDeltaNet2Model)
            let cap = try #require(ModelCapabilities.of(modelType: alias))
            #expect(cap.allows(.promptCacheReuse) && cap.allows(.batchedDecode) && cap.allows(.grammarJumpForward))
            #expect(!cap.allows(.kvQuantization))
        }
        let engine = try await TinyModel.engine(.gdn2)
        let container = try #require(await engine.container)
        let epoch = await engine.modelEpoch
        #expect(await engine.cacheIsHybrid(container, epoch: epoch))
        #expect(await engine.modelHasRecurrentState == true)
    }
}

extension GatedDeltaNet2ModelTests {
    struct FixtureBundle {
        let mlx: GatedDeltaNet2Inputs
        let cpu: GatedDeltaNet2CPUInputs
        let initState: MLXArray
        let mask: MLXArray?
    }

    static func makeFixture(_ spec: KernelParitySpec) -> FixtureBundle {
        let (b, t, hk, hv, dk, dv) = (spec.batch, spec.steps, spec.hk, spec.hv, spec.dk, spec.dv)
        let qCount = b * t * hk * dk
        let vCount = b * t * hv * dv
        let gCount = b * t * hv * dk
        let sCount = b * hv * dv * dk

        let qRaw = MLXArray((0 ..< qCount).map { Float(($0 * 17) % 31 - 15) * 0.04 }, [b, t, hk, dk])
        let kRaw = MLXArray((0 ..< qCount).map { Float(($0 * 13) % 29 - 14) * 0.04 }, [b, t, hk, dk])
        let qNorm = BailingHybridKDAOps.l2Normalize(qRaw)
        let kNorm = BailingHybridKDAOps.l2Normalize(kRaw)
        eval(qNorm, kNorm)

        let vArr = (0 ..< vCount).map { Float(($0 * 11) % 23 - 11) * 0.05 }
        let gLogArr = (0 ..< gCount).map { Float(-0.08 - Double($0 % 9) * 0.14) }
        let eraseScale: Float = spec.allowNegEigval ? 2.0 : 1.0
        let eraseArr = (0 ..< gCount).map { Float(($0 % 2 == 0) ? 0.92 : 0.08) * eraseScale }
        let writeArr = (0 ..< vCount).map { Float(0.2 + Double($0 % 7) * 0.1) }
        let stateArr = (0 ..< sCount).map { Float(($0 * 7) % 19 - 9) * 0.02 }
        let maskArr: [Bool]? = spec.masked ? (0 ..< b * t).map { ($0 % 5) != 2 } : nil

        let qMLX = qNorm.asType(spec.dtype)
        let kMLX = kNorm.asType(spec.dtype)
        let vMLX = MLXArray(vArr, [b, t, hv, dv]).asType(spec.dtype)
        let gLogMLX = MLXArray(gLogArr, [b, t, hv, dk])
        let eraseMLX = MLXArray(eraseArr, [b, t, hv, dk])
        let writeMLX = MLXArray(writeArr, [b, t, hv, dv])
        let stateMLX = MLXArray(stateArr, [b, hv, dv, dk])

        let mlxInputs = GatedDeltaNet2Inputs(
            q: qMLX, k: kMLX, v: vMLX, gLog: gLogMLX, erase: eraseMLX, write: writeMLX
        ).withBetaGuard(spec.betaGuard)

        let decayArr = gLogArr.map { Foundation.exp($0) }
        let shape = GatedDeltaNet2CPUShape(batch: b, steps: t, hk: hk, hv: hv, dk: dk, dv: dv)
        let cpuInputs = GatedDeltaNet2CPUInputs(
            shape: shape,
            q: qNorm.asArray(Float.self),
            k: kNorm.asArray(Float.self),
            v: vArr,
            decay: decayArr,
            erase: eraseArr
        ).withWriteAndState(write: writeArr, betaGuard: spec.betaGuard, initialState: stateArr, mask: maskArr)

        return FixtureBundle(
            mlx: mlxInputs,
            cpu: cpuInputs,
            initState: stateMLX,
            mask: maskArr.map { MLXArray($0, [b, t]) }
        )
    }

    @Test("GDN-2 Metal kernel supports CustomFunction VJP at training == false, Kahan +Inf non-NaN at Dk=64, and float64 mask CPU fallback")
    func customVJPAndEdgeHardening() throws {
        let spec = KernelParitySpec(
            label: "vjp-f32", batch: 1, steps: 2, hk: 1, hv: 2, dk: 32, dv: 16,
            dtype: .float32, allowNegEigval: true, betaGuard: true, masked: true, tolerance: 1e-4
        )
        let fix = Self.makeFixture(spec)
        let (_, metalGrads) = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in
            let inp = GatedDeltaNet2Inputs(
                q: a[0], k: a[1], v: a[2], gLog: a[3], erase: a[4], write: a[5]
            ).withBetaGuard(true)
            let (out, state) = GatedDeltaNet2Kernel.run(inp, state: a[6], mask: fix.mask)
            return [out.sum() + state.sum()]
        })([fix.mlx.q, fix.mlx.k, fix.mlx.v, fix.mlx.gLog, fix.mlx.erase, fix.mlx.write, fix.initState])
        let (_, opsGrads) = valueAndGrad({ (a: [MLXArray]) -> [MLXArray] in
            let inp = GatedDeltaNet2Inputs(
                q: a[0], k: a[1], v: a[2], gLog: a[3], erase: a[4], write: a[5]
            ).withBetaGuard(true)
            let (out, state) = GatedDeltaNet2Ops.recurrentOps(inp, state: a[6], mask: fix.mask)
            return [out.sum() + state.sum()]
        })([fix.mlx.q, fix.mlx.k, fix.mlx.v, fix.mlx.gLog, fix.mlx.erase, fix.mlx.write, fix.initState])
        eval(metalGrads + opsGrads)
        for (mg, og) in zip(metalGrads, opsGrads) {
            #expect(MLX.abs(mg - og).max().item(Float.self) < 1e-4)
        }

        let model = try #require(try TinyModel.make(.gdn2) as? GatedDeltaNet2Model)
        model.train(false)
        let singleTok = MLXArray([Int32(7)])[.newAxis]
        let (lossEval, _) = valueAndGrad(model: model) { _, arrays in
            [model.forward(arrays[0], cache: nil, useChunked: false, useMetal: true).sum()]
        }(model, [singleTok])
        eval(lossEval)
        #expect(lossEval[0].item(Float.self).isFinite)

        var qInf = [Float](repeating: 1, count: 64)
        qInf[0] = 1e20
        var sInf = [Float](repeating: 0, count: 64)
        sInf[0] = 1e20
        let infInputs = GatedDeltaNet2Inputs(
            q: MLXArray(qInf, [1, 1, 1, 64]),
            k: MLXArray.zeros([1, 1, 1, 64], dtype: .float32),
            v: MLXArray.zeros([1, 1, 1, 1], dtype: .float32),
            gLog: MLXArray.zeros([1, 1, 1, 64], dtype: .float32),
            erase: MLXArray.zeros([1, 1, 1, 64], dtype: .float32),
            write: MLXArray.zeros([1, 1, 1, 1], dtype: .float32)
        ).withBetaGuard(false)
        let (kOut, _) = GatedDeltaNet2Kernel.run(infInputs, state: MLXArray(sInf, [1, 1, 1, 64]))
        eval(kOut)
        let val = kOut.item(Float.self)
        #expect(val == .infinity && !val.isNaN)

        let f64Mask = Device.withDefaultDevice(.cpu) { MLXArray([Double(1), 0], [1, 2]) }
        #expect(!GatedDeltaNet2Kernel.canUseKernel(fix.mlx, state: fix.initState, mask: f64Mask))
        let (f64Out, _) = GatedDeltaNet2Kernel.run(fix.mlx, state: fix.initState, mask: f64Mask)
        eval(f64Out)
        #expect(f64Out.shape == [1, 2, 2, 16])
    }
}
