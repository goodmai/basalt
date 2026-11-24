import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("BailingHybrid (Ling-3.0) P0: decode parity, sanitize, disk cache and capabilities (#672)", .serialized)
struct BailingHybridIntegrationTests {
    static let sequenceTokens: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31, 200, 3, 3, 150, 77, 12, 44, 101, 6, 250]

    @Test("Tiny bailing_hybrid cache layout has 4-slot ArraysCache on KDA layers and KVCacheSimple on MLA layers")
    func cacheLayoutAndAnatomy() throws {
        let model = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        #expect(model.loraLayers.count == 6 && model.toolCallFormat == .glm4)
        let caches = try model.newCache(parameters: nil)
        #expect(caches.count == 6)
        #expect(caches.map { $0 is ArraysCache } == [true, true, false, true, true, false])
        #expect(caches[2] is KVCacheSimple && caches[5] is KVCacheSimple)

        _ = model(MLXArray(Self.sequenceTokens)[.newAxis], cache: caches)
        eval(caches.flatMap { $0.innerState() })
        for idx in [0, 1, 3, 4] {
            let kdaCache = try #require(caches[idx] as? ArraysCache)
            #expect(kdaCache[0]?.shape == [1, 3, 128])
            #expect(kdaCache[1]?.shape == [1, 3, 128])
            #expect(kdaCache[2]?.shape == [1, 3, 128])
            #expect(kdaCache[3]?.shape == [1, 4, 32, 32])
            #expect(kdaCache[3]?.dtype == .float32)
        }

        let scan = AnatomyScan(model: model, caches: try model.newCache(parameters: nil))
        let configJSON = try #require(try JSONSerialization.jsonObject(with: Data(TinyModel.Family.bailingHybrid.config.utf8)) as? [String: Any])
        let anatomy = scan.anatomy(config: configJSON)
        #expect(anatomy.hiddenSize == 64 && anatomy.layers.count == 6)
        #expect(anatomy.layers.map(\.kind) == ["linear attention", "linear attention", "full attention", "linear attention", "linear attention", "full attention"])
        #expect(anatomy.layers[0].heads == 4 && anatomy.layers[0].kvHeads == 4 && anatomy.layers[0].headDim == 32)
        #expect(anatomy.layers[0].experts == nil && anatomy.layers[1].experts == 8 && anatomy.layers[5].experts == 8)
    }

    @Test("Full forward matches token-by-token and chunked (1, 3, 7) cached decode with exact greedy ids and logits within 1e-4",
          arguments: [1, 3, 7])
    func decodeParityAcrossChunks(chunk: Int) throws {
        let model = try TinyModel.make(.bailingHybrid)
        let full = model(MLXArray(Self.sequenceTokens)[.newAxis], cache: nil).asType(.float32)
        let caches = try model.newCache(parameters: nil)
        var pieces: [MLXArray] = []
        var cursor = 0
        while cursor < Self.sequenceTokens.count {
            let end = min(Self.sequenceTokens.count, cursor + chunk)
            let slice = Array(Self.sequenceTokens[cursor ..< end])
            pieces.append(model(MLXArray(slice)[.newAxis], cache: caches).asType(.float32))
            cursor = end
        }
        let stepped = concatenated(pieces, axis: 1)
        let gap = abs(stepped - full).max().item(Float.self)
        #expect(stepped.shape == [1, Self.sequenceTokens.count, 256])
        #expect(gap < 1e-4, "chunk \(chunk) max logit gap \(gap)")
        #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))
    }

    @Test("Optional variants (KDA LoRA, no q_lora_rank, element_wise/nil gate, tied embeddings, kernel 1, rope scaling) decode identically full vs T=1")
    func optionalArchitectureVariants() throws {
        let variants: [String] = [
            #""no_kda_lora": false, "q_lora_rank": null, "gated_attention_proj_granularity_type": "element_wise", "tie_word_embeddings": true, "#,
            #""short_conv_kernel_size": 1, "gated_attention_proj_granularity_type": null, "kda_safe_gate": false, "num_shared_experts": 0, "#,
            #""rope_scaling": {"type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096}, "moe_router_enable_expert_bias": false, "#,
        ]
        for extra in variants {
            let model = try TinyModel.make(.bailingHybrid, adding: extra)
            let tokens = Array(Self.sequenceTokens.prefix(8))
            let full = model(MLXArray(tokens)[.newAxis], cache: nil).asType(.float32)
            let caches = try model.newCache(parameters: nil)
            let steps = tokens.map { model(MLXArray([$0]).reshaped(1, 1), cache: caches).asType(.float32) }
            let stepped = concatenated(steps, axis: 1)
            let gap = abs(stepped - full).max().item(Float.self)
            #expect(gap < 1e-4, "variant gap \(gap)")
            #expect(argMax(stepped, axis: -1).asArray(Int32.self) == argMax(full, axis: -1).asArray(Int32.self))
        }
    }

    @Test("Packed-weight sanitize round-trip keeps router, short-conv, A_log and dt_bias in fp while quantizing packed linear and switch layers")
    func packedWeightSanitizeRoundTrip() throws {
        let source = try Self.makePackedTinyModel()
        let expectedLogits = source(MLXArray(Self.sequenceTokens)[.newAxis], cache: nil).asType(.float32)
        let hfStyleWeights = try Self.unmergeIntoHFCheckpoint(Dictionary(uniqueKeysWithValues: source.parameters().flattened()))

        let restored = try BailingHybridModel(configurationData: Data(TinyModel.Family.bailingHybrid.config.utf8))
        let sanitized = restored.sanitize(weights: hfStyleWeights)
        try restored.update(parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        eval(restored)

        let modules = Dictionary(uniqueKeysWithValues: restored.namedModules())
        #expect(!(modules["model.layers.1.mlp.gate"] is any Quantized))
        #expect(!(modules["model.layers.0.attention.q_conv1d.conv"] is any Quantized))
        #expect(modules["model.layers.0.attention.q_proj"] is any Quantized)
        #expect(modules["model.layers.2.attention.dense"] is any Quantized)
        #expect(modules["model.layers.1.mlp.experts.gate_proj"] is any Quantized)

        let params = Dictionary(uniqueKeysWithValues: restored.parameters().flattened())
        #expect(params["model.layers.1.mlp.gate.weight"]?.dtype != .uint32)
        #expect(params["model.layers.1.mlp.gate.expert_bias"]?.dtype != .uint32)
        #expect(params["model.layers.0.attention.q_conv1d.conv.weight"]?.dtype != .uint32)
        #expect(params["model.layers.0.attention.A_log"]?.dtype == .float32)
        #expect(params["model.layers.0.attention.dt_bias"]?.dtype == .float32)

        let restoredLogits = restored(MLXArray(Self.sequenceTokens)[.newAxis], cache: nil).asType(.float32)
        #expect(abs(restoredLogits - expectedLogits).max().item(Float.self) == 0)
    }

    @Test("Sanitize handles 2D conv weights, tied embeddings, unused expert_bias, and preserves incomplete unmerged experts")
    func sanitizeEdgeCases() throws {
        var config = try JSONDecoder.json5().decode(BailingHybridConfiguration.self, from: Data(TinyModel.Family.bailingHybrid.config.utf8))
        config.tieWordEmbeddings = true
        config.moeRouterEnableExpertBias = false

        let raw: [String: MLXArray] = [
            "lm_head.weight": MLXArray.zeros([256, 64]),
            "model.layers.1.mlp.gate.expert_bias": MLXArray.zeros([8]),
            "model.layers.0.attention.q_conv1d.weight": MLXArray.zeros([128, 4]),
            "model.layers.0.attention.k_conv1d.conv.weight": MLXArray.zeros([128, 1, 4]),
            "model.layers.1.mlp.experts.0.gate_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.1.mlp.experts.1.gate_proj.weight": MLXArray.zeros([32, 64]),
            "mtp_head.weight": MLXArray.zeros([8, 8]),
        ]
        let sanitized = BailingHybridCheckpoint.sanitize(raw, config: config)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.gate.expert_bias"] == nil)
        #expect(sanitized["mtp_head.weight"] == nil)
        #expect(sanitized["model.layers.0.attention.q_conv1d.conv.weight"]?.shape == [128, 4, 1])
        #expect(sanitized["model.layers.0.attention.k_conv1d.conv.weight"]?.shape == [128, 4, 1])
        #expect(sanitized["model.layers.1.mlp.experts.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.experts.1.gate_proj.weight"] != nil)
    }

    @Test("ModelDeepCheck and ModelInspector handle quantization_config-only and dual quantization configs without double-quantizing")
    func deepCheckAndInspectorRecognizeQuantizedCheckpoints() async throws {
        let source = try Self.makePackedTinyModel()
        let rawWeights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let tensors = Dictionary(uniqueKeysWithValues: rawWeights.map { key, arr in
            (key, ModelInspector.Tensor(dtype: arr.dtype == .uint32 ? "U32" : "F32", shape: arr.shape, end: 0))
        })
        let quantOnly = TinyModel.Family.bailingHybrid.config.replacingOccurrences(
            of: #""model_type""#,
            with: #""quantization_config": {"bits": 4, "group_size": 32, "quant_method": "mlx"}, "model_type""#
        )
        let dualQuant = TinyModel.Family.bailingHybrid.config.replacingOccurrences(
            of: #""model_type""#,
            with: #""quantization": {"bits": 4, "group_size": 32}, "quantization_config": {"bits": 4, "group_size": 32}, "model_type""#
        )
        for cfg in [quantOnly, dualQuant] {
            let problems = await ModelDeepCheck.loadProblems(configuration: Data(cfg.utf8), tensors: tensors, metadata: [:])
            #expect(problems.isEmpty, "ModelDeepCheck reported \(problems)")
        }

        let json = try #require(try JSONSerialization.jsonObject(with: Data(quantOnly.utf8)) as? [String: Any])
        let bitWidth = ModelInspector.bitWidth(
            tensors: tensors, dtypes: ["U32", "F32"], config: ModelInspector.QuantizationConfig(json: json), modelId: "Ling-3.0-tiny-MLX-4bit"
        )
        #expect(bitWidth.bits == 4 && bitWidth.mismatch == nil)
    }
}

extension BailingHybridIntegrationTests {
    @Test("ModelCapabilities, ModelProfile, registry registration, recurrent state probe, and disk snapshot round-trip work for bailing_hybrid")
    func capabilitiesProfileRegistryAndSnapshot() async throws {
        await MLXInferenceEngine.registerAliases()
        #expect(await LLMTypeRegistry.shared.contains("bailing_hybrid"))
        #expect(await VLMTypeRegistry.shared.contains("bailing_hybrid"))

        let capabilities = try #require(ModelCapabilities.of(modelType: "bailing_hybrid"))
        #expect(capabilities.allows(.promptCacheReuse) && capabilities.allows(.grammarJumpForward) && capabilities.allows(.batchedDecode))
        #expect(!capabilities.allows(.speculativeLookup) && !capabilities.allows(.kvZones) && !capabilities.allows(.kvHeadSplit))
        #expect(!capabilities.allows(.kvQuantization) && !capabilities.allows(.quest) && !capabilities.allows(.mtp) && !capabilities.allows(.cacheBlend))
        #expect(throws: BasaltError.self) {
            try ModelCapabilities.refuseUnverifiedKVBits(.q4, modelType: "bailing_hybrid", path: "ling3")
        }

        let profile = ModelProfileRegistry.profile(forModelType: "bailing_hybrid")
        #expect(profile.name == "bailing_hybrid" && profile.defaultMaxTokens == 16_384 && profile.defaultMinP == nil)
        #expect(profile.thinkingMarkers.open == ["<think>"] && profile.thinkingMarkers.close == ["</think>"])
        #expect(profile.templateContext(for: .modelDefault).isEmpty)
        #expect(profile.templateContext(for: .disabled)["enable_thinking"] as? Bool == false)
        #expect(profile.templateContext(for: .effort("high"))["enable_thinking"] as? Bool == true)
        #expect(profile.templateContext(for: .effort("none"))["enable_thinking"] as? Bool == false)

        let userTurn = [ChatMessageInput(role: .user, text: "hi")]
        #expect(profile.turns(userTurn, for: .modelDefault).map(\.text) == ["hi"])
        #expect(profile.turns(userTurn, for: .disabled).first?.text == "detailed thinking off")
        #expect(profile.turns(userTurn, for: .effort("high")).first?.text == "detailed thinking on")
        let withSystem = [ChatMessageInput(role: .system, text: "Be brief."), ChatMessageInput(role: .user, text: "hi")]
        #expect(profile.turns(withSystem, for: .disabled).first?.text == "Be brief.\ndetailed thinking off")
        let alreadyExplicit = [ChatMessageInput(role: .system, text: "detailed thinking off"), ChatMessageInput(role: .user, text: "hi")]
        #expect(profile.turns(alreadyExplicit, for: .effort("high")).first?.text == "detailed thinking off")

        let engine = try await TinyModel.engine(.bailingHybrid)
        let container = try #require(await engine.container)
        let epoch = await engine.modelEpoch
        #expect(await engine.cacheIsHybrid(container, epoch: epoch))
        #expect(await engine.modelHasRecurrentState == true)
    }

    @Test("Recurrent prompt-cache snapshots replay identically from memory tree and PromptCacheDiskStore")
    func promptCacheTreeAndDiskRoundTrip() throws {
        let model = try TinyModel.make(.bailingHybrid)
        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.afterToolCall)
        let boundary = first.count - 1
        let snapshot = try TinyHybrid.snapshotted(model, first)
        #expect(snapshot.isHybrid && snapshot.isSnapshot(of: boundary))

        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.checkedStore("ns", tokens: Array(first.prefix(boundary)), cache: snapshot)
        let lease = try tree.required("ns", for: second)
        let tail = model(TinyHybrid.row(Array(second[boundary...])), cache: lease.cache.layers)
        eval(tail)
        _ = consume lease
        let expected = model(TinyHybrid.row(second), cache: try model.newCache(parameters: nil))[0..., boundary..., 0...]
        #expect(allClose(tail, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let directory = DiskFixture.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = Array(first.prefix(boundary))
        let memoryLease = try tree.required("ns", for: first)
        var writer = DiskFixture.store(directory)
        #expect(writer.store(fingerprint: "fp", model: "tiny", namespace: "ns", tokens: prefix, cache: memoryLease.cache) != nil)
        var reader = DiskFixture.store(directory)
        let layout = try model.newCache(parameters: nil).map(PromptCacheDiskStore.layoutName)
        let found = reader.longestPrefix(fingerprint: "fp", namespace: "ns", of: prefix, longerThan: 0, maxBytes: ByteCount(1 << 26))?.entry
        let entry = try #require(found)
        let loaded = try reader.load(entry, layout: layout)
        #expect(loaded.isSnapshot(of: prefix.count))
        let fromDisk = DiskFixture.nextLogits(model, first[boundary], loaded)
        let fromMemory = DiskFixture.nextLogits(model, first[boundary], memoryLease.cache)
        #expect(DiskFixture.identical(fromDisk, fromMemory))
        _ = consume memoryLease
    }

    static func makePackedTinyModel() throws -> BailingHybridModel {
        let source = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        quantize(model: source, groupSize: 32, bits: 4, mode: .affine) { path, module in
            guard !path.contains("_conv1d"), !path.contains("mlp.gate") else { return false }
            return module is Linear || module is Embedding || module is SwitchLinear
        }
        eval(source)
        return source
    }

    static func unmergeIntoHFCheckpoint(_ raw: [String: MLXArray]) throws -> [String: MLXArray] {
        var out = raw
        for key in Array(out.keys) where key.hasSuffix("_conv1d.conv.weight") {
            let removed = out.removeValue(forKey: key)
            let weight = try #require(removed)
            let hfKey = key.replacingOccurrences(of: "_conv1d.conv.weight", with: "_conv1d.weight")
            out[hfKey] = weight.swappedAxes(1, 2)
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for part in ["weight", "scales", "biases"] {
                let removed = out.removeValue(forKey: "model.layers.1.mlp.experts.\(proj).\(part)")
                let stacked = try #require(removed)
                for expert in 0 ..< 8 {
                    out["model.layers.1.mlp.experts.\(expert).\(proj).\(part)"] = stacked[expert]
                }
            }
        }
        out["model.layers.6.attention.dense.weight"] = MLXArray.zeros([64, 64])
        out["mtp.head.weight"] = MLXArray.zeros([16, 16])
        return out
    }
}
