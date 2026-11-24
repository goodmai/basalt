import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Gemma 4 self-rewriting P0: topology, validator, WeightCopy, MACaQ, SmallM, NF4, and NVFP4 (#770)", .serialized)
struct Gemma4SelfRewritingP0Tests {
    static let textOProj = "model.layers.0.self_attn.o_proj"
    static let textDownProj = "model.layers.0.mlp.down_proj"
    static let wrappedOProj = "language_model.model.layers.0.self_attn.o_proj"
    static let routerPerExpertScale = "model.layers.0.router.per_expert_scale"
    static let moeSwitchGate = "model.layers.0.experts.switch_glu.gate_proj"

    static var promptTokens: MLXArray {
        MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)).reshaped(1, -1)
    }

    static func logitsFloat(_ model: any LanguageModel) throws -> [Float] {
        let cache = try model.newCache(parameters: nil)
        let out = model(promptTokens, cache: cache).asType(.float32)
        eval(out)
        return out.asArray(Float.self)
    }

    static func logitsBits(_ model: any LanguageModel) throws -> [UInt32] {
        try logitsFloat(model).map(\.bitPattern)
    }

    static func unitVector(size: Int, seed: UInt64 = 770) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let v = MLXRandom.normal([size]).asType(.float32)
            let unit = v / sqrt((v * v).sum())
            eval(unit)
            return unit
        }
    }

    static func makeMACaQLinear(
        outputs: Int,
        inputs: Int,
        mode: QuantizationMode,
        withRowScale: Bool,
        withBias: Bool
    ) -> MACaQQuantizedLinear {
        withRandomState(MLXRandom.RandomState(seed: mode == .mxfp4 ? 771 : 772)) {
            let dtype: DType = mode == .mxfp4 ? .bfloat16 : .float16
            let dense = (MLXRandom.normal([outputs, inputs]).asType(.float32) * 0.25).asType(dtype)
            let (qWeight, qScales, qBiases) = MLX.quantized(dense, groupSize: 32, bits: 4, mode: mode)
            let preScale = (MLXArray.ones([inputs], dtype: .float32) + MLXRandom.uniform(low: -0.2, high: 0.2, [inputs])).asType(dtype)
            let rowScale: MLXArray? = withRowScale
                ? (MLXArray.ones([outputs], dtype: .float32) + MLXRandom.uniform(low: -0.15, high: 0.15, [outputs])).asType(dtype)
                : nil
            let bias: MLXArray? = withBias ? (MLXRandom.normal([outputs]).asType(.float32) * 0.05).asType(dtype) : nil
            var spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: mode, hadamardBlock: 32)
            spec.scaleDType = "\(qScales.dtype)"
            spec.hasBias = withBias
            let packed = MACaQPackedWeights(weight: qWeight, scales: qScales, biases: qBiases)
            let layer = MACaQQuantizedLinear(packed: packed, bias: bias, preScale: preScale, spec: spec, rowScale: rowScale)
            eval(layer)
            return layer
        }
    }

    static func makeNVFP4Linear(outputs: Int, inputs: Int, globalScaleValue: Float, withBias: Bool) -> QuantizedLinear {
        withRandomState(MLXRandom.RandomState(seed: 773)) {
            let dense = (MLXRandom.normal([outputs, inputs]).asType(.float32) * 0.2).asType(.bfloat16)
            let bias: MLXArray? = withBias ? (MLXRandom.normal([outputs]).asType(.float32) * 0.05).asType(.bfloat16) : nil
            let globalScale = MLXArray(globalScaleValue, dtype: .float32)
            let layer = QuantizedLinear(weight: dense, bias: bias, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
            eval(layer)
            return layer
        }
    }

    static func makeTinyGemma4VLM() throws -> MLXVLM::Gemma4 {
        let json = """
            {"model_type": "gemma4", "vocab_size": 256,
             "text_config": \(TinyModel.Family.gemma4.config),
             "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 64,
              "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "patch_size": 16,
              "position_embedding_size": 16, "default_output_length": 4, "pooling_kernel_size": 2}}
            """
        let config = try JSONDecoder.json5().decode(MLXVLM::Gemma4Configuration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 774)) { MLXVLM::Gemma4(config) }
        try model.prepare()
        eval(model)
        return model
    }

    static func makeTinyGemma4LLMWrapper() throws -> MLXLLM::Gemma4Model {
        let json = """
            {"model_type": "gemma4", "vocab_size": 256, "text_config": \(TinyModel.Family.gemma4.config)}
            """
        let config = try JSONDecoder.json5().decode(MLXLLM::Gemma4Configuration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 775)) { MLXLLM::Gemma4Model(config) }
        try model.prepare()
        eval(model)
        return model
    }

    static func quantizeDivisible32(_ model: Module) {
        quantize(model: model, groupSize: 32, bits: 4) { _, module in
            if let linear = module as? Linear { return linear.shape.1 % 32 == 0 }
            if let sw = module as? SwitchLinear,
               let w = sw.parameters().flattened().first(where: { $0.0 == "weight" })?.1 {
                return w.dim(2) % 32 == 0
            }
            return false
        }
    }

    static func verifyLayerRank1DampParity(
        base: Linear,
        path: String,
        alpha: Float,
        tolerance: Float,
        seed: UInt64
    ) throws {
        let model = try TinyModel.make(.gemma4)
        try MLXInferenceEngine.replace(path, with: base, in: model)
        let origBits = try logitsBits(model)
        let (outputs, inputs) = base.shape
        let u = unitVector(size: outputs, seed: seed)
        let x = withRandomState(MLXRandom.RandomState(seed: seed + 1)) {
            MLXRandom.normal([1, 4, inputs]).asType(EditOpExecutor.computeType(of: base))
        }
        let baseY = base(x).asType(.float32)
        let linearOnlyY = base.bias.map { baseY - $0.asType(.float32) } ?? baseY
        let expectedY = baseY - alpha * (linearOnlyY * u.reshaped(1, 1, outputs)).sum(axis: -1, keepDims: true) * u.reshaped(1, 1, outputs)
        eval(expectedY)

        var copy = try WeightCopy.makeSync(from: model)
        let program = EditProgram(ops: [.rank1Damp(module: path, vectorKey: "u", alpha: alpha)])
        _ = try copy.apply(program, tensors: ["u": u], verifying: model)
        let editedLayer = try #require((try copy.model).namedModules().first(where: { $0.0 == path })?.1 as? Linear)
        let actualY = editedLayer(x).asType(.float32)
        eval(actualY)
        let maxDiff = abs(actualY - expectedY).max().item(Float.self)
        #expect(maxDiff < tolerance, "path=\(path) maxDiff=\(maxDiff)")
        #expect(try logitsBits(model) == origBits)
        try copy.undoAll(verifying: model)
        #expect(try copy.logits(for: promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == origBits)
        try copy.release(verifying: model)
    }
}

extension Gemma4SelfRewritingP0Tests {
    @Test("Gemma4EditTopology extracts KV donor map, attention_k_eq_v, MoE, PLE, softcap, and blast-radius warnings")
    func gemma4TopologyMetadataAndWarnings() throws {
        let e4bConfig: [String: Any] = [
            "model_type": "gemma4",
            "text_config": [
                "model_type": "gemma4_text",
                "num_hidden_layers": 6,
                "num_kv_shared_layers": 2,
                "sliding_window_pattern": 2,
                "hidden_size_per_layer_input": 256,
                "final_logit_softcapping": 30.0,
            ],
        ]
        let e4b = try #require(Gemma4EditTopology(config: e4bConfig))
        #expect(e4b.numLayers == 6)
        #expect(e4b.kvDonorForLayers == [2: [4], 3: [5]])
        #expect(e4b.kEqVLayers.isEmpty)
        #expect(!e4b.hasParallelDenseAndMoE)
        #expect(e4b.hasPerLayerEmbeddings)
        #expect(e4b.finalLogitSoftcapping == 30.0)
        #expect(e4b.blastRadiusWarning(for: "language_model.model.layers.2.self_attn.k_proj")?.contains("shared layers [4]") == true)
        #expect(e4b.blastRadiusWarning(for: "language_model.model.layers.3.self_attn.v_proj")?.contains("shared layers [5]") == true)
        #expect(e4b.blastRadiusWarning(for: "language_model.model.layers.2.self_attn.o_proj") == nil)
        #expect(e4b.blastRadiusWarning(for: "language_model.model.layers.0.self_attn.k_proj") == nil)

        let moeConfig: [String: Any] = [
            "model_type": "gemma4_text",
            "num_hidden_layers": 4,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "attention_k_eq_v": true,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "hidden_size_per_layer_input": 0,
        ]
        let moe = try #require(Gemma4EditTopology(config: moeConfig))
        #expect(moe.hasParallelDenseAndMoE && moe.numExperts == 128 && moe.topKExperts == 8)
        #expect(moe.kEqVLayers == [1, 3])
        #expect(!moe.hasPerLayerEmbeddings)
        #expect(moe.blastRadiusWarning(for: "model.layers.1.self_attn.k_proj")?.contains("attention_k_eq_v") == true)
        #expect(moe.blastRadiusWarning(for: "model.layers.0.self_attn.k_proj") == nil)

        #expect(Gemma4EditTopology(config: ["model_type": "qwen3_5_text", "num_hidden_layers": 4]) == nil)
    }

    @Test("Validator blocks Gemma 4 PLE embeddings and multimodal towers under textBackboneOnly while allowing multimodal towers under includeMultimodal")
    func validatorEnforcesGemma4TargetScope() throws {
        let forbidden: [(String, EditForbiddenTargetKind)] = [
            ("model.embed_tokens_per_layer.weight", .embedding),
            ("language_model.model.per_layer_model_projection.weight", .perLayerProjection),
            ("language_model.model.per_layer_projection_norm.weight", .perLayerProjection),
            ("vision_tower.encoder.layers.0.self_attn.o_proj", .multimodalTower),
            ("audio_tower.layers.0.self_attn.o_proj", .multimodalTower),
            ("embed_vision.embedding_projection", .multimodalTower),
            ("embed_audio.embedding_projection", .multimodalTower),
            ("vision_embedder.proj", .multimodalTower),
            ("multi_modal_projector.linear", .multimodalTower),
        ]
        for (target, kind) in forbidden {
            let err = #expect(throws: EditValidationError.self) {
                try EditProgramValidator.validate(EditProgram(ops: [.rank1Damp(module: target, vectorKey: "u", alpha: 0.2)]))
            }
            #expect(err == .forbiddenTarget(target: target, kind: kind), "\(target)")
        }

        try EditProgramValidator.validate(
            EditProgram(ops: [.rank1Damp(module: "vision_tower.encoder.layers.0.self_attn.o_proj", vectorKey: "u", alpha: 0.2)])
                .withPolicy(targetScope: .includeMultimodal)
        )
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.shift(tensor: "model.embed_tokens.weight", start: 0, count: 1, valueKey: "v")])
                    .withPolicy(targetScope: .includeMultimodal)
            )
        }
    }

    @Test("Validator closes router.scale hole, permits partial per_expert_scale shift, validates maxAlpha, and decodes legacy edit.json")
    func validatorEnforcesScaleAndAlphaPolicy() throws {
        let routerScaleErr = #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: "model.layers.0.router.scale", start: 0, count: 4, valueKey: "v")]))
        }
        guard case .wholeTensorRescale? = routerScaleErr else {
            Issue.record("expected wholeTensorRescale on router.scale, got \(String(describing: routerScaleErr))")
            return
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: "model.layers.0.layer_scalar", start: 0, count: 1, valueKey: "v")]))
        }
        try EditProgramValidator.validate(
            EditProgram(ops: [.shift(tensor: Self.routerPerExpertScale, start: 1, count: 2, valueKey: "v")]),
            tensorShapes: [Self.routerPerExpertScale: [8]],
            tensorValues: ["v": [0.1, -0.1]]
        )
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.shift(tensor: Self.routerPerExpertScale, start: 0, count: 8, valueKey: "v")]),
                tensorShapes: [Self.routerPerExpertScale: [8]]
            )
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.clip(tensor: Self.routerPerExpertScale, factor: 2.0)]))
        }
        try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: Self.routerPerExpertScale, start: 1, count: 2, valueKey: "v")]))
        try EditProgramValidator.validate(
            EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 5.0)]).withPolicy(maxAlpha: 5.0)
        )
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 5.1)]).withPolicy(maxAlpha: 5.0)
            )
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 1.0)]).withPolicy(maxAlpha: 10.5)
            )
        }

        let legacyJSON = #"{"format_version":1,"session_id":"s","base_model_fingerprint":"fp","approval":"pending","ops":[{"op":"clip","tensor":"model.layers.0.mlp.down_proj.weight","factor":2.0}]}"#
        let decoded = try JSONDecoder().decode(EditProgram.self, from: Data(legacyJSON.utf8))
        #expect(decoded.targetScope == nil && decoded.maxAlpha == nil)
    }
}

extension Gemma4SelfRewritingP0Tests {
    @Test("WeightCopy isolates edits and restores bit-exact logits across Gemma4TextModel, Gemma4KEqV, Gemma4Model, MLXVLM.Gemma4, and Gemma4Unified")
    func weightCopyCoversAllFourGemma4ModelClasses() throws {
        let cases: [(name: String, model: any LanguageModel, oProj: String)] = [
            ("gemma4", try TinyModel.make(.gemma4), Self.textOProj),
            ("gemma4KEqV", try TinyModel.make(.gemma4KEqV), Self.textOProj),
            ("gemma4LLMWrapper", try Self.makeTinyGemma4LLMWrapper(), Self.wrappedOProj),
            ("gemma4VLM", try Self.makeTinyGemma4VLM(), Self.wrappedOProj),
            ("gemma4Unified", try TinyModel.make(.gemma4Unified), Self.wrappedOProj),
        ]
        for entry in cases {
            let origBits = try Self.logitsBits(entry.model)
            var copy = try WeightCopy.makeSync(from: entry.model)
            let preBits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
            #expect(preBits == origBits, "\(entry.name) pre-edit logits mismatch")

            let u = Self.unitVector(size: 64, seed: 780)
            let program = EditProgram(ops: [
                .rank1Damp(module: entry.oProj, vectorKey: "u", alpha: 0.3),
                .clip(tensor: entry.oProj + ".weight", factor: 0.75),
            ])
            _ = try copy.apply(program, tensors: ["u": u], verifying: entry.model)
            let editedBits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
            #expect(editedBits != origBits, "\(entry.name) edit did not change logits")
            #expect(try Self.logitsBits(entry.model) == origBits, "\(entry.name) original mutated")

            try copy.undoAll(verifying: entry.model)
            let undoneBits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
            #expect(undoneBits == origBits, "\(entry.name) undo not bit-exact")
            try copy.release(verifying: entry.model)
        }
    }

    @Test("Gemma 4 MoE WeightCopy clones SwitchLinear, QuantizedSwitchLinear, and MACaQSwitchLinear cleanly and isolates partial per_expert_scale shifts")
    func gemma4MoeSwitchesAndPartialExpertScaleShift() throws {
        let floatMoe = try TinyModel.make(.gemma4Moe)
        let origScale = try #require(floatMoe.parameters().flattened().first(where: { $0.0 == Self.routerPerExpertScale })?.1.asArray(Float.self))
        #expect(origScale.count == 8)
        let floatOrigBits = try Self.logitsBits(floatMoe)

        var floatCopy = try WeightCopy.makeSync(from: floatMoe)
        let shiftProg = EditProgram(ops: [.shift(tensor: Self.routerPerExpertScale, start: 1, count: 2, valueKey: "d")])
        _ = try floatCopy.apply(shiftProg, tensors: ["d": MLXArray([Float(0.6), Float(-0.4)])], verifying: floatMoe)
        let editedScale = try #require((try floatCopy.model).parameters().flattened().first(where: { $0.0 == Self.routerPerExpertScale })?.1.asArray(Float.self))
        #expect(editedScale[0].bitPattern == origScale[0].bitPattern)
        #expect(abs(editedScale[1] - (origScale[1] + 0.6)) < 1e-6)
        #expect(abs(editedScale[2] - (origScale[2] - 0.4)) < 1e-6)
        #expect(editedScale[3].bitPattern == origScale[3].bitPattern)
        try floatCopy.undoAll(verifying: floatMoe)
        #expect(try floatCopy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == floatOrigBits)
        try floatCopy.release(verifying: floatMoe)

        let routerAdapter = try EditAdapter(program: shiftProg, tensors: ["d": MLXArray([Float(0.6), Float(-0.4)])])
        try floatMoe.load(adapter: routerAdapter)
        #expect(try Self.logitsBits(floatMoe) != floatOrigBits)
        floatMoe.unload(adapter: routerAdapter)
        #expect(try Self.logitsBits(floatMoe) == floatOrigBits)

        let q4Moe = try TinyModel.make(.gemma4Moe)
        q4Moe.update(parameters: q4Moe.parameters().mapValues { $0.asType(.float16) })
        Self.quantizeDivisible32(q4Moe)
        try q4Moe.prepare()
        eval(q4Moe)
        #expect(q4Moe.namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 is QuantizedSwitchLinear)
        let q4Bits = try Self.logitsBits(q4Moe)
        var q4Copy = try WeightCopy.makeSync(from: q4Moe)
        #expect((try q4Copy.model).namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 is QuantizedSwitchLinear)
        #expect(try q4Copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == q4Bits)
        _ = try q4Copy.apply(EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 0.25)]), tensors: ["u": Self.unitVector(size: 64, seed: 781)], verifying: q4Moe)
        try q4Copy.undoAll(verifying: q4Moe)
        #expect(try q4Copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == q4Bits)
        try q4Copy.release(verifying: q4Moe)

        let macMoe = try TinyModel.make(.gemma4Moe)
        let origSwitch = try #require(macMoe.namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 as? SwitchLinear)
        let switchSpec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
        try MLXInferenceEngine.replace(Self.moeSwitchGate, with: MACaQSwitchLinear(replacing: origSwitch, spec: switchSpec), in: macMoe)
        let macBits = try Self.logitsBits(macMoe)
        let macCopy = try WeightCopy.makeSync(from: macMoe)
        #expect((try macCopy.model).namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 is MACaQSwitchLinear)
        #expect(try macCopy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == macBits)
        try macCopy.release(verifying: macMoe)

        let nf4Moe = try TinyModel.make(.gemma4Moe)
        let origNf4Switch = try #require(nf4Moe.namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 as? SwitchLinear)
        try MLXInferenceEngine.replace(Self.moeSwitchGate, with: NF4SwitchLinear(origNf4Switch, groupSize: 32), in: nf4Moe)
        let nf4MoeBits = try Self.logitsBits(nf4Moe)
        let nf4MoeCopy = try WeightCopy.makeSync(from: nf4Moe)
        #expect((try nf4MoeCopy.model).namedModules().first(where: { $0.0 == Self.moeSwitchGate })?.1 is NF4SwitchLinear)
        #expect(try nf4MoeCopy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == nf4MoeBits)
        try nf4MoeCopy.release(verifying: nf4Moe)
    }
}

extension Gemma4SelfRewritingP0Tests {
    @Test("MACaQQuantizedLinear (MXFP4 + affine, with preScale/Hadamard/rowScale) supports exact rank1Damp, stacked edits, clip, rows, fuse, round-trip, and cloning")
    func macaqRank1DampClipRowsFuseAndCloning() throws {
        let mxfp4MAC = Self.makeMACaQLinear(outputs: 64, inputs: 64, mode: .mxfp4, withRowScale: true, withBias: true)
        try Self.verifyLayerRank1DampParity(base: mxfp4MAC, path: Self.textOProj, alpha: 0.35, tolerance: 2e-2, seed: 790)

        let affineMAC = Self.makeMACaQLinear(outputs: 64, inputs: 64, mode: .affine, withRowScale: true, withBias: false)
        try Self.verifyLayerRank1DampParity(base: affineMAC, path: Self.textOProj, alpha: 0.35, tolerance: 5e-3, seed: 791)

        let model = try TinyModel.make(.gemma4)
        let macLayer = Self.makeMACaQLinear(outputs: 64, inputs: 64, mode: .mxfp4, withRowScale: true, withBias: true)
        try MLXInferenceEngine.replace(Self.textOProj, with: macLayer, in: model)
        let origBits = try Self.logitsBits(model)

        var copy = try WeightCopy.makeSync(from: model)
        let baseline = try copy.measureRoundTrip(tensor: Self.textOProj + ".weight")
        #expect(baseline.format == "4-bit macaq-mxfp4 g32 h32")
        #expect(baseline.rmse.isFinite && baseline.rmse < 5e-2)

        let u1 = Self.unitVector(size: 64, seed: 792)
        let u2 = Self.unitVector(size: 64, seed: 793)
        _ = try copy.apply(EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u1", alpha: 0.25)]), tensors: ["u1": u1], verifying: model)
        let afterFirst = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect((try copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 is EditRankOneLinear)

        let nestedCopy = try WeightCopy.makeSync(from: try copy.model)
        #expect(try nestedCopy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == afterFirst)
        try nestedCopy.release(verifying: try copy.model)

        _ = try copy.apply(EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u2", alpha: 0.2)]), tensors: ["u2": u2], verifying: model)
        try copy.undoLast(verifying: model)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == afterFirst)

        let clipMetrics = try copy.apply(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.0)]), tensors: [:], verifying: model)
        #expect(clipMetrics[0].rmse.isFinite && clipMetrics[0].relativeError < 0.35)
        let repackedMAC = try #require((try copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 as? MACaQQuantizedLinear)
        #expect(repackedMAC.hadamardBlock == 32 && repackedMAC.rowScale != nil && repackedMAC.bias != nil)
        _ = try copy.apply(
            EditProgram(ops: [.shift(tensor: Self.textOProj + ".bias", start: 0, count: 4, valueKey: "bd")]),
            tensors: ["bd": MLXArray([Float(0.1), Float(-0.1), Float(0.2), Float(-0.2)])],
            verifying: model
        )

        try copy.undoAll(verifying: model)
        #expect((try copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 is MACaQQuantizedLinear)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == origBits)
        try copy.release(verifying: model)

        let fuseTarget = try TinyModel.make(.gemma4)
        try MLXInferenceEngine.replace(Self.textOProj, with: Self.makeMACaQLinear(outputs: 64, inputs: 64, mode: .affine, withRowScale: true, withBias: false), in: fuseTarget)
        let adapter = try EditAdapter(program: EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 0.3)]), tensors: ["u": u1])
        try adapter.fuse(with: fuseTarget)
        #expect(fuseTarget.namedModules().first(where: { $0.0 == Self.textOProj })?.1 is MACaQQuantizedLinear)
    }

    @Test("NVFP4 QuantizedLinear (with globalScale) and SmallMQuantizedLinear preserve scale and type across rank1Damp, clip, round-trip, and undo")
    func nvfp4AndSmallMParityAndPreservation() throws {
        let nvfp4 = Self.makeNVFP4Linear(outputs: 64, inputs: 64, globalScaleValue: 3.25, withBias: true)
        try Self.verifyLayerRank1DampParity(base: nvfp4, path: Self.textOProj, alpha: 0.35, tolerance: 2e-2, seed: 801)

        let nvfp4Model = try TinyModel.make(.gemma4)
        try MLXInferenceEngine.replace(Self.textOProj, with: Self.makeNVFP4Linear(outputs: 64, inputs: 64, globalScaleValue: 3.25, withBias: false), in: nvfp4Model)
        var nvfp4Copy = try WeightCopy.makeSync(from: nvfp4Model)
        let nvfp4RT = try nvfp4Copy.measureRoundTrip(tensor: Self.textOProj + ".weight")
        #expect(nvfp4RT.rmse.isFinite && nvfp4RT.rmse < 5e-2)
        _ = try nvfp4Copy.apply(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]), tensors: [:], verifying: nvfp4Model)
        let clippedNV = try #require((try nvfp4Copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 as? QuantizedLinear)
        #expect(clippedNV.mode == .nvfp4 && clippedNV.globalScale != nil)
        try nvfp4Copy.release(verifying: nvfp4Model)

        let smallMModel = try TinyModel.make(.gemma4)
        smallMModel.update(parameters: smallMModel.parameters().mapValues { $0.asType(.float16) })
        Self.quantizeDivisible32(smallMModel)
        #expect(SmallMQuantizedLinear.install(in: smallMModel) > 0)
        try smallMModel.prepare()
        eval(smallMModel)
        #expect(smallMModel.namedModules().first(where: { $0.0 == Self.textOProj })?.1 is SmallMQuantizedLinear)
        let smallMBits = try Self.logitsBits(smallMModel)
        var smallMCopy = try WeightCopy.makeSync(from: smallMModel)
        let smallMRT = try smallMCopy.measureRoundTrip(tensor: Self.textOProj + ".weight")
        #expect(smallMRT.format.contains("smallm") && smallMRT.rmse.isFinite)
        _ = try smallMCopy.apply(
            EditProgram(ops: [
                .rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 0.25),
                .clip(tensor: Self.textDownProj + ".weight", factor: 1.0),
            ]),
            tensors: ["u": Self.unitVector(size: 64, seed: 802)],
            verifying: smallMModel
        )
        #expect((try smallMCopy.model).namedModules().first(where: { $0.0 == Self.textDownProj })?.1 is SmallMQuantizedLinear)
        try smallMCopy.undoAll(verifying: smallMModel)
        #expect((try smallMCopy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 is SmallMQuantizedLinear)
        #expect((try smallMCopy.model).namedModules().first(where: { $0.0 == Self.textDownProj })?.1 is SmallMQuantizedLinear)
        #expect(try smallMCopy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == smallMBits)
        try smallMCopy.release(verifying: smallMModel)
    }
}

extension Gemma4SelfRewritingP0Tests {
    @Test("NF4Linear preserves type across rank1Damp, clip, bias shift, and undo, with exact budget accounting")
    func nf4Rank1DampClipShiftAndBudget() throws {
        let nf4Base = withRandomState(MLXRandom.RandomState(seed: 803)) {
            NF4Linear(
                dense: (MLXRandom.normal([64, 64]).asType(.float32) * 0.25).asType(.float16),
                bias: (MLXRandom.normal([64]).asType(.float32) * 0.05).asType(.float16),
                groupSize: 32
            )
        }
        try Self.verifyLayerRank1DampParity(base: nf4Base, path: Self.textOProj, alpha: 0.3, tolerance: 5e-3, seed: 804)
        let nf4Model = try TinyModel.make(.gemma4)
        try MLXInferenceEngine.replace(Self.textOProj, with: nf4Base, in: nf4Model)
        var nf4Copy = try WeightCopy.makeSync(from: nf4Model)
        let nf4Prog = EditProgram(ops: [.rank1Damp(module: Self.textOProj, vectorKey: "u", alpha: 0.2)])
        let nf4Plan = try nf4Copy.planBudget(for: nf4Prog)
        _ = try nf4Copy.apply(nf4Prog, tensors: ["u": Self.unitVector(size: 64, seed: 805)], verifying: nf4Model)
        #expect(nf4Copy.accounting.totalExtraBytes == nf4Plan.totalExtraBytes)
        _ = try nf4Copy.apply(EditProgram(ops: [.clip(tensor: Self.textOProj + ".weight", factor: 1.5)]), tensors: [:], verifying: nf4Model)
        _ = try nf4Copy.apply(
            EditProgram(ops: [.shift(tensor: Self.textOProj + ".bias", start: 0, count: 2, valueKey: "b")]),
            tensors: ["b": MLXArray([Float(0.25), Float(-0.25)])],
            verifying: nf4Model
        )
        #expect((try nf4Copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 is NF4Linear)
        try nf4Copy.undoAll(verifying: nf4Model)
        #expect((try nf4Copy.model).namedModules().first(where: { $0.0 == Self.textOProj })?.1 is NF4Linear)
        try nf4Copy.release(verifying: nf4Model)
    }
}
