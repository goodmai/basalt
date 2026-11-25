import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

private actor RecordingConvertService: ModelConversionService {
    private(set) var lastRequest: BasaltCore::ModelConversionRequest?

    func plan(
        source: String,
        target: QuantizationTarget,
        budgetGB: Double?,
        pvTune: Bool,
        cacheRoot: URL
    ) async throws -> BasaltCore::ModelConversionPlan {
        BasaltCore::ModelConversionPlan(
            sourceId: source,
            sourceFormat: "mlx",
            target: target.specString,
            supportsPVTune: target.supportsPVTune,
            footprint: .init(
                sourceBytes: 1024,
                estimatedOutputBytes: 512,
                peakConversionRAMBytes: 2048,
                availableRAMBytes: 8192,
                fitsConversionOnThisMac: true,
                outputFitsInference: true
            ),
            warnings: []
        )
    }

    func convert(
        _ request: BasaltCore::ModelConversionRequest,
        cacheRoot: URL,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> BasaltCore::ModelConversionResult {
        lastRequest = request
        return BasaltCore::ModelConversionResult(outputId: request.outputId, bytesBefore: 1024, bytesAfter: 512, shards: 1)
    }
}

@Suite("PEFT & mlx-lm fused export: WeightConverter, UniversalModelConverter, CLI, REST (#664), and ConvertModelSheet (#675 P2, #228)", .serialized)
struct PEFTFusedExportTests {
    static func writeTinyCheckpoint(_ family: TinyModel.Family, to dir: URL) throws -> any LanguageModel {
        let model = try TinyModel.make(family)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var flat: [String: MLXArray] = [:]
        for (k, v) in model.parameters().flattened() {
            flat[k] = v.asType(.bfloat16)
        }
        try save(arrays: flat, metadata: ["format": "mlx"], url: dir.appending(component: "model.safetensors"))
        try Data(family.config.utf8).write(to: dir.appending(component: "config.json"))
        return model
    }

    static func writeMLXAdapter(pairs: [PEFTAdapterTests.Pair], to dir: URL, scale: Float = 2.0) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let keys = Set(pairs.compactMap { PEFTLoRAReader.layerRelative($0.path) }).sorted()
        let encodedKeys = String(decoding: try JSONEncoder().encode(keys), as: UTF8.self)
        let cfg = #"{"fine_tune_type":"lora","num_layers":1024,"lora_parameters":{"rank":\#(PEFTAdapterTests.rank),"scale":\#(scale),"keys":\#(encodedKeys)}}"#
        try Data(cfg.utf8).write(to: dir.appending(component: "adapter_config.json"))
        var arrays: [String: MLXArray] = [:]
        for pair in pairs {
            arrays["\(pair.path).lora_a"] = pair.a.T
            arrays["\(pair.path).lora_b"] = pair.b.T
        }
        try save(arrays: arrays, url: dir.appending(component: "adapters.safetensors"))
    }

    static func loadFusedModel(_ family: TinyModel.Family, from dir: URL) throws -> any LanguageModel {
        let weights = try WeightConverterTests.weights(dir)
        let model = try TinyModel.make(family)
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
        return model
    }
}

extension PEFTFusedExportTests {
    @Test("Fused bf16 PEFT export matches W + s·B·A bit-identically, equals mlx-lm fused export, and quantizes cleanly to affine 4-bit on .qwen35 and .lfm2Moe",
          arguments: [TinyModel.Family.qwen35, .lfm2Moe])
    func fusedDenseParityAndQuantizedExport(family: TinyModel.Family) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "peft-fuse-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let srcDir = root.appending(component: "base")
        let model = try Self.writeTinyCheckpoint(family, to: srcDir)
        let pairs = PEFTAdapterTests.pairs(model, family: family)
        let peftDir = try PEFTAdapterTests.directory(pairs, config: PEFTAdapterTests.config())
        defer { try? FileManager.default.removeItem(at: peftDir) }
        let mlxDir = root.appending(component: "mlx-adapter")
        try Self.writeMLXAdapter(pairs: pairs, to: mlxDir)

        let cacheRoot = root.appending(component: "hf-cache")
        let snapDir = cacheRoot.appending(path: "models--harness-hub--tiny-peft/snapshots/main")
        try FileManager.default.createDirectory(at: snapDir.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: peftDir, to: snapDir)

        let fusedPEFTDir = root.appending(component: "fused-peft-bf16")
        var denseReq = ModelConversionRequest(source: srcDir.path, outputId: "local/fused", target: .denseBF16, stream: false)
        denseReq.adapter = "hf:harness-hub/tiny-peft"
        _ = try await UniversalModelConverter.convertToDirectory(request: denseReq, sourceURL: srcDir, outputURL: fusedPEFTDir, cacheRoot: cacheRoot) { _ in }

        let fusedMLXDir = root.appending(component: "fused-mlx-bf16")
        _ = try WeightConverter(precision: .dense(.bfloat16), adapter: mlxDir).run(source: srcDir, output: fusedMLXDir)

        let baseWeights = try loadArrays(url: srcDir.appending(component: "model.safetensors"))
        let fusedWeights = try WeightConverterTests.weights(fusedPEFTDir)
        for pair in pairs {
            let expected = (baseWeights["\(pair.path).weight"]!.asType(.float32) + 2.0 * matmul(pair.b.asType(.float32), pair.a.asType(.float32))).asType(.bfloat16)
            let actual = try #require(fusedWeights["\(pair.path).weight"])
            #expect(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self) == 0)
        }

        let fusedPEFT = try Self.loadFusedModel(family, from: fusedPEFTDir)
        let fusedMLX = try Self.loadFusedModel(family, from: fusedMLXDir)
        let peftLogits = fusedPEFT(PEFTAdapterTests.tokens, cache: nil).asType(.float32)
        let mlxLogits = fusedMLX(PEFTAdapterTests.tokens, cache: nil).asType(.float32)
        #expect(abs(peftLogits - mlxLogits).max().item(Float.self) == 0)
        #expect(argMax(peftLogits, axis: -1).asArray(Int32.self) == argMax(mlxLogits, axis: -1).asArray(Int32.self))

        let fusedQ4Dir = root.appending(component: "fused-q4")
        var q4Req = ModelConversionRequest(source: srcDir.path, outputId: "local/fused-q4", target: .affine(bits: 4, groupSize: 32), stream: false)
        q4Req.adapter = "harness-hub/tiny-peft"
        _ = try await UniversalModelConverter.convertToDirectory(request: q4Req, sourceURL: srcDir, outputURL: fusedQ4Dir, cacheRoot: cacheRoot) { _ in }
        let q4Weights = try WeightConverterTests.weights(fusedQ4Dir)
        #expect(q4Weights.keys.contains { $0.hasSuffix(".scales") })
        let deqDir = root.appending(component: "fused-q4-deq")
        _ = try WeightConverter(precision: .dense(.float32)).run(source: fusedQ4Dir, output: deqDir)
        let deqWeights = try WeightConverterTests.weights(deqDir)
        for pair in pairs {
            let q4W = try #require(deqWeights["\(pair.path).weight"])
            let refW = try #require(fusedWeights["\(pair.path).weight"]).asType(.float32)
            #expect(abs(q4W - refW).mean().item(Float.self) < 0.05)
        }
    }

    @Test("Fused float32 PEFT export matches unfused PEFT model within 1e-4 (#228)", arguments: [TinyModel.Family.qwen35, .lfm2Moe])
    func fusedFloat32ExactParity(family: TinyModel.Family) throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "peft-f32-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = try TinyModel.make(family)
        let pairs = PEFTAdapterTests.pairs(model, family: family)
        let peftDir = try PEFTAdapterTests.directory(pairs, config: PEFTAdapterTests.config())
        defer { try? FileManager.default.removeItem(at: peftDir) }

        let f32Dir = root.appending(component: "base-f32")
        try FileManager.default.createDirectory(at: f32Dir, withIntermediateDirectories: true)
        var f32Flat: [String: MLXArray] = [:]
        for (k, v) in model.parameters().flattened() { f32Flat[k] = v.asType(.float32) }
        try save(arrays: f32Flat, metadata: ["format": "mlx"], url: f32Dir.appending(component: "model.safetensors"))
        try Data(family.config.utf8).write(to: f32Dir.appending(component: "config.json"))

        let fusedF32Dir = root.appending(component: "fused-f32")
        _ = try WeightConverter(precision: .source, adapter: peftDir).run(source: f32Dir, output: fusedF32Dir)
        let unfusedF32 = try TinyModel.make(family)
        _ = try MLXInferenceEngine.attachAnyAdapterDirectory(peftDir, to: unfusedF32)
        let fusedF32 = try Self.loadFusedModel(family, from: fusedF32Dir)
        let tokens = PEFTAdapterTests.tokens
        let diff = abs(fusedF32(tokens, cache: nil) - unfusedF32(tokens, cache: nil)).max().item(Float.self)
        #expect(diff < 1e-4)
        #expect(argMax(fusedF32(tokens, cache: nil), axis: -1).asArray(Int32.self) == argMax(unfusedF32(tokens, cache: nil), axis: -1).asArray(Int32.self))
    }

    @Test("LoRAUpdate PEFT reader honors rsLoRA scale and refuses DoRA, modules_to_save, rank_pattern, unmatched key, and lone lora_A")
    func peftLoRAUpdateRefusalsAndRsLoRA() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "peft-refuse-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let srcDir = root.appending(component: "base")
        let model = try Self.writeTinyCheckpoint(.qwen35, to: srcDir)
        let pairs = PEFTAdapterTests.pairs(model, family: .qwen35)
        let shard = srcDir.appending(component: "model.safetensors")
        let keys = Dictionary(uniqueKeysWithValues: try loadArrays(url: shard).keys.map { ($0, shard) })

        let rsCfg = PEFTAdapterTests.config(alpha: 8).replacingOccurrences(of: #""use_rslora": false"#, with: #""use_rslora": true"#)
        let rsDir = try PEFTAdapterTests.directory(pairs, config: rsCfg)
        defer { try? FileManager.default.removeItem(at: rsDir) }
        let rsUpdate = try LoRAUpdate(directory: rsDir, keys: keys)
        #expect(rsUpdate.scale == 4.0)

        let badConfigs = [
            PEFTAdapterTests.config().replacingOccurrences(of: #""use_dora": false"#, with: #""use_dora": true"#),
            PEFTAdapterTests.config().replacingOccurrences(of: #""modules_to_save": null"#, with: #""modules_to_save": ["lm_head"]"#),
            PEFTAdapterTests.config().replacingOccurrences(of: #""rank_pattern": {}"#, with: #""rank_pattern": {"q_proj": 8}"#),
        ]
        for bad in badConfigs {
            let dir = try PEFTAdapterTests.directory(pairs, config: bad)
            defer { try? FileManager.default.removeItem(at: dir) }
            #expect(throws: WeightConverter.Failure.self) { _ = try LoRAUpdate(directory: dir, keys: keys) }
        }

        let badKeyDir = root.appending(component: "unmatched")
        try FileManager.default.createDirectory(at: badKeyDir, withIntermediateDirectories: true)
        try Data(PEFTAdapterTests.config().utf8).write(to: badKeyDir.appending(component: "adapter_config.json"))
        try save(arrays: [
            "base_model.model.model.layers.99.self_attn.q_proj.lora_A.weight": MLXArray.zeros([4, 8]),
            "base_model.model.model.layers.99.self_attn.q_proj.lora_B.weight": MLXArray.zeros([8, 4]),
        ], url: badKeyDir.appending(component: PEFTLoRAReader.weightsFile))
        #expect(throws: WeightConverter.Failure.self) { _ = try LoRAUpdate(directory: badKeyDir, keys: keys) }
    }
}

extension PEFTFusedExportTests {
    @Test("ServedAdapterSource.parseSingle, REST /v1/models/convert repo-only adapter guard (#664), and ConvertModelSheet draft wiring")
    func restRepoGuardAndDraftWiring() async throws {
        #expect(try ServedAdapterSource.parseSingle("hf:harness-hub/Qwen3.8-27B-pi-adapted") == .repository("harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(try ServedAdapterSource.parseSingle("harness-hub/Qwen3.8-27B-pi-adapted") == .repository("harness-hub/Qwen3.8-27B-pi-adapted"))
        if case .directory(let url) = try ServedAdapterSource.parseSingle("/tmp/local-adapter") {
            #expect(url.path == "/tmp/local-adapter")
        } else {
            Issue.record("expected local directory")
        }
        #expect(throws: ServedAdapterSource.Invalid.self) { _ = try ServedAdapterSource.parseSingle("   ") }

        let service = RecordingConvertService()
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orchestrator.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: service) { _, _ in FileManager.default.temporaryDirectory }
        let server = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "mock")).withModelHub(hub)
        let app = Application(router: server.makeRouter())
        try await app.test(.router) { client in
            let localBody = Data(#"{"source":"org/base","output_id":"local/out","target":"dense:bf16","adapter":"/tmp/secret","stream":false}"#.utf8)
            let localStatus = try await client.execute(
                uri: "/v1/models/convert", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: localBody)
            ) { $0.status }
            #expect(localStatus == .badRequest)

            let repoBody = Data(#"{"source":"org/base","output_id":"local/out","target":"dense:bf16","adapter":"hf:harness-hub/Qwen3.8-27B-pi-adapted","stream":false}"#.utf8)
            let repoStatus = try await client.execute(
                uri: "/v1/models/convert", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(bytes: repoBody)
            ) { $0.status }
            #expect(repoStatus == .ok)
        }
        #expect(await service.lastRequest?.adapter == "harness-hub/Qwen3.8-27B-pi-adapted")

        var draft = ConversionDraft(sourceId: "org/base", sourceBytes: 4096)
        #expect(draft.toRequest().adapter == nil)
        #expect(ConvertModelSheet.isValidOptionalAdapter(""))
        #expect(ConvertModelSheet.isValidOptionalAdapter("harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(ConvertModelSheet.isValidOptionalAdapter("hf:harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(!ConvertModelSheet.isValidOptionalAdapter("/etc/passwd"))
        draft.adapterRepoId = " harness-hub/Qwen3.8-27B-pi-adapted "
        #expect(draft.toRequest().adapter == "harness-hub/Qwen3.8-27B-pi-adapted")

        var badState = LibraryState()
        var badDraft = ConversionDraft(sourceId: "org/base", sourceBytes: 4096)
        badDraft.adapterRepoId = "/etc/passwd"
        badState.convertDraft = badDraft
        let effects = LibraryReducer.reduce(&badState, .startConversion)
        #expect(effects.isEmpty && badState.problem?.contains("org/name") == true)
    }

    @MainActor
    @Test("ConvertModelSheet renders optional LoRA adapter section cleanly in Light, Dark, and Russian")
    func renderConvertModelSheetWithAdapter() throws {
        var state = LibraryState()
        var draft = ConversionDraft(sourceId: "lmstudio-community/Qwen3.8-27B-MLX-4bit", sourceBytes: 15_500_000_000)
        draft.adapterRepoId = "harness-hub/Qwen3.8-27B-pi-adapted"
        state.convertDraft = draft
        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })
        let store = LibraryStore(state, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
        let outDir = ProcessInfo.processInfo.environment["BASALT_SAVE_GUI_675"].map { URL(fileURLWithPath: $0) }
        if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }
        for (name, variant) in [("light", TraitVariant.lightStandard), ("dark", .darkStandard), ("ru", .darkRussian)] {
            let bmp = try OffscreenSceneRenderer.render(
                ConvertModelSheet(store: store), spec: .init(id: .libraryMixed, width: 580, height: 700), variant: variant)
            #expect(bmp.width == 1_160 && bmp.height == 1_400 && !bmp.png.isEmpty)
            if let outDir {
                try bmp.png.write(to: outDir.appending(component: "convert-sheet-adapter-\(name).png"))
            }
        }
    }
}
