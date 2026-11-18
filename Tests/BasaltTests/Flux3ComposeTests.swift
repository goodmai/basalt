import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCloud
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("FLUX.3 4-stage rewriter-encoder-painter-decoder Compose profiles & modular MACaQ DiT (#539)", .serialized)
struct Flux3ComposeTests {
    @Test("All 5 curated FLUX.3 4-stage Compose combinations wire distinct rewriter, Qwen3-VL encoder, FLUX.3 painter, and VAE decoder roles")
    func curatedFlux3CombinationsParseAndPlan() throws {
        let presets = ComposeProfileCatalog.flux3Combinations
        #expect(presets.map(\.id) == [
            "flux3-16gb-fast",
            "flux3-24gb-balanced",
            "flux3-36gb-moe-vision",
            "flux3-64gb-studio-base",
            "flux3-128gb-reference",
        ])
        for preset in presets {
            let spec = try ComposeFile.load(preset.id)
            #expect(spec.issues.isEmpty)
            let imagine = try #require(spec.imagine)
            #expect(imagine.upsampleVia == "rewriter" && imagine.shareEncoder == "encoder")
            #expect(imagine.decoderVia == "vae" && imagine.chain.first == "painter")
            #expect(spec.imagineDecoderModels == ["vae"])
            let sizes = ComposeSpec.Sizes(bytes: { _ in 2_000_000_000 }, onNeuralEngine: { _ in false })
            let (workloads, unknown) = spec.workloads(sizes)
            #expect(unknown.isEmpty && workloads.contains { $0.name == "vae" && $0.role == "imagine_decoder" })
        }
    }

    @Test("ComposeFile rejects undeclared or cloud decoder_via models and tracks decoder role diffs")
    func validatesDecoderViaAndDiffsRoleChanges() throws {
        let missingDecoder = """
            version: 2
            name: bad-decoder
            models:
              rewriter: { source: mlx-community/Qwen3.5-9B-4bit }
              painter: { source: goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX }
            pipeline:
              answer: rewriter
              imagine: { chain: [painter], decoder_via: ghost_vae }
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(missingDecoder, file: "bad_dec1.yaml") }

        let cloudDecoder = """
            version: 2
            name: bad-cloud-decoder
            models:
              rewriter: { source: mlx-community/Qwen3.5-9B-4bit }
              painter: { source: goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX }
              vae: { source: cloud:openrouter/org/vae, api_key_env: OPENROUTER_API_KEY }
            pipeline:
              answer: rewriter
              imagine: { chain: [painter], decoder_via: vae }
            """
        #expect(throws: ServeConfigError.self) { try ComposeFile.parse(cloudDecoder, file: "bad_dec2.yaml") }

        let diff = ComposeDiff.between(try ComposeFile.load("flux3-16gb-fast"), try ComposeFile.load("flux3-24gb-balanced"))
        #expect(diff.reloaded.contains("vae"))
    }

    @Test("ImageModelDescriptor and ImageRunnerCore parse modular Flux3Transformer2DModel, Klein Base 50-step, MACaQ 4-bit, and FLUX.2-small-decoder")
    func modularFlux3AndSmallDecoderInspection() async throws {
        let macaqDesc = ImageModelDescriptor.detect(identifier: "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX")
        #expect(macaqDesc.family == .flux3Klein && macaqDesc.minimumRecommendedBits == 4)
        #expect(macaqDesc.estimateFootprint(quantizationBits: 4).qualityWarning == nil)
        #expect(ImageModelDescriptor.detect(identifier: "black-forest-labs/flux-3-image-klein-base").defaultSteps == 50)

        let (root, smallVaeDir) = try Self.makeModularFlux3MACaQFixture()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: smallVaeDir)
        }
        let (parsedDesc, parsedCfg) = try ImageRunnerCore.parseDirectoryConfig(
            directory: root, identifier: "custom/local-model", externalDecoderDirectory: smallVaeDir
        )
        #expect(parsedDesc.family == .flux3Klein && parsedCfg.decoderBaseChannels == 18)

        let synth = try await ImageSynthesizerLoader.load(
            identifier: "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX",
            directory: root,
            externalDecoderDirectory: smallVaeDir
        )
        let artifact = try await ImageSynthesizerLoader.probeImageSynthesizer(synth)
        #expect(!artifact.imageData.isEmpty && artifact.width == 64 && artifact.height == 64)
        await (synth as? FluxKleinSynthesizer)?.unload()
    }

    @Test("ComposeHost executes end-to-end 4-stage FLUX.3 pipeline with real TinyModel rewriter, dedicated Qwen3 encoder, MACaQ painter, and small VAE")
    func composeHostFourStageFlux3EndToEnd() async throws {
        let (painterDir, vaeDir) = try Self.makeTinyPainterAndSmallVAE()
        defer {
            try? FileManager.default.removeItem(at: painterDir)
            try? FileManager.default.removeItem(at: vaeDir)
        }
        let host = try await Self.bootFourStageHost(painterPath: painterDir.path, vaePath: vaeDir.path)
        let status = await host.status()
        #expect(status.models.first { $0.name == "vae" }?.state == ComposeModelState.ready)

        let imageSynth = try #require(await host.registry.system(ComposeSpec.defaultPipeline)?.image)
        let req = try ImageGenerationRequest(
            model: "painter", prompt: "volcanic basalt coastline at dusk",
            dimensions: ImageDimensions(width: 64, height: 64), steps: 2
        )
        let route = await imageSynth.resolveRoute(req)
        #expect(route.upsamplerModel == "rewriter" && route.sharedEncoderModel == "encoder" && route.decoderModel == "vae")

        var done: ImageCompletedArtifact?
        for try await event in try await imageSynth.generate(req) {
            if case .completed(let art) = event { done = art }
        }
        let artifact = try #require(done)
        #expect(!artifact.imageData.isEmpty && artifact.decoderModel == "vae")
    }
}

extension Flux3ComposeTests {
    fileprivate static func makeModularFlux3MACaQFixture() throws -> (root: URL, vae: URL) {
        let root = FileManager.default.temporaryDirectory.appending(path: "flux3-mod-\(UUID().uuidString)")
        let tfDir = root.appending(component: "transformer")
        let vaeDir = FileManager.default.temporaryDirectory.appending(path: "flux2-small-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tfDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vaeDir, withIntermediateDirectories: true)

        try Data(#"{"_class_name":"AutoencoderKLFlux2","latent_channels":16,"decoder_block_out_channels":[96,192,384,384]}"#.utf8)
            .write(to: vaeDir.appending(component: "config.json"))
        var cfg = ImageDiTConfig(modelID: "flux3-test", hiddenSize: 64, numHeads: 4, numLayers: 2, latentChannels: 16, patchSize: 2)
        cfg.decoderBaseChannels = 18
        let built = withRandomState(MLXRandom.RandomState(seed: cfg.initSeed)) {
            FluxKleinDiTModule(config: cfg, usePerceiver: false)
        }
        let layerSpec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .mxfp4, hadamardBlock: 32)
        let manifestModules = ["dual_blocks.0.img_qkv": layerSpec, "single_blocks.0.qkv": layerSpec]
        try MACaQNativeLoader.install(manifestModules, in: built)
        eval(built)

        let manifestObj = MACaQNativeLoader.Manifest(version: 1, alpha: 0.75, modules: manifestModules)
        let savedCfg: [String: Any] = [
            "_class_name": "Flux3Transformer2DModel", "attention_head_dim": 16, "num_attention_heads": 4,
            "num_layers": 1, "num_single_layers": 1, "in_channels": 16, "seed": cfg.initSeed,
            MACaQNativeLoader.configKey: try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifestObj)),
        ]
        try JSONSerialization.data(withJSONObject: savedCfg, options: [.sortedKeys])
            .write(to: tfDir.appending(component: "config.json"))
        try save(arrays: Dictionary(uniqueKeysWithValues: built.parameters().flattened()), metadata: ["format": "mlx"], url: tfDir.appending(component: "diffusion_pytorch_model.safetensors"))
        try save(arrays: Dictionary(uniqueKeysWithValues: built.vae.parameters().flattened()), metadata: ["format": "mlx"], url: vaeDir.appending(component: "diffusion_pytorch_model.safetensors"))
        return (root, vaeDir)
    }

    fileprivate static func makeTinyPainterAndSmallVAE() throws -> (painter: URL, vae: URL) {
        let painterDir = FileManager.default.temporaryDirectory.appending(path: "flux3-p-\(UUID().uuidString)")
        let vaeDir = FileManager.default.temporaryDirectory.appending(path: "flux3-v-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: painterDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vaeDir, withIntermediateDirectories: true)
        try Data(#"{"_class_name":"Flux3Transformer2DModel","hidden_size":64,"num_attention_heads":4,"num_layers":2,"latent_channels":16}"#.utf8)
            .write(to: painterDir.appending(component: "config.json"))
        try Data(#"{"_class_name":"AutoencoderKLFlux2","latent_channels":16,"decoder_block_out_channels":[96,192,384,384]}"#.utf8)
            .write(to: vaeDir.appending(component: "config.json"))
        return (painterDir, vaeDir)
    }

    fileprivate static func bootFourStageHost(painterPath: String, vaePath: String) async throws -> ComposeHost {
        let rewriterOrch = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen35), telemetry: TelemetryTracker())
        let encoderOrch = ModelOrchestratorActor(engine: try await TinyModel.engine(.qwen3Small), telemetry: TelemetryTracker())
        let yaml = """
            version: 2
            name: flux3-four-stage-e2e
            memory: auto
            models:
              rewriter: { source: tiny/qwen35 }
              encoder: { source: tiny/qwen3-vl }
              painter: { source: goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX }
              vae: { source: black-forest-labs/FLUX.2-small-decoder }
            pipeline:
              answer: rewriter
              imagine:
                chain: [painter]
                upsample_via: rewriter
                share_encoder: encoder
                decoder_via: vae
                steps: 2
                size: 64x64
            """
        let spec = try ComposeFile.parse(yaml, file: "flux3-e2e.yaml")
        let paths = [
            "goodmai/FLUX.3-klein-MACaQ-MXFP4-MLX": painterPath,
            "black-forest-labs/FLUX.2-small-decoder": vaePath,
        ]
        let services = ComposeServices(
            resolve: { paths[$0] ?? $0 },
            load: { _, _, _, _ in encoderOrch },
            decider: { _ in StubDecisionEngine() }, decisions: { _, _, _, _, _ in nil },
            key: { _ in "" }, cloudLimits: { _ in [:] }
        )
        var start = ComposeStart(
            chat: rewriterOrch, decider: nil, decisions: nil,
            cloudGate: CloudGate(limits: [:], storage: nil), startup: PipelineStages(), besideDecode: true
        )
        start.admission = ComposeAdmission(
            sizes: ComposeSpec.Sizes(bytes: { _ in 10_000_000 }, onNeuralEngine: { _ in false })
        ) { (32_000_000_000, .desktop) }
        let host = ComposeHost(services: services, spec: spec, start: start)
        try await host.start(defaultDecisions: nil)
        return host
    }
}
