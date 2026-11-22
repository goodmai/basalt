import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import MLXNN
import SwiftUI
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Universal Any-to-Any Quantization Converter, REST SSE & macOS Liquid Glass ConvertModelSheet (#546)", .serialized)
struct UniversalConverterTests {
    @Test("QuantizationTarget parses and round-trips all supported target specifications")
    func parseAndRoundTripTargets() throws {
        for spec in QuantizationTarget.standardCatalogSpecs {
            let parsed = try QuantizationTarget.parse(spec)
            let reparsed = try QuantizationTarget.parse(parsed.specString)
            #expect(parsed == reparsed)
        }
        #expect(QuantizationTarget.denseBF16.isDense)
        #expect(!QuantizationTarget.denseBF16.supportsPVTune)
        #expect(QuantizationTarget.macaqAffine(bits: 3, groupSize: 64).supportsPVTune)
        #expect(throws: BasaltError.self) {
            _ = try QuantizationTarget.parse("unknown:9:99")
        }
    }

    @Test("UniversalModelConverter converts NF4 checkpoint to Affine+SINQ and back to Dense BF16 cleanly")
    func anyToAnyCrossFormatConversion() async throws {
        let rootDir = FileManager.default.temporaryDirectory.appending(path: "uconv-\(UUID().uuidString)")
        let denseSrc = rootDir.appending(component: "dense-src")
        let nf4Dir = rootDir.appending(component: "nf4-mid")
        let sinqAffineDir = rootDir.appending(component: "sinq-affine")
        let recoveredDenseDir = rootDir.appending(component: "recovered-bf16")
        try FileManager.default.createDirectory(at: denseSrc, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootDir) }

        let model = try TinyModel.make(.qwen3Small)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: flat, url: denseSrc.appending(component: "model.safetensors"))
        var cfg = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Small.config.utf8)) as! [String: Any]
        cfg["model_type"] = "qwen3"
        try JSONSerialization.data(withJSONObject: cfg).write(to: denseSrc.appending(component: "config.json"))

        let req1 = ModelConversionRequest(source: denseSrc.path, outputId: "local/nf4", target: .nf4(groupSize: 32), stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(request: req1, sourceURL: denseSrc, outputURL: nf4Dir) { _ in }
        let detNF4 = try UniversalSourceReader.detect(at: nf4Dir)
        #expect(detNF4.format == "nf4:32" && detNF4.requiresCanonicalDecode)

        let req2 = ModelConversionRequest(source: nf4Dir.path, outputId: "local/q3-sinq", target: .affine(bits: 3, groupSize: 32), recipe: "sinq", stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(request: req2, sourceURL: nf4Dir, outputURL: sinqAffineDir) { _ in }
        let detAffine = try UniversalSourceReader.detect(at: sinqAffineDir)
        #expect(detAffine.format == "affine:3:32")

        let req3 = ModelConversionRequest(source: sinqAffineDir.path, outputId: "local/bf16", target: .denseBF16, stream: false)
        let res3 = try await UniversalModelConverter.convertToDirectory(request: req3, sourceURL: sinqAffineDir, outputURL: recoveredDenseDir) { _ in }
        #expect(res3.shards >= 1 && res3.bytesAfter > 0)
        let detDense = try UniversalSourceReader.detect(at: recoveredDenseDir)
        #expect(detDense.format == "dense:bf16" && !detDense.requiresCanonicalDecode)
    }

    struct StubConversionService: ModelConversionService {
        func plan(source: String, target: QuantizationTarget, budgetGB: Double?, pvTune: Bool, cacheRoot: URL) async throws -> ModelConversionPlan {
            let fp = ModelConversionPlan.MemoryFootprint(
                sourceBytes: 6_800_000_000, estimatedOutputBytes: 5_100_000_000,
                peakConversionRAMBytes: 12_000_000_000, availableRAMBytes: 18_000_000_000)
            return ModelConversionPlan(
                sourceId: source, sourceFormat: "affine:4:64", target: target.specString,
                supportsPVTune: target.supportsPVTune, footprint: fp)
        }

        func convert(_ request: ModelConversionRequest, cacheRoot: URL, onProgress: @escaping @Sendable (ModelConversionProgress) -> Void) async throws -> ModelConversionResult {
            onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: "Quantizing layers", valKLBefore: 0.068, valKLCurrent: 0.052))
            return ModelConversionResult(
                outputId: request.outputId, bytesBefore: 6_800_000_000, bytesAfter: 5_100_000_000, shards: 1,
                pvTune: request.pvTune ? .init(valKLBefore: 0.068, valKLAfter: 0.039, updatedCodes: 1_240) : nil)
        }
    }

    @Test("REST GET /v1/models/convert/plan and SSE POST /v1/models/convert stream progress and completion")
    func restPlanAndSSEConvertEndpoints() async throws {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: StubConversionService()) { _, _ in URL(fileURLWithPath: "/tmp") }
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock")).withModelHub(hub)
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/models/convert/plan?source=mlx-community/gemma-4-12B-it-qat-4bit&target=macaq:affine:3:64&pv_tune=true",
                method: .get
            ) { res in
                #expect(res.status == .ok)
                let plan = try JSONDecoder().decode(ModelConversionPlan.self, from: Data(res.body.readableBytesView))
                #expect(plan.target == "macaq:affine:3:64" && plan.supportsPVTune)
            }

            let reqBody = try JSONEncoder().encode(ModelConversionRequest(
                source: "mlx-community/gemma-4-12B-it-qat-4bit",
                outputId: "local/gemma-4-12B-MACaQ-3bit-PV",
                target: .macaqAffine(bits: 3, groupSize: 64),
                pvTune: true,
                stream: true))
            try await client.execute(
                uri: "/v1/models/convert",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: reqBody)
            ) { res in
                #expect(res.status == .ok)
                let text = String(buffer: res.body)
                #expect(text.contains("event: progress"))
                #expect(text.contains("event: done"))
                #expect(text.contains("local/gemma-4-12B-MACaQ-3bit-PV"))
            }
        }
    }

    @Test("REST convert refuses a local path as source or pv_teacher with 400, so it is neither a path oracle nor a local reader (#664)")
    func restConvertRefusesLocalPaths() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        let hub = ModelHubOperations(converter: StubConversionService()) { _, _ in URL(fileURLWithPath: "/tmp") }
        let app = Application(router: RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock")).withModelHub(hub).makeRouter())
        try await app.test(.router) { client in
            for source in ["/Users/victim/.ssh", "~/.ssh", "../secrets"] {
                let status = try await client.execute(uri: "/v1/models/convert/plan?source=\(source)", method: .get) { $0.status }
                #expect(status == .badRequest, "\(source)")
            }
            var local = ModelConversionRequest(source: "/etc", outputId: "local/x", target: .macaqAffine(bits: 4, groupSize: 64))
            var teacher = ModelConversionRequest(source: "org/model", outputId: "local/x", target: .macaqAffine(bits: 4, groupSize: 64))
            teacher.pvTeacher = "~/models/teacher"
            var adapter = ModelConversionRequest(source: "org/model", outputId: "local/x", target: .macaqAffine(bits: 4, groupSize: 64))
            adapter.adapter = "/Users/victim/lora"
            local.stream = false
            for request in [local, teacher, adapter] {
                let status = try await client.execute(
                    uri: "/v1/models/convert", method: .post, headers: [.contentType: "application/json"],
                    body: ByteBuffer(bytes: try JSONEncoder().encode(request))) { $0.status }
                #expect(status == .badRequest)
            }
        }
    }

    @Test("ModelConvertReducer transitions through open, draft update, progress KL tracking, and completion")
    func reducerStateTransitions() {
        var state = LibraryState()
        let src = LibraryModel(id: "mlx-community/Qwen3.8-9B-4bit", sizeBytes: 5_400_000_000, isLoaded: false)
        let openFx = LibraryReducer.reduce(&state, .openConvert(src))
        #expect(state.convertDraft?.sourceId == src.id)
        #expect(openFx.count == 1)

        var draft = state.convertDraft!
        draft.family = .qtip
        draft.qtipRate = 2.5
        draft.refreshSuggestedOutputId()
        _ = LibraryReducer.reduce(&state, .updateConvertDraft(draft))
        #expect(state.convertDraft?.outputId.contains("QTIP-2.5bpw") == true)

        let startFx = LibraryReducer.reduce(&state, .startConversion)
        #expect(state.convertDraft == nil && startFx.count == 1)
        let outId = draft.outputId
        #expect(state.conversions[outId]?.active == true)

        _ = LibraryReducer.reduce(&state, .conversionProgressed(
            outId, .init(phase: "pv_tune", completed: 3, total: 4, message: "Step 2/3", valKLBefore: 0.071, valKLCurrent: 0.044)))
        #expect(state.conversions[outId]?.fraction == 0.75)
        #expect(state.conversions[outId]?.valKLCurrent == 0.044)

        let doneFx = LibraryReducer.reduce(&state, .conversionCompleted(
            outId, .init(outputId: outId, bytesBefore: 5_400_000_000, bytesAfter: 3_100_000_000, shards: 1)))
        #expect(state.conversions[outId] == nil)
        #expect(doneFx == [.list, .fit(outId)])
    }
}

@Suite("macOS Liquid Glass ConvertModelSheet and LibraryView visual rendering (#546)", .serialized)
struct UniversalConverterVisualTests {
    @MainActor
    @Test("ConvertModelSheet and LibraryView active conversion row render cleanly across Light, Dark, and Russian")
    func renderConvertModelSheetAndLibraryRow() throws {
        var sheetState = LibraryState()
        let models = [
            LibraryModel(id: "mlx-community/gemma-4-12B-it-qat-4bit", sizeBytes: 6_891_245_568, isLoaded: true),
            LibraryModel(id: "goodmai/Ornith-1.5-9B-MACaQ-MXFP4-MLX", sizeBytes: 5_210_000_000, isLoaded: false),
        ]
        sheetState.models = models
        sheetState.fits = [
            models[0].id: ModelFit(fits: true, level: "comfortable", requiredBytes: 8_000_000_000, availableBytes: 18_000_000_000),
            models[1].id: ModelFit(fits: true, level: "comfortable", requiredBytes: 6_200_000_000, availableBytes: 18_000_000_000),
        ]
        var draft = ConversionDraft(sourceId: models[0].id, sourceBytes: models[0].sizeBytes)
        draft.bits = 3
        draft.refreshSuggestedOutputId()
        let fp = ModelConversionPlan.MemoryFootprint(
            sourceBytes: 6_891_245_568, estimatedOutputBytes: 5_318_401_024,
            peakConversionRAMBytes: 14_200_000_000, availableRAMBytes: 18_400_000_000)
        draft.plan = ModelConversionPlan(
            sourceId: models[0].id, sourceFormat: "affine:4:64", target: "macaq:affine:3:64",
            supportsPVTune: true, footprint: fp)
        sheetState.convertDraft = draft

        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })
        let sheetStore = LibraryStore(sheetState, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
        let libStore = makeActiveLibraryStore(base: sheetState, env: env)
        let outDir = ProcessInfo.processInfo.environment["BASALT_SAVE_GUI_546"].map { URL(fileURLWithPath: $0) }
        if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }

        for variant in [TraitVariant.lightStandard, .darkStandard, .darkRussian] {
            let bmpSheet = try OffscreenSceneRenderer.render(
                ConvertModelSheet(store: sheetStore), spec: .init(id: .libraryMixed, width: 580, height: 630), variant: variant)
            #expect(bmpSheet.width == 1_160 && bmpSheet.height == 1_260 && !bmpSheet.png.isEmpty)
            let bmpLib = try OffscreenSceneRenderer.render(
                LibraryView(store: libStore), spec: .init(id: .libraryMixed, width: 720, height: 480), variant: variant)
            #expect(bmpLib.width == 1_440 && bmpLib.height == 960 && !bmpLib.png.isEmpty)
            if let outDir {
                try bmpSheet.png.write(to: outDir.appending(component: "gui_546_convert_sheet__\(variant.slug).png"), options: .atomic)
                try bmpLib.png.write(to: outDir.appending(component: "gui_546_library_converting__\(variant.slug).png"), options: .atomic)
            }
        }
    }

    @MainActor
    private func makeActiveLibraryStore(base: LibraryState, env: LibraryEnvironment) -> LibraryStore {
        var activeState = base
        activeState.convertDraft = nil
        var job = ConversionJobState(
            sourceId: base.models[0].id, outputId: "local/gemma-4-12B-MACaQ-3bit-PV", targetSpec: "macaq:affine:3:64")
        job.phase = "pv_tune"
        job.completed = 3
        job.total = 4
        job.message = "PV-Tuning group 2/3 — 1,840 codes refined"
        job.valKLBefore = 0.0681
        job.valKLCurrent = 0.0412
        activeState.conversions[job.outputId] = job
        return LibraryStore(activeState, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
    }

    @MainActor
    @Test("ConvertModelSheet renders all target families (Affine, NF4, MXFP4, QTIP, Budget, Dense)")
    func renderAllSheetFamilies() throws {
        var state = LibraryState()
        state.models = [LibraryModel(id: "Qwen/Qwen3-30B-A3B-BF16", sizeBytes: 61_000_000_000, isLoaded: false)]
        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })
        let outDir = ProcessInfo.processInfo.environment["BASALT_SAVE_GUI_569"].map { URL(fileURLWithPath: $0) }
        if let outDir { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }
        for fam in ConversionTargetFamily.allCases {
            var draft = ConversionDraft(sourceId: "Qwen/Qwen3-30B-A3B-BF16", sourceBytes: 61_000_000_000)
            draft.family = fam
            draft.pvTune = draft.resolvedTarget.supportsPVTune
            draft.refreshSuggestedOutputId()
            state.convertDraft = draft
            let store = LibraryStore(state, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
            let bmp = try OffscreenSceneRenderer.render(
                ConvertModelSheet(store: store), spec: .init(id: .libraryMixed, width: 580, height: 630), variant: .darkStandard)
            #expect(bmp.width == 1_160 && !bmp.png.isEmpty)
            if fam == .macaqMoE, let outDir {
                for variant in [TraitVariant.lightStandard, .darkStandard, .darkRussian] {
                    let vBmp = try OffscreenSceneRenderer.render(
                        ConvertModelSheet(store: store), spec: .init(id: .libraryMixed, width: 580, height: 630), variant: variant)
                    try vBmp.png.write(to: outDir.appending(component: "gui_569_convert_moe_sheet__\(variant.slug).png"), options: .atomic)
                }
            }
        }
    }
}

@Suite("Universal Converter Live HTTP/SSE Client, CLI --to & MACaQ Source Round-Trip (#546)", .serialized)
struct UniversalConverterEndToEndTests {
    @Test("LibraryClient.live streams SSE progress and plan over live HTTP, and REST supports non-streaming JSON")
    func liveClientAndJSONEndpoint() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orch.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: UniversalConverterTests.StubConversionService()) { _, _ in URL(fileURLWithPath: "/tmp") }
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock", apiKey: "conv-tok")).withModelHub(hub)

        try await Application(router: server.makeRouter()).test(.live) { live in
            let port = try #require(live.port)
            let ep = EngineEndpoint(base: try #require(URL(string: "http://localhost:\(port)")), token: "conv-tok")
            let client = LibraryClient.live(ep, session: .isolated)
            let plan = try await client.planConversion("org/src", .macaqMXFP4, 16.0, true)
            #expect(plan.target == "macaq:mxfp4" && plan.supportsPVTune)

            let seenPhases = Synchronization::Mutex<[String]>([])
            let req = ModelConversionRequest(source: "org/src", outputId: "local/out-macaq", target: .macaqMXFP4, pvTune: true)
            let done = try await client.convert(req) { p in seenPhases.withLock { $0.append(p.phase) } }
            #expect(done.outputId == "local/out-macaq" && seenPhases.withLock { $0 } == ["quantizing"])

            var syncReq = req
            syncReq.stream = false
            let body = try JSONEncoder().encode(syncReq)
            var urlReq = ep.request("v1/models/convert", method: "POST", body: body)
            urlReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let (data, resp) = try await URLSession.isolated.data(for: urlReq)
            #expect((resp as? HTTPURLResponse)?.statusCode == 200)
            let decoded = try JSONDecoder().decode(ModelConversionResult.self, from: data)
            #expect(decoded.shards == 1)
        }
    }

    @Test("CLI ConvertSubcommand --to converts Dense -> MACaQ -> MXFP4 and UniversalModelConverter.plan estimates footprint")
    func cliUnifiedConversionAndMACaQSource() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "ucli-\(UUID().uuidString)")
        let denseDir = root.appending(component: "dense")
        let macaqDir = root.appending(component: "macaq")
        let mxfp4Dir = root.appending(component: "mxfp4")
        let cacheRoot = root.appending(component: "hub")
        try FileManager.default.createDirectory(at: denseDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = try TinyModel.make(.qwen3Small)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: flat, url: denseDir.appending(component: "model.safetensors"))
        var cfg = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Small.config.utf8)) as! [String: Any]
        cfg["model_type"] = "qwen3"
        try JSONSerialization.data(withJSONObject: cfg).write(to: denseDir.appending(component: "config.json"))
        try Data(#"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false}"#.utf8)
            .write(to: denseDir.appending(component: "tokenizer_config.json"))
        try Data(#"{"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, " ": 4}, "merges": ["a b"], "unk_token": "<unk>"}, "added_tokens": []}"#.utf8)
            .write(to: denseDir.appending(component: "tokenizer.json"))

        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse([denseDir.path, "--to", "dense:bf16", "--pv-tune", "--out", macaqDir.path])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse([denseDir.path, "--to", "mixed:3:4:64", "--out", macaqDir.path])
        }

        let calFile = root.appending(component: "cal.txt")
        try Data(String(repeating: "ab a b ", count: 64).utf8).write(to: calFile)
        let cmd1 = try ConvertSubcommand.parse([
            denseDir.path, "--to", "macaq:affine:4:32", "--macaq-block", "32",
            "--pv-calibration", calFile.path, "--out", macaqDir.path,
        ])
        try await cmd1.runUnifiedConversion()
        let detMac = try UniversalSourceReader.detect(at: macaqDir)
        #expect(detMac.format.hasPrefix("macaq:") && detMac.requiresCanonicalDecode)

        for p in PVTunePreset.allCases {
            #expect(p.calibrationBatches > 0 && p.globalContinuousSteps > 0)
            _ = p.maxLayerGroups
        }
        let snap = cacheRoot.appending(path: "models--local--mac-src/snapshots/main")
        try FileManager.default.createDirectory(at: snap.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: macaqDir, to: snap)
        let svc = UniversalModelConverter()
        let plan = try await svc.plan(source: "local/mac-src", target: .mxfp4, budgetGB: nil, pvTune: false, cacheRoot: cacheRoot)
        #expect(plan.sourceFormat.hasPrefix("macaq:") && plan.estimatedOutputBytes > 0)

        let cmd2 = try ConvertSubcommand.parse([macaqDir.path, "--to", "mxfp4", "--sinq", "--out", mxfp4Dir.path])
        try await cmd2.runUnifiedConversion()
        #expect(try UniversalSourceReader.detect(at: mxfp4Dir).format == "mxfp4")
    }

    @Test("UniversalModelConverter converts MoE checkpoint Dense -> macaq:moe:2.8 -> Dense BF16 cleanly (#569)")
    func moeRouterWeightedBudgetAnyToAnyRoundTrip() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "umoe-\(UUID().uuidString)")
        let denseMoEDir = root.appending(component: "dense-moe")
        let macaqMoEDir = root.appending(component: "macaq-moe-28")
        let recoveredDir = root.appending(component: "recovered-moe-bf16")
        try FileManager.default.createDirectory(at: denseMoEDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = try TinyModel.make(.qwen3Moe)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: flat, url: denseMoEDir.appending(component: "model.safetensors"))
        let cfgObj = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Moe.config.utf8))
        try JSONSerialization.data(withJSONObject: cfgObj).write(to: denseMoEDir.appending(component: "config.json"))
        try Data(#"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false}"#.utf8)
            .write(to: denseMoEDir.appending(component: "tokenizer_config.json"))
        try Data(#"{"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, " ": 4}, "merges": ["a b"], "unk_token": "<unk>"}, "added_tokens": []}"#.utf8)
            .write(to: denseMoEDir.appending(component: "tokenizer.json"))

        var req1 = ModelConversionRequest(
            source: denseMoEDir.path, outputId: "local/qwen3-moe-28",
            target: .macaqMoEBudget(bpw: 2.8), recipe: "moe-rw", stream: false)
        req1.hadamardBlock = 32
        req1.calibrationText = String(repeating: "ab a b ", count: 160)
        let res1 = try await UniversalModelConverter.convertToDirectory(
            request: req1, sourceURL: denseMoEDir, outputURL: macaqMoEDir) { _ in }
        #expect(res1.shards >= 1 && res1.bytesAfter > 0)
        let detMoE = try UniversalSourceReader.detect(at: macaqMoEDir)
        #expect(detMoE.format.hasPrefix("macaq:moe:") && detMoE.requiresCanonicalDecode)

        let req2 = ModelConversionRequest(
            source: macaqMoEDir.path, outputId: "local/qwen3-moe-bf16",
            target: .denseBF16, stream: false)
        let res2 = try await UniversalModelConverter.convertToDirectory(
            request: req2, sourceURL: macaqMoEDir, outputURL: recoveredDir) { _ in }
        #expect(res2.shards >= 1 && res2.bytesAfter > 0)
        let detDense = try UniversalSourceReader.detect(at: recoveredDir)
        #expect(detDense.format == "dense:bf16" && !detDense.requiresCanonicalDecode)
    }
}
