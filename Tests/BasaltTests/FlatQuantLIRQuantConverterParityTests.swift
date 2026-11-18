import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import MLXLMCommon
import MLXNN
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("FlatQuant & LIRQuant Universal Converter Write Parity, REST/CLI & macOS Sheet (#547)", .serialized)
struct FlatQuantLIRQuantConverterParityTests {
    @Test("QuantizationTarget parses, normalizes, and JSON-encodes FlatQuant and LIRQuant specifications")
    func parseAndSerializeBasisTargets() throws {
        let fq = try QuantizationTarget.parse("flatquant:4:64")
        #expect(fq == .flatQuant(bits: 4, groupSize: 64))
        #expect(fq.specString == "flatquant:4:64")
        #expect(fq.suggestedTag == "FlatQuant-4bit")
        #expect(fq.supportsPVTune && !fq.isDense)
        #expect(abs(fq.effectiveBitsPerWeight - 4.52) < 1e-6)

        let fq32 = try QuantizationTarget.parse(" flatquant:3:32 ")
        #expect(fq32 == .flatQuant(bits: 3, groupSize: 32) && fq32.specString == "flatquant:3:32")

        let lirFull = try QuantizationTarget.parse("lirquant:4:64:128")
        #expect(lirFull == .lirQuant(bits: 4, groupSize: 64, subspaceDim: 128))
        #expect(lirFull.specString == "lirquant:4:64:128")
        #expect(lirFull.suggestedTag == "LIRQuant-4bit-k128")
        #expect(lirFull.supportsPVTune && !lirFull.isDense)

        let lirShort = try QuantizationTarget.parse("lirquant:4:64")
        #expect(lirShort == lirFull)

        let encoded = try JSONEncoder().encode([fq, lirFull])
        let decoded = try JSONDecoder().decode([QuantizationTarget].self, from: encoded)
        #expect(decoded == [fq, lirFull])
        #expect(QuantizationTarget.standardCatalogSpecs.contains("flatquant:4:64"))
        #expect(QuantizationTarget.standardCatalogSpecs.contains("lirquant:4:64:128"))

        for invalid in ["flatquant:5:64", "flatquant:4:16", "lirquant:4:64:48", "lirquant:7:64:128"] {
            #expect(throws: BasaltError.self) {
                _ = try QuantizationTarget.parse(invalid)
            }
        }

        let badDir = FileManager.default.temporaryDirectory.appending(path: "bad-manifest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: badDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: badDir) }
        try Data(#"{"model_type":"qwen3","flatquant_quantization":{"version":99,"modules":{}}}"#.utf8)
            .write(to: badDir.appending(component: "config.json"))
        #expect(throws: BasaltError.self) {
            _ = try BasisTransformNativeLoader.flatQuantManifest(in: badDir)
        }
        try Data(#"{"model_type":"qwen3","lirquant_quantization":{"version":99,"modules":{}}}"#.utf8)
            .write(to: badDir.appending(component: "config.json"))
        #expect(throws: BasaltError.self) {
            _ = try BasisTransformNativeLoader.lirQuantManifest(in: badDir)
        }
    }

    @Test("UniversalModelConverter round-trips Dense -> FlatQuant -> LIRQuant -> MXFP4 -> Dense and reloads natively")
    func anyToAnyBasisTransformRoundTrip() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "fq-lir-\(UUID().uuidString)")
        let denseDir = root.appending(component: "dense-src")
        let fqDir = root.appending(component: "flatquant-4-64")
        let lirDir = root.appending(component: "lirquant-4-64-128")
        let mxfp4Dir = root.appending(component: "mxfp4-out")
        let denseRec = root.appending(component: "dense-recovered")
        try Self.writeTinyCheckpoint(to: denseDir)
        defer { try? FileManager.default.removeItem(at: root) }

        let reqFQ = ModelConversionRequest(
            source: denseDir.path, outputId: "local/fq", target: .flatQuant(bits: 4, groupSize: 64), stream: false)
        let resFQ = try await UniversalModelConverter.convertToDirectory(
            request: reqFQ, sourceURL: denseDir, outputURL: fqDir) { _ in }
        #expect(resFQ.shards == 1 && resFQ.bytesAfter > 0)
        let detFQ = try UniversalSourceReader.detect(at: fqDir)
        #expect(detFQ.format == "flatquant:4:64" && detFQ.requiresCanonicalDecode)
        let fqInspect = await ModelInspector().inspect(modelIdOrPath: fqDir.path)
        #expect(fqInspect.intact && fqInspect.quantizationMode == "flatquant")
        try await Self.assertNativeContainerForward(at: fqDir)

        let reqLIR = ModelConversionRequest(
            source: fqDir.path, outputId: "local/lir",
            target: .lirQuant(bits: 4, groupSize: 64, subspaceDim: 128), stream: false)
        let resLIR = try await UniversalModelConverter.convertToDirectory(
            request: reqLIR, sourceURL: fqDir, outputURL: lirDir) { _ in }
        #expect(resLIR.shards == 1 && resLIR.bytesAfter > 0)
        let detLIR = try UniversalSourceReader.detect(at: lirDir)
        #expect(detLIR.format == "lirquant:4:64:128" && detLIR.requiresCanonicalDecode)
        let lirInspect = await ModelInspector().inspect(modelIdOrPath: lirDir.path)
        #expect(lirInspect.intact && lirInspect.quantizationMode == "lirquant")
        try await Self.assertNativeContainerForward(at: lirDir)

        let reqMX = ModelConversionRequest(
            source: lirDir.path, outputId: "local/mxfp4", target: .mxfp4, recipe: "sinq", stream: false)
        _ = try await UniversalModelConverter.convertToDirectory(
            request: reqMX, sourceURL: lirDir, outputURL: mxfp4Dir) { _ in }
        #expect(try UniversalSourceReader.detect(at: mxfp4Dir).format == "mxfp4")

        let reqDense = ModelConversionRequest(
            source: lirDir.path, outputId: "local/dense", target: .denseBF16, stream: false)
        let resDense = try await UniversalModelConverter.convertToDirectory(
            request: reqDense, sourceURL: lirDir, outputURL: denseRec) { _ in }
        #expect(resDense.shards == 1 && resDense.bytesAfter > 0)
        #expect(try UniversalSourceReader.detect(at: denseRec).format == "dense:bf16")
    }

    @Test("CLI --experimental-flatquant, --experimental-lirquant, --to, and REST /v1/models/convert* wire cleanly")
    func cliFlagsAndRESTEndpoints() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "fq-cli-\(UUID().uuidString)")
        let denseDir = root.appending(component: "dense")
        let fqOut = root.appending(component: "fq-exp")
        let lirOut = root.appending(component: "lir-exp")
        try Self.writeTinyCheckpoint(to: denseDir)
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse([
                denseDir.path, "--experimental-flatquant", "4:64",
                "--experimental-lirquant", "4:64:128", "--out", fqOut.path,
            ])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse([
                denseDir.path, "--to", "flatquant:4:64",
                "--experimental-flatquant", "4:64", "--out", fqOut.path,
            ])
        }

        var cmdFQ = try ConvertSubcommand.parse([
            denseDir.path, "--experimental-flatquant", "4:64", "--macaq-alpha", "0.5", "--out", fqOut.path,
        ])
        try await cmdFQ.run()
        #expect(try UniversalSourceReader.detect(at: fqOut).format == "flatquant:4:64")

        var cmdLIR = try ConvertSubcommand.parse([
            fqOut.path, "--experimental-lirquant", "4:64:64", "--out", lirOut.path,
        ])
        try await cmdLIR.run()
        #expect(try UniversalSourceReader.detect(at: lirOut).format == "lirquant:4:64:64")

        try await Self.assertRESTPlanAndSSE()
    }

    @MainActor
    @Test("ConvertModelSheet renders FlatQuant and LIRQuant across Light, Dark, and Russian with 100% Localizable.strings parity")
    func sheetRenderingAndLocalizationParity() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let en = try #require(
            NSDictionary(contentsOf: repoRoot.appending(path: "Localization/en.lproj/Localizable.strings")) as? [String: String])
        let ru = try #require(
            NSDictionary(contentsOf: repoRoot.appending(path: "Localization/ru.lproj/Localizable.strings")) as? [String: String])
        #expect(Set(en.keys) == Set(ru.keys))
        for key in ["FlatQuant (Kronecker P1⊗P2)", "LIRQuant (Givens Subspace)", "Vulnerable Subspace", "128", "256"] {
            #expect(en[key]?.isEmpty == false && ru[key]?.isEmpty == false)
        }

        var base = LibraryState()
        let model = LibraryModel(id: "mlx-community/Qwen3.8-9B-4bit", sizeBytes: 5_400_000_000, isLoaded: false)
        base.models = [model]
        let env = LibraryEnvironment(client: { nil }, savePending: { _ in })

        for fam in [ConversionTargetFamily.flatQuant, .lirQuant] {
            var draft = ConversionDraft(sourceId: model.id, sourceBytes: model.sizeBytes)
            draft.family = fam
            draft.bits = 4
            draft.groupSize = 64
            draft.lirSubspaceDim = 128
            draft.refreshSuggestedOutputId()
            let req = draft.toRequest()
            if fam == .flatQuant {
                #expect(req.target == .flatQuant(bits: 4, groupSize: 64))
                #expect(draft.outputId.contains("FlatQuant-4bit"))
            } else {
                #expect(req.target == .lirQuant(bits: 4, groupSize: 64, subspaceDim: 128))
                #expect(draft.outputId.contains("LIRQuant-4bit-k128"))
            }
            base.convertDraft = draft
            let store = LibraryStore(base, reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(env))
            for variant in [TraitVariant.lightStandard, .darkStandard, .darkRussian] {
                let bmp = try OffscreenSceneRenderer.render(
                    ConvertModelSheet(store: store), spec: .init(id: .libraryMixed, width: 580, height: 630), variant: variant)
                #expect(bmp.width == 1_160 && bmp.height == 1_260 && !bmp.png.isEmpty)
            }
        }
    }
}

extension FlatQuantLIRQuantConverterParityTests {
    fileprivate static func writeTinyCheckpoint(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(.qwen3Small)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: flat, url: directory.appending(component: "model.safetensors"))
        var cfg = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen3Small.config.utf8)) as? [String: Any] ?? [:]
        cfg["model_type"] = "qwen3"
        try JSONSerialization.data(withJSONObject: cfg).write(to: directory.appending(component: "config.json"))
        try Data(#"{"tokenizer_class": "LlamaTokenizerFast", "legacy": false}"#.utf8)
            .write(to: directory.appending(component: "tokenizer_config.json"))
        try Data(#"{"model": {"type": "BPE", "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, " ": 4}, "merges": ["a b"], "unk_token": "<unk>"}, "added_tokens": []}"#.utf8)
            .write(to: directory.appending(component: "tokenizer.json"))
    }

    fileprivate static func assertNativeContainerForward(at directory: URL) async throws {
        let loaded = try #require(
            try await BasisTransformNativeLoader.loadContainer(from: directory, tokenizerLoader: ByteTokenizerLoader()))
        let finite = await loaded.perform { ctx in
            let prompt = MLXArray([Int32(1), 2, 3, 4], [1, 4])
            let logits = ctx.model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits.asType(.float32)
            eval(logits)
            return logits.asArray(Float.self).allSatisfy(\.isFinite)
        }
        #expect(finite)
    }

    fileprivate static func assertRESTPlanAndSSE() async throws {
        let orch = ModelOrchestratorActor(engine: MockInferenceEngine())
        try await orch.loadModel(path: "mock")
        let hub = ModelHubOperations(converter: UniversalConverterTests.StubConversionService()) { _, _ in
            URL(fileURLWithPath: "/tmp")
        }
        let server = RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock")).withModelHub(hub)
        let app = Application(router: server.makeRouter())

        try await app.test(.router) { client in
            for spec in ["flatquant:4:64", "lirquant:4:64:128"] {
                try await client.execute(
                    uri: "/v1/models/convert/plan?source=mlx-community/Qwen3.8-9B-4bit&target=\(spec)&pv_tune=true",
                    method: .get
                ) { res in
                    #expect(res.status == .ok)
                    let plan = try JSONDecoder().decode(ModelConversionPlan.self, from: Data(res.body.readableBytesView))
                    #expect(plan.target == spec && plan.supportsPVTune)
                    #expect(plan.supportedTargets.contains("flatquant:4:64"))
                    #expect(plan.supportedTargets.contains("lirquant:4:64:128"))
                }
            }

            let body = try JSONEncoder().encode(ModelConversionRequest(
                source: "mlx-community/Qwen3.8-9B-4bit",
                outputId: "local/Qwen3.8-9B-LIRQuant-4bit-k128",
                target: .lirQuant(bits: 4, groupSize: 64, subspaceDim: 128),
                pvTune: true,
                stream: true))
            try await client.execute(
                uri: "/v1/models/convert",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(bytes: body)
            ) { res in
                #expect(res.status == .ok)
                let sse = String(buffer: res.body)
                #expect(sse.contains("event: progress") && sse.contains("event: done"))
                #expect(sse.contains("local/Qwen3.8-9B-LIRQuant-4bit-k128"))
            }
        }
    }
}
