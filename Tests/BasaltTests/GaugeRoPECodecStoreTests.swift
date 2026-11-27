import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("GaugeRoPE stored codec: calibrate once, keep gaugerope.safetensors beside the weights, compress from the first token (#647, #228, #546)", .serialized)
struct GaugeRoPECodecStoreTests {
    typealias Cache = GaugeRoPEKVCacheTests

    static func temporary() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(component: "gaugerope-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func qwen35Directory(at directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let model = try TinyModel.make(.qwen35)
        try save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: directory.appending(component: "model.safetensors"))
        let config = #"{"rope_theta": 100000.0, "# + TinyModel.Family.qwen35.config.dropFirst()
        try Data(config.utf8).write(to: directory.appending(component: "config.json"))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
    }

    static func request(_ source: URL, pvTune: Bool = false) -> ModelConversionRequest {
        ModelConversionRequest(source: source.path, outputId: "local/gaugerope", target: .gaugeRoPE, pvTune: pvTune, stream: false)
    }

    static func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && arrayEqual(a, b).item(Bool.self)
    }

    @Test("Save then load gives codecs whose encode and decode outputs equal the originals exactly, and the sidecar is not part of the weight fingerprint")
    func roundTripIsExact() throws {
        let layout = Cache.layout(carrierRank: 1, codeRank: 16)
        let codecs = try GaugeRoPECalibration.codecs(model: try TinyModel.make(.qwen35), tokens: Cache.tokens, layout: layout)
        #expect(!codecs.isEmpty)
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("weights".utf8).write(to: directory.appending(component: "model.safetensors"))
        let fingerprint = GaugeRoPECodecStore.fingerprint(modelDirectory: directory)
        try GaugeRoPECodecStore.save(codecs, layout: layout, fingerprint: fingerprint, to: directory)
        #expect(GaugeRoPECodecStore.fingerprint(modelDirectory: directory) == fingerprint)
        let loaded = try #require(try GaugeRoPECodecStore.load(from: directory, layout: layout, fingerprint: fingerprint))
        #expect(Set(loaded.keys) == Set(codecs.keys))
        let keys = MLXRandom.normal([10, 2, 16], key: MLXRandom.key(6479))
        let values = MLXRandom.normal([10, 2, 16], key: MLXRandom.key(6480))
        for (index, codec) in codecs {
            let restored = try #require(loaded[index])
            let (original, reread) = (codec.encode(keys: keys, values: values, offset: 3), restored.encode(keys: keys, values: values, offset: 3))
            #expect(Self.same(original.carrier, reread.carrier) && Self.same(original.joint, reread.joint))
            let (decoded, redecoded) = (codec.decode(original), restored.decode(original))
            #expect(Self.same(decoded.keys, redecoded.keys) && Self.same(decoded.values, redecoded.values))
        }
    }

    @Test("An absent file gives nil; another weight fingerprint, another RoPE layout or changed weights are refused, never silently recalibrated")
    func mismatchIsRefused() throws {
        let layout = Cache.layout(carrierRank: 1, codeRank: 16)
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = directory.appending(component: "model.safetensors")
        try Data("weights".utf8).write(to: weights)
        let fingerprint = GaugeRoPECodecStore.fingerprint(modelDirectory: directory)
        #expect(try GaugeRoPECodecStore.load(from: directory, layout: layout, fingerprint: fingerprint) == nil)
        let codecs = try GaugeRoPECalibration.codecs(model: try TinyModel.make(.qwen35), tokens: Cache.tokens, layout: layout)
        try GaugeRoPECodecStore.save(codecs, layout: layout, fingerprint: fingerprint, to: directory)
        #expect(try GaugeRoPECodecStore.load(from: directory, layout: layout, fingerprint: fingerprint) != nil)
        let wider = GaugeRoPELayout(rotaryDims: 8, ropeBase: 100_000, calibrationTokens: 64, carrierRank: 1, codeRank: 16)
        let otherBase = GaugeRoPELayout(rotaryDims: Cache.rotaryDims, ropeBase: 10_000, calibrationTokens: 64, carrierRank: 1, codeRank: 16)
        for other in [wider, otherBase, Cache.layout(carrierRank: 2, codeRank: 16)] {
            #expect(throws: BasaltError.self) { _ = try GaugeRoPECodecStore.load(from: directory, layout: other, fingerprint: fingerprint) }
        }
        try Data("other weights".utf8).write(to: weights)
        let changed = GaugeRoPECodecStore.fingerprint(modelDirectory: directory)
        #expect(changed != fingerprint)
        let error = #expect(throws: BasaltError.self) { _ = try GaugeRoPECodecStore.load(from: directory, layout: layout, fingerprint: changed) }
        guard case .modelConfigurationInvalid(_, let reason)? = error else {
            Issue.record("expected modelConfigurationInvalid, got \(String(describing: error))")
            return
        }
        #expect(reason.contains("fingerprint") && reason.contains("gaugerope"))
        #expect(throws: BasaltError.self) { try GaugeRoPECodecStore.save([:], layout: layout, fingerprint: fingerprint, to: directory) }
    }

    @Test("A cache with a preset codec stores codes from the first token, keeps no exact K/V, and returns what a cache calibrated online on the same tokens returns")
    func presetEncodesFromTheFirstToken() throws {
        let layout = Cache.layout(carrierRank: 1, codeRank: 16)
        let keys = MLXRandom.normal([1, 2, 64, 16], key: MLXRandom.key(6481))
        let values = MLXRandom.normal([1, 2, 64, 16], key: MLXRandom.key(6482))
        let online = GaugeRoPEKVCache(layout: layout)
        let expected = online.update(keys: keys, values: values)
        let codec = try #require(online.codec)
        let preset = GaugeRoPEKVCache(layout: layout, codec: codec)
        let got = preset.update(keys: keys, values: values)
        #expect(preset.encodedTokens == 64 && preset.state.count == 2)
        #expect(Self.same(got.0, expected.0) && Self.same(got.1, expected.1))
        let stepped = GaugeRoPEKVCache(layout: layout, codec: codec)
        _ = stepped.update(keys: keys[0..., 0..., 0 ..< 1], values: values[0..., 0..., 0 ..< 1])
        #expect(stepped.encodedTokens == 1 && stepped.offset == 1 && stepped.state.count == 2)
        let rest = stepped.update(keys: keys[0..., 0..., 1...], values: values[0..., 0..., 1...])
        #expect(stepped.encodedTokens == 64 && stepped.state.count == 2)
        #expect(Cache.relativeError(rest, expected) < 1e-4)
        let copied = try #require(stepped.copy() as? GaugeRoPEKVCache)
        #expect(copied.codec != nil && copied.encodedTokens == 64)
    }

    @Test("#228 parity: at full carrier and code rank a model decoding through preset-codec caches gives the stock cache's greedy token ids, with every full-attention layer encoded from token 0")
    func presetFullRankMatchesStock() throws {
        let model = try TinyModel.make(.qwen35)
        let full = Cache.layout(carrierRank: 2, codeRank: Cache.jointDims)
        let codecs = try GaugeRoPECalibration.codecs(model: model, tokens: Cache.tokens, layout: full)
        let stock = try model.newCache(parameters: nil)
        let preset = GaugeRoPEKVCache.replacingFullAttention(try model.newCache(parameters: nil), layout: full, codecs: codecs)
        let encoded = preset.compactMap { $0 as? GaugeRoPEKVCache }
        #expect(!encoded.isEmpty && encoded.count == codecs.count)
        #expect(try Cache.greedy(model, caches: preset) == Cache.greedy(model, caches: stock))
        #expect(encoded.allSatisfy { $0.encodedTokens > Cache.tokens.count && $0.encodedTokens == $0.offset && $0.state.count == 2 })
    }

    @Test("The byte tokenizer keeps the built-in calibration corpus inside a 256-row vocabulary, so no tiny fixture reads past its embedding table")
    func calibrationCorpusStaysInVocabulary() {
        let tokens = ByteTokenizer().encode(text: MultilingualMoECalibrationCorpus.defaultText, addSpecialTokens: false)
        #expect(tokens.allSatisfy { 0 ..< 256 ~= $0 })
    }

    @Test("A calibration text shorter than the calibration window is refused")
    func shortCalibrationTextIsRefused() throws {
        let model = try TinyModel.make(.qwen35)
        #expect(throws: BasaltError.self) {
            _ = try GaugeRoPECalibration.codecs(model: model, tokens: Array(Cache.tokens.prefix(63)), layout: Cache.layout(carrierRank: 1, codeRank: 16))
        }
    }

    @Test("gaugerope parses, prints and round-trips as a converter target, takes no PV-Tuning, and is priced at the source size")
    func targetSpec() async throws {
        #expect(try QuantizationTarget.parse("gaugerope") == .gaugeRoPE)
        #expect(try QuantizationTarget.parse(QuantizationTarget.gaugeRoPE.specString) == .gaugeRoPE)
        #expect(try JSONDecoder().decode(QuantizationTarget.self, from: JSONEncoder().encode(QuantizationTarget.gaugeRoPE)) == .gaugeRoPE)
        #expect(QuantizationTarget.gaugeRoPE.suggestedTag == "GaugeRoPE")
        #expect(!QuantizationTarget.gaugeRoPE.supportsPVTune && !QuantizationTarget.gaugeRoPE.isDense)
        #expect(QuantizationTarget.standardCatalogSpecs.contains("gaugerope"))
        #expect(throws: BasaltError.self) { _ = try QuantizationTarget.parse("gaugerope:4") }
        #expect(!UniversalModelConverter.needsAdapterPreFuse(target: .gaugeRoPE))
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type": "qwen3_5_text"}"#.utf8).write(to: directory.appending(component: "config.json"))
        try Data(count: 2 << 20).write(to: directory.appending(component: "model.safetensors"))
        let plan = try await UniversalModelConverter().plan(source: directory.path, target: .gaugeRoPE, budgetGB: nil, pvTune: false)
        #expect(plan.target == "gaugerope" && plan.sourceBytes == 2 << 20 && plan.estimatedOutputBytes == plan.sourceBytes && !plan.supportsPVTune)
    }

    @Test("Converting a tiny Qwen 3.5 directory to gaugerope copies the weights byte for byte and writes a sidecar that loads against the output's fingerprint")
    func convertKeepsWeightsAndWritesSidecar() async throws {
        let root = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let (source, output) = (root.appending(component: "source"), root.appending(component: "gaugerope"))
        try Self.qwen35Directory(at: source)
        let layout = try UniversalModelConverter.copyForGaugeRoPE(
            request: Self.request(source), detected: try UniversalSourceReader.detect(at: source), sourceURL: source, outputURL: output)
        #expect(layout == GaugeRoPELayout(rotaryDims: Cache.rotaryDims, ropeBase: 100_000))
        for name in ["model.safetensors", "config.json"] {
            #expect(try Data(contentsOf: output.appending(component: name)) == Data(contentsOf: source.appending(component: name)))
        }
        let container = try await loadModelContainer(from: output, using: ByteTokenizerLoader())
        await #expect(throws: BasaltError.self) {
            _ = try await UniversalModelConverter.writeGaugeRoPECodec(container, layout: layout, calibrationText: "too short", to: output)
        }
        let layers = try await UniversalModelConverter.writeGaugeRoPECodec(container, layout: layout, calibrationText: nil, to: output)
        let fingerprint = GaugeRoPECodecStore.fingerprint(modelDirectory: output)
        #expect(fingerprint == GaugeRoPECodecStore.fingerprint(modelDirectory: source))
        let stored = try #require(try GaugeRoPECodecStore.load(from: output, layout: layout, fingerprint: fingerprint))
        #expect(layers > 0 && stored.count == layers)
        #expect(try Data(contentsOf: output.appending(component: "model.safetensors")) == Data(contentsOf: source.appending(component: "model.safetensors")))
    }

    @Test("convert --to gaugerope refuses a Gemma 4 source, a request with PV-Tuning and a directory without safetensors, and leaves no output behind")
    func convertRefusals() async throws {
        let root = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        let (gemma, qwen, output) = (root.appending(component: "gemma4"), root.appending(component: "qwen35"), root.appending(component: "out"))
        try ModelOptConverterWiringTests.denseDirectory(.gemma4, at: gemma)
        try Self.qwen35Directory(at: qwen)
        let error = await #expect(throws: BasaltError.self) {
            _ = try await UniversalModelConverter.convertToDirectory(request: Self.request(gemma), sourceURL: gemma, outputURL: output) { _ in }
        }
        guard case .modelConfigurationInvalid(_, let reason)? = error else {
            Issue.record("expected modelConfigurationInvalid, got \(String(describing: error))")
            return
        }
        #expect(reason.contains("p-RoPE"))
        await #expect(throws: BasaltError.self) {
            _ = try await UniversalModelConverter.convertToDirectory(request: Self.request(qwen, pvTune: true), sourceURL: qwen, outputURL: output) { _ in }
        }
        try FileManager.default.removeItem(at: qwen.appending(component: "model.safetensors"))
        await #expect(throws: BasaltError.self) {
            _ = try await UniversalModelConverter.convertToDirectory(request: Self.request(qwen), sourceURL: qwen, outputURL: output) { _ in }
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }
}
