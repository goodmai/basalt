import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct WhisperTestTokenizer: MLXLMCommon.Tokenizer {
    var specials = ["<|endoftext|>": 50, "<|startoftranscript|>": 51, "<|en|>": 52, "<|transcribe|>": 53, "<|notimestamps|>": 54]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.unicodeScalars.map { Int($0.value % 50) } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map { "t\($0)" }.joined(separator: " ") }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key ?? "t\(id)" }
    var bosToken: String? { nil }
    var eosToken: String? { "<|endoftext|>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
}

struct WhisperReference {
    let weights: [String: MLXArray]
    let configuration: WhisperConfiguration

    init(_ model: WhisperModel) {
        weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        configuration = model.configuration
    }

    func p(_ name: String) -> MLXArray { weights[name]! }

    func linear(_ name: String, _ x: MLXArray, bias: Bool = true) -> MLXArray {
        let y = matmul(x, p(name + ".weight").transposed())
        return bias ? y + p(name + ".bias") : y
    }

    func layerNorm(_ name: String, _ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) / sqrt(variance + Float(1e-5)) * p(name + ".weight") + p(name + ".bias")
    }

    func exactGELU(_ x: MLXArray) -> MLXArray { Float(0.5) * x * (Float(1) + erf(x / Float(2).squareRoot())) }

    func splitHeads(_ x: MLXArray, _ heads: Int) -> MLXArray { x.reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3) }

    func attention(_ name: String, _ x: MLXArray, source: MLXArray, heads: Int, causal: Bool) -> MLXArray {
        let q = splitHeads(linear(name + ".query", x), heads)
        let k = splitHeads(linear(name + ".key", source, bias: false), heads)
        let v = splitHeads(linear(name + ".value", source), heads)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / Float(q.dim(-1)).squareRoot()
        if causal {
            let n = x.dim(1)
            let mask = (0 ..< n).flatMap { i in (0 ..< n).map { j in j > i ? Float(-1e9) : Float(0) } }
            scores = scores + MLXArray(mask, [n, n])
        }
        let out = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3)
        return linear(name + ".out", out.reshaped(x.dim(0), x.dim(1), -1))
    }

    func block(_ name: String, _ x: MLXArray, heads: Int, causal: Bool, features: MLXArray?) -> MLXArray {
        let normed = layerNorm(name + ".attn_ln", x)
        var h = x + attention(name + ".attn", normed, source: normed, heads: heads, causal: causal)
        if let features {
            h = h + attention(name + ".cross_attn", layerNorm(name + ".cross_attn_ln", h), source: features, heads: heads, causal: false)
        }
        return h + linear(name + ".mlp2", exactGELU(linear(name + ".mlp1", layerNorm(name + ".mlp_ln", h))))
    }

    func conv(_ name: String, _ x: MLXArray, stride: Int) -> MLXArray {
        let w = p(name + ".weight")
        let zeros = MLXArray.zeros([x.dim(0), 1, x.dim(2)])
        let padded = concatenated([zeros, x, zeros], axis: 1)
        let length = (x.dim(1) - 1) / stride + 1
        var y = p(name + ".bias")
        for tap in 0 ..< 3 {
            let rows = MLXArray((0 ..< length).map { Int32($0 * stride + tap) })
            y = y + matmul(padded.take(rows, axis: 1), w[0..., tap, 0...].transposed())
        }
        return y
    }

    func sinusoids(length: Int, channels: Int) -> MLXArray {
        let half = channels / 2
        let step = Foundation.log(10_000.0) / Double(half - 1)
        var values: [Float] = []
        for t in 0 ..< length {
            for c in 0 ..< channels {
                let angle = Double(t) * Foundation.exp(-step * Double(c % half))
                values.append(Float(c < half ? Foundation.sin(angle) : Foundation.cos(angle)))
            }
        }
        return MLXArray(values, [length, channels])
    }

    func encoder(_ mel: MLXArray) -> MLXArray {
        var x = exactGELU(conv("encoder.conv1", mel, stride: 1))
        x = exactGELU(conv("encoder.conv2", x, stride: 2))
        x = x + sinusoids(length: x.dim(1), channels: x.dim(2))
        for layer in 0 ..< configuration.audioLayers {
            x = block("encoder.blocks.\(layer)", x, heads: configuration.audioHeads, causal: false, features: nil)
        }
        return layerNorm("encoder.ln_post", x)
    }

    func logits(mel: MLXArray, tokens: MLXArray) -> MLXArray {
        let features = encoder(mel)
        let embedding = p("decoder.token_embedding.weight")
        var x = embedding[tokens] + p("decoder.positional_embedding")[0 ..< tokens.dim(1)]
        for layer in 0 ..< configuration.textLayers {
            x = block("decoder.blocks.\(layer)", x, heads: configuration.textHeads, causal: true, features: features)
        }
        return matmul(layerNorm("decoder.ln", x), embedding.transposed())
    }
}

@Suite("Whisper speech recognition: log-mel front end, encoder-decoder parity, cached greedy decoding and engine routing (#818)")
struct WhisperModelTests {
    static let tinyJSON = #"""
        {"n_mels": 8, "n_audio_ctx": 4, "n_audio_state": 16, "n_audio_head": 2, "n_audio_layer": 2, "n_vocab": 64,
         "n_text_ctx": 12, "n_text_state": 16, "n_text_head": 2, "n_text_layer": 2, "model_type": "whisper"}
        """#

    static func tinyModel() throws -> WhisperModel {
        let model = try withRandomState(MLXRandom.RandomState(seed: 818)) { try WhisperModel(configurationData: Data(tinyJSON.utf8)) }
        eval(model)
        return model
    }

    static func sine(count: Int) -> [Float] {
        (0 ..< count).map { Float(0.5 * Foundation.sin(2 * Double.pi * 440 * Double($0) / 16_000)) }
    }

    static func tinyMel() -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([1, 8, 8]) }
    }

    static func referenceLogMel(_ samples: [Float], filters: [Float], melCount: Int, frames: Int) -> [Float] {
        let (size, hop, bins, length) = (400, 160, 201, frames * 160)
        let window = (0 ..< length).map { $0 < samples.count ? Double(samples[$0]) : 0 }
        let padded = (1 ... 200).reversed().map { window[$0] } + window + (1 ... 200).map { window[length - 1 - $0] }
        let hann = (0 ..< size).map { 0.5 - 0.5 * Foundation.cos(2 * Double.pi * Double($0) / Double(size)) }
        var mel = [Double](repeating: 0, count: frames * melCount)
        for frame in 0 ..< frames {
            let power = (0 ..< bins).map { bin -> Double in
                var (real, imaginary) = (0.0, 0.0)
                for i in 0 ..< size {
                    let x = padded[frame * hop + i] * hann[i]
                    let angle = 2 * Double.pi * Double(bin * i) / Double(size)
                    real += x * Foundation.cos(angle)
                    imaginary -= x * Foundation.sin(angle)
                }
                return real * real + imaginary * imaginary
            }
            for filter in 0 ..< melCount {
                mel[frame * melCount + filter] = (0 ..< bins).reduce(0.0) { $0 + Double(filters[filter * bins + $1]) * power[$1] }
            }
        }
        let logs = mel.map { Foundation.log10(Swift.max($0, 1e-10)) }
        let top = logs.max() ?? 0
        return logs.map { Float((Swift.max($0, top - 8) + 4) / 4) }
    }

    @Test("The Slaney mel filterbank has one non-negative unit-area triangle per band, in rising order")
    func melFilterbank() {
        let filters = WhisperAudio.melFilters(count: 8)
        #expect(filters.shape == [8, 201])
        let values = filters.asArray(Float.self)
        #expect(values.allSatisfy { $0 >= 0 })
        let rows = (0 ..< 8).map { Array(values[($0 * 201) ..< ($0 * 201 + 201)]) }
        for row in rows {
            #expect(abs(row.reduce(0, +) * 40 - 1) < 0.1)
        }
        let peaks = rows.map { row in row.indices.max { row[$0] < row[$1] } ?? 0 }
        #expect(peaks == peaks.sorted() && Set(peaks).count == 8)
    }

    @Test("logMel matches a plain DFT reference on a 440 Hz sine shorter than the window: shape, padding and the max - 8 clamp")
    func logMelMatchesPlainDFT() {
        let samples = Self.sine(count: 1_000)
        let filters = WhisperAudio.melFilters(count: 8)
        let mel = WhisperAudio.logMel(samples[...], filters: filters, frames: 8)
        #expect(mel.shape == [8, 8])
        let reference = Self.referenceLogMel(samples, filters: filters.asArray(Float.self), melCount: 8, frames: 8)
        let worst = zip(mel.asArray(Float.self), reference).map { abs($0 - $1) }.max() ?? .infinity
        #expect(worst < 5e-3)
    }

    @Test("Special token ids follow openai/whisper's n_vocab rule for v3, v2 and English-only vocabularies")
    func specialTokenRule() {
        let bare = WhisperTestTokenizer(specials: [:])
        let v3 = WhisperSpecialTokens(vocabularySize: 51_866).resolved(by: bare)
        #expect(WhisperSpecialTokens.languageCodes.count == 100)
        let v3Ids: [Int] = [v3.endOfText, v3.startOfTranscript, v3.transcribe, v3.noTimestamps]
        #expect(v3Ids == [50_257, 50_258, 50_360, 50_364])
        #expect(v3.language("en", tokenizer: bare) == 50_259)
        #expect(v3.language("yue", tokenizer: bare) == 50_358)
        #expect(v3.language("en-US", tokenizer: bare) == 50_259)
        #expect(v3.prompt(language: 50_259) == [50_258, 50_259, 50_360, 50_364])
        let v2 = WhisperSpecialTokens(vocabularySize: 51_865)
        let v2Ids: [Int] = [v2.transcribe, v2.noTimestamps]
        #expect(v2Ids == [50_359, 50_363])
        #expect(v2.language("yue", tokenizer: bare) == nil)
        let english = WhisperSpecialTokens(vocabularySize: 51_864)
        let englishIds: [Int] = [english.endOfText, english.startOfTranscript, english.transcribe, english.noTimestamps]
        #expect(englishIds == [50_256, 50_257, 50_358, 50_362])
        #expect(english.language("en", tokenizer: bare) == nil)
        #expect(english.prompt(language: nil) == [50_257, 50_362])
        let tiny = WhisperSpecialTokens(vocabularySize: 64).resolved(by: WhisperTestTokenizer())
        #expect(tiny.prompt(language: nil) == [51, 54] && tiny.endOfText == 50)
    }

    @Test("The configuration decodes the mlx-community keys and rejects heads that do not divide the state")
    func configurationDecoding() throws {
        let configuration = try JSONDecoder().decode(WhisperConfiguration.self, from: Data(Self.tinyJSON.utf8))
        #expect(configuration.melFrames == 8 && configuration.windowSamples == 1_280 && configuration.textContext == 12)
        let broken = Self.tinyJSON.replacingOccurrences(of: #""n_text_head": 2"#, with: #""n_text_head": 3"#)
        #expect(throws: DecodingError.self) { try JSONDecoder().decode(WhisperConfiguration.self, from: Data(broken.utf8)) }
    }

    @Test("Parameter names follow the mlx-community checkpoint, and sanitize drops alignment_heads so a full update verifies")
    func parameterNames() throws {
        let model = try Self.tinyModel()
        let keys = Set(model.parameters().flattened().map { $0.0 })
        let expected = [
            "encoder.conv1.weight", "encoder.conv2.bias", "encoder.blocks.1.attn.query.bias", "encoder.ln_post.weight",
            "decoder.token_embedding.weight", "decoder.positional_embedding", "decoder.blocks.1.cross_attn.value.weight",
            "decoder.blocks.0.cross_attn_ln.bias", "decoder.ln.bias",
        ]
        #expect(expected.allSatisfy { keys.contains($0) })
        #expect(!keys.contains("decoder.blocks.0.attn.key.bias"))
        #expect(!keys.contains(where: { $0.hasPrefix("encoder.blocks.0.cross_attn") }))
        #expect(model.encoder.conv1.weight.shape == [16, 3, 8])
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        weights["alignment_heads"] = MLXArray([Int32(1), Int32(0)]).reshaped(1, 2)
        try model.update(parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)), verify: .all)
    }

    @Test("Encoder and decoder logits match a plain-math reference built from the same weights within 1e-4, with the same argmax")
    func forwardMatchesReference() throws {
        let model = try Self.tinyModel()
        let mel = Self.tinyMel()
        let tokens = MLXArray([Int32(51), Int32(54), Int32(7), Int32(9), Int32(3)]).reshaped(1, -1)
        let logits = model.decoder(tokens, cross: model.decoder.crossKeysValues(model.encoder(mel)), cache: nil)
        let reference = WhisperReference(model).logits(mel: mel, tokens: tokens)
        #expect(logits.shape == [1, 5, 64])
        #expect(abs(logits - reference).max().item(Float.self) < 1e-4)
        #expect(argMax(logits, axis: -1).asArray(UInt32.self) == argMax(reference, axis: -1).asArray(UInt32.self))
    }

    @Test("Decoding through the self-attention KV cache with cached cross keys and values equals the uncached forward")
    func cachedDecodingMatchesUncached() throws {
        let model = try Self.tinyModel()
        let cross = model.decoder.crossKeysValues(model.encoder(Self.tinyMel()))
        let tokens: [Int32] = [51, 54, 7, 9, 3, 12]
        let full = model.decoder(MLXArray(tokens).reshaped(1, -1), cross: cross, cache: nil)
        let cache = model.newCache(parameters: nil)
        var steps = [model.decoder(MLXArray(Array(tokens[..<2])).reshaped(1, -1), cross: cross, cache: cache)]
        for token in tokens[2...] {
            steps.append(model.decoder(MLXArray([token]).reshaped(1, 1), cross: cross, cache: cache))
        }
        #expect(abs(full - concatenated(steps, axis: 1)).max().item(Float.self) < 1e-4)
    }

    @Test("Greedy decoding is deterministic, stays inside n_text_ctx and below eot, and stops at once when eot wins")
    func greedyDecoding() throws {
        let model = try Self.tinyModel()
        let cross = model.encode(Self.sine(count: 1_000)[...])
        let first = try model.decode(cross: cross, prompt: [51, 54], endOfText: 50, temperature: 0)
        let second = try model.decode(cross: cross, prompt: [51, 54], endOfText: 50, temperature: 0)
        #expect(first == second)
        #expect(first.count <= 12 - 2 + 1 && first.allSatisfy { $0 < 50 })
        let table = model.decoder.tokenEmbedding.weight
        let row = MLXArray.ones([1, 16]) * Float(10)
        model.update(parameters: ModuleParameters.unflattened([
            "decoder.ln.weight": MLXArray.zeros([16]), "decoder.ln.bias": MLXArray.ones([16]),
            "decoder.token_embedding.weight": concatenated([table[..<50], row, table[51...]], axis: 0),
        ]))
        #expect(try model.decode(cross: cross, prompt: [51, 54], endOfText: 50, temperature: 0).isEmpty)
        #expect(try model.transcribe(audio: Self.sine(count: 1_000), tokenizer: WhisperTestTokenizer(), temperature: 0, language: "en") == "")
    }

    @Test("The engine routes /v1/audio/transcriptions through SpeechRecognitionModel: Whisper transcribes two 30 s windows like a direct call")
    func engineRoutesWhisper() async throws {
        let model = try Self.tinyModel()
        let samples = Self.sine(count: 2_000)
        let direct = try model.transcribe(audio: samples, tokenizer: WhisperTestTokenizer(), temperature: 0, language: "en")
        #expect(!direct.isEmpty)
        let container = ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-whisper"), model: model, processor: ByteInputProcessor(), tokenizer: WhisperTestTokenizer()))
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        await engine.attach(container)
        #expect(await engine.transcribesAudio)
        #expect(MLXInferenceEngine.transcribesAudio(modelType: WhisperModel.modelType))
        let result = try await engine.transcribe(TranscriptionRequest(audio: AudioClip(samples: samples), language: "en", temperature: 0))
        #expect(result.text == direct)
        #expect(result.language == "en")
    }
}
