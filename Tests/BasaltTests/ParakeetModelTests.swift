import Testing
import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct ParakeetReference {
    let weights: [String: MLXArray]
    let configuration: ParakeetConfiguration

    init(_ model: ParakeetModel) {
        weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        configuration = model.configuration
    }

    func p(_ name: String) -> MLXArray { weights[name]! }

    func linear(_ name: String, _ x: MLXArray, bias: Bool = false) -> MLXArray {
        let y = matmul(x, p(name + ".weight").transposed())
        return bias ? y + p(name + ".bias") : y
    }

    func layerNorm(_ name: String, _ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = (x - mean).square().mean(axis: -1, keepDims: true)
        return (x - mean) / sqrt(variance + Float(1e-5)) * p(name + ".weight") + p(name + ".bias")
    }

    func swish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    func zeroPad(_ x: MLXArray, axis: Int, by count: Int) -> MLXArray {
        var shape = x.shape
        shape[axis] = count
        let zeros = MLXArray.zeros(shape)
        return concatenated([zeros, x, zeros], axis: axis)
    }

    func conv2d(_ name: String, _ x: MLXArray, depthwise: Bool) -> MLXArray {
        let w = p(name + ".weight")
        let padded = zeroPad(zeroPad(x, axis: 1, by: 1), axis: 2, by: 1)
        let (rows, cols) = ((x.dim(1) - 1) / 2 + 1, (x.dim(2) - 1) / 2 + 1)
        var y = p(name + ".bias")
        for di in 0 ..< 3 {
            for dj in 0 ..< 3 {
                let patch = padded.take(MLXArray((0 ..< rows).map { Int32(2 * $0 + di) }), axis: 1)
                    .take(MLXArray((0 ..< cols).map { Int32(2 * $0 + dj) }), axis: 2)
                y = y + (depthwise ? patch * w[0..., di, dj, 0] : matmul(patch, w[0..., di, dj, 0...].transposed()))
            }
        }
        return y
    }

    func pointwise2d(_ name: String, _ x: MLXArray) -> MLXArray {
        matmul(x, p(name + ".weight")[0..., 0, 0, 0...].transposed()) + p(name + ".bias")
    }

    func subsample(_ mel: MLXArray) -> MLXArray {
        let prefix = "encoder.pre_encode.conv."
        var x = relu(conv2d(prefix + "0", mel.expandedDimensions(axis: -1), depthwise: false))
        for (depthwise, pointwise) in [(2, 3), (5, 6)] {
            x = relu(pointwise2d(prefix + "\(pointwise)", conv2d(prefix + "\(depthwise)", x, depthwise: true)))
        }
        return linear("encoder.pre_encode.out", x.transposed(0, 1, 3, 2).reshaped(x.dim(0), x.dim(1), -1), bias: true)
    }

    func positions(length: Int, size: Int) -> MLXArray {
        var values: [Float] = []
        for k in 0 ..< (2 * length - 1) {
            let position = Double(length - 1 - k)
            for c in 0 ..< size {
                let angle = position * Foundation.exp(-Double(c - c % 2) * Foundation.log(10_000.0) / Double(size))
                values.append(Float(c % 2 == 0 ? Foundation.sin(angle) : Foundation.cos(angle)))
            }
        }
        return MLXArray(values, [2 * length - 1, size])
    }

    func attention(_ name: String, _ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, t, heads) = (x.dim(0), x.dim(1), configuration.encoder.heads)
        let width = x.dim(2) / heads
        let q = linear(name + ".linear_q", x).reshaped(b, t, heads, width)
        let k = linear(name + ".linear_k", x).reshaped(b, t, heads, width).transposed(0, 2, 1, 3)
        let v = linear(name + ".linear_v", x).reshaped(b, t, heads, width).transposed(0, 2, 1, 3)
        let pos = linear(name + ".linear_pos", positions).reshaped(-1, heads, width).transposed(1, 2, 0)
        let content = matmul((q + p(name + ".pos_bias_u")).transposed(0, 2, 1, 3), k.transposed(0, 1, 3, 2))
        let raw = matmul((q + p(name + ".pos_bias_v")).transposed(0, 2, 1, 3), pos)
        let relative = stacked((0 ..< t).map { i in raw[0..., 0..., i, (t - 1 - i) ..< (2 * t - 1 - i)] }, axis: 2)
        let weights = softmax((content + relative) / Float(width).squareRoot(), axis: -1)
        return linear(name + ".linear_out", matmul(weights, v).transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }

    func convolution(_ name: String, _ x: MLXArray) -> MLXArray {
        let expanded = matmul(x, p(name + ".pointwise_conv1.weight")[0..., 0, 0...].transposed())
        let half = expanded.dim(-1) / 2
        let gated = expanded[.ellipsis, ..<half] * sigmoid(expanded[.ellipsis, half...])
        let kernel = p(name + ".depthwise_conv.weight")
        let padded = zeroPad(gated, axis: 1, by: (kernel.dim(1) - 1) / 2)
        var y = MLXArray.zeros(like: gated)
        for tap in 0 ..< kernel.dim(1) {
            y = y + padded[0..., tap ..< (tap + gated.dim(1)), 0...] * kernel[0..., tap, 0]
        }
        let norm = name + ".batch_norm."
        let normed = (y - p(norm + "running_mean")) / sqrt(p(norm + "running_var") + Float(1e-5)) * p(norm + "weight") + p(norm + "bias")
        return matmul(swish(normed), p(name + ".pointwise_conv2.weight")[0..., 0, 0...].transposed())
    }

    func feedForward(_ name: String, _ x: MLXArray) -> MLXArray {
        linear(name + ".linear2", swish(linear(name + ".linear1", x)))
    }

    func block(_ name: String, _ x: MLXArray, positions: MLXArray) -> MLXArray {
        var h = x + Float(0.5) * feedForward(name + ".feed_forward1", layerNorm(name + ".norm_feed_forward1", x))
        h = h + attention(name + ".self_attn", layerNorm(name + ".norm_self_att", h), positions: positions)
        h = h + convolution(name + ".conv", layerNorm(name + ".norm_conv", h))
        h = h + Float(0.5) * feedForward(name + ".feed_forward2", layerNorm(name + ".norm_feed_forward2", h))
        return layerNorm(name + ".norm_out", h)
    }

    func encoder(_ mel: MLXArray) -> MLXArray {
        var x = subsample(mel)
        let table = positions(length: x.dim(1), size: x.dim(2))
        for layer in 0 ..< configuration.encoder.layers {
            x = block("encoder.layers.\(layer)", x, positions: table)
        }
        return x
    }
}

@Suite("Parakeet TDT speech recognition: NeMo mel front end, FastConformer parity, greedy TDT decoding, detokenization and engine routing (#818)")
struct ParakeetModelTests {
    static let tinyJSON = #"""
        {"target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel", "model_defaults": {"tdt_durations": [0, 1, 2]},
         "preprocessor": {"sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025, "window_stride": 0.01, "window": "hann",
                          "features": 16, "n_fft": 512, "log": true, "frame_splicing": 1, "dither": 0.00001, "pad_to": 0},
         "encoder": {"feat_in": 16, "n_layers": 2, "d_model": 8, "n_heads": 2, "ff_expansion_factor": 2, "subsampling": "dw_striding",
                     "subsampling_factor": 8, "subsampling_conv_channels": 4, "self_attention_model": "rel_pos", "conv_kernel_size": 3,
                     "conv_norm_type": "batch_norm", "use_bias": false, "xscaling": false, "causal_downsampling": false},
         "decoder": {"blank_as_pad": true, "vocab_size": 4, "prednet": {"pred_hidden": 6, "pred_rnn_layers": 2}},
         "joint": {"num_classes": 4, "num_extra_outputs": 3, "vocabulary": ["▁he", "llo", "▁wor", "ld"],
                   "jointnet": {"joint_hidden": 8, "activation": "relu", "encoder_hidden": 8, "pred_hidden": 6}},
         "decoding": {"model_type": "tdt", "durations": [0, 1, 2], "greedy": {"max_symbols": 3}}}
        """#

    static func tinyModel() throws -> ParakeetModel {
        let model = try ParakeetModel(configurationData: Data(tinyJSON.utf8))
        let randomized = withRandomState(MLXRandom.RandomState(seed: 818)) {
            model.parameters().flattened().map { name, value in
                (name, name.hasSuffix("running_var") ? MLXRandom.uniform(Float(0.5) ..< Float(1.5), value.shape) : MLXRandom.normal(value.shape) * Float(0.3))
            }
        }
        model.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: randomized)))
        eval(model)
        return model
    }

    static func noise(count: Int) -> [Float] {
        var state: UInt32 = 818
        return (0 ..< count).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24) - 0.5
        }
    }

    static func identity(_ size: Int) -> MLXArray {
        MLXArray((0 ..< size * size).map { $0 / size == $0 % size ? Float(1) : 0 }, [size, size])
    }

    static func oneHots(_ pairs: [(Int, Int)], width: Int) -> MLXArray {
        MLXArray(pairs.flatMap { pair in (0 ..< width).map { $0 == pair.0 || $0 == pair.1 ? Float(1) : 0 } }, [pairs.count, width])
    }

    static func jointPassesThrough(_ model: ParakeetModel, encoderWeight: MLXArray, encoderBias: MLXArray) {
        model.update(parameters: ModuleParameters.unflattened([
            "joint.enc.weight": encoderWeight, "joint.enc.bias": encoderBias,
            "joint.pred.weight": MLXArray.zeros([8, 6]), "joint.pred.bias": MLXArray.zeros([8]),
            "joint.joint_net.2.weight": identity(8), "joint.joint_net.2.bias": MLXArray.zeros([8]),
        ]))
    }

    static func referencePower(_ padded: [Double], window: [Double], start: Int) -> [Double] {
        let size = window.count
        return (0 ... size / 2).map { bin in
            var (real, imaginary) = (0.0, 0.0)
            for i in 0 ..< size {
                let x = padded[start + i] * window[i]
                let angle = 2 * Double.pi * Double(bin * i) / Double(size)
                real += x * Foundation.cos(angle)
                imaginary -= x * Foundation.sin(angle)
            }
            return real * real + imaginary * imaginary
        }
    }

    static func referenceLogMel(_ samples: [Float], filters: [Float], features: Int) -> [Float] {
        let (size, length, hop, bins) = (512, 400, 160, 257)
        let x = samples.map(Double.init)
        let emphasized = x.indices.map { $0 == 0 ? x[0] : x[$0] - Double(Float(0.97)) * x[$0 - 1] }
        let padded = [Double](repeating: 0, count: size / 2) + emphasized + [Double](repeating: 0, count: size / 2)
        let window = (0 ..< size).map { n -> Double in
            let m = n - (size - length) / 2
            return (0 ..< length).contains(m) ? 0.5 - 0.5 * Foundation.cos(2 * Double.pi * Double(m) / Double(length - 1)) : 0
        }
        let logs = (0 ..< samples.count / hop + 1).map { frame in
            let power = referencePower(padded, window: window, start: frame * hop)
            return (0 ..< features).map { filter in
                Foundation.log((0 ..< bins).reduce(0.0) { $0 + Double(filters[filter * bins + $1]) * power[$1] } + 1 / Double(1 << 24))
            }
        }
        let normalized = (0 ..< features).map { filter -> [Double] in
            let column = logs.map { $0[filter] }
            let mean = column.reduce(0, +) / Double(column.count)
            let deviation = (column.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(column.count - 1)).squareRoot() + 1e-5
            return column.map { ($0 - mean) / deviation }
        }
        return logs.indices.flatMap { frame in normalized.map { Float($0[frame]) } }
    }

    @Test("The configuration decodes the NeMo keys, is detected by its target and TDT durations, and rejects a duration head of the wrong size")
    func configurationAndDetection() throws {
        let configuration = try ParakeetConfiguration.decode(Data(Self.tinyJSON.utf8))
        #expect(configuration.preprocessor.windowLength == 400 && configuration.preprocessor.hopLength == 160)
        #expect(configuration.blank == 4 && configuration.maxSymbols == 3 && configuration.encoder.subsamplingStages == 3)
        let dictionary = try #require(try JSONSerialization.jsonObject(with: Data(Self.tinyJSON.utf8)) as? [String: Any])
        #expect(ParakeetConfiguration.describes(dictionary))
        #expect(MLXInferenceEngine.speechModelType(config: dictionary) == ParakeetModel.modelType)
        #expect(MLXInferenceEngine.transcribesAudio(modelType: MLXInferenceEngine.speechModelType(config: dictionary)))
        #expect(!ParakeetConfiguration.describes(dictionary.merging(["model_type": "whisper"]) { $1 }))
        #expect(MLXInferenceEngine.speechModelType(config: ["model_type": "whisper"]) == "whisper")
        let broken = Self.tinyJSON.replacingOccurrences(of: #""num_extra_outputs": 3"#, with: #""num_extra_outputs": 5"#)
        #expect(throws: DecodingError.self) { try ParakeetConfiguration.decode(Data(broken.utf8)) }
    }

    @Test("Parameter names and shapes follow the parakeet-mlx checkpoint (28 tensors per layer, ReLU gaps in pre_encode.conv and joint_net)")
    func parameterNames() throws {
        let model = try Self.tinyModel()
        let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        #expect(shapes.count == 2 * 28 + 12 + 7 + 6)
        #expect(shapes.keys.filter { $0.hasPrefix("encoder.layers.0.") }.count == 28)
        let expected: [String: [Int]] = [
            "encoder.pre_encode.conv.0.weight": [4, 3, 3, 1], "encoder.pre_encode.conv.2.weight": [4, 3, 3, 1],
            "encoder.pre_encode.conv.3.weight": [4, 1, 1, 4], "encoder.pre_encode.conv.6.bias": [4], "encoder.pre_encode.out.weight": [8, 8],
            "encoder.layers.1.self_attn.pos_bias_u": [2, 4], "encoder.layers.1.self_attn.linear_pos.weight": [8, 8],
            "encoder.layers.0.conv.pointwise_conv1.weight": [16, 1, 8], "encoder.layers.0.conv.depthwise_conv.weight": [8, 3, 1],
            "encoder.layers.0.conv.batch_norm.running_var": [8], "encoder.layers.0.feed_forward2.linear1.weight": [16, 8],
            "decoder.prediction.embed.weight": [5, 6], "decoder.prediction.dec_rnn.lstm.1.Wx": [24, 6],
            "decoder.prediction.dec_rnn.lstm.1.Wh": [24, 6], "decoder.prediction.dec_rnn.lstm.0.bias": [24],
            "joint.enc.weight": [8, 8], "joint.pred.bias": [8], "joint.joint_net.2.weight": [8, 8],
        ]
        for (name, shape) in expected {
            #expect(shapes[name] == shape, "\(name)")
        }
        #expect(shapes["encoder.layers.0.self_attn.linear_q.bias"] == nil)
    }

    @Test("logMel matches a plain DFT reference: preemphasis, zero centre padding, symmetric Hann centred in n_fft, log guard and per-feature normalization")
    func logMelMatchesPlainDFT() throws {
        let configuration = try ParakeetConfiguration.decode(Data(Self.tinyJSON.utf8))
        let samples = Self.noise(count: 1_600)
        let filters = WhisperAudio.melFilters(count: 16, fftSize: 512)
        #expect(filters.shape == [16, 257])
        let mel = ParakeetAudio.logMel(samples[...], preprocessor: configuration.preprocessor, filters: filters)
        #expect(mel.shape == [11, 16])
        let reference = Self.referenceLogMel(samples, filters: filters.asArray(Float.self), features: 16)
        let worst = zip(mel.asArray(Float.self), reference).map { abs($0 - $1) }.max() ?? .infinity
        #expect(worst < 1e-3)
    }

    @Test("The FastConformer encoder matches a plain-math reference (strided depthwise subsampling, rel-pos attention with pos_bias_u/v, GLU conv module with batch norm, macaron halves) within 1e-4")
    func encoderMatchesReference() throws {
        let model = try Self.tinyModel()
        let mel = withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([1, 40, 16]) }
        let encoded = model.encoder(mel)
        let reference = ParakeetReference(model).encoder(mel)
        #expect(encoded.shape == [1, 5, 8])
        #expect(abs(encoded - reference).max().item(Float.self) < 1e-4)
    }

    @Test("Greedy TDT decoding emits until blank, advances by the predicted duration, forces a step on blank with duration 0 and caps symbols per frame")
    func greedyTDTDecoding() throws {
        let model = try Self.tinyModel()
        Self.jointPassesThrough(model, encoderWeight: Self.identity(8), encoderBias: MLXArray.zeros([8]))
        let frames = Self.oneHots([(1, 6), (4, 7), (2, 6), (3, 5), (4, 5), (0, 7)], width: 8)
        #expect(try model.greedyTokens(frames) == [1, 3, 3, 3, 0])
        let skip = ParakeetModel.advance(duration: 2, blank: false, symbols: 3, maxSymbols: 3)
        let stay = ParakeetModel.advance(duration: 0, blank: false, symbols: 2, maxSymbols: 3)
        #expect(skip.frames == 2 && skip.symbols == 0 && stay.frames == 0 && stay.symbols == 2)
    }

    @Test("Detokenization joins SentencePiece pieces with ▁ as a space, skips special pieces on request, and encode round-trips")
    func detokenization() {
        let tokenizer = ParakeetTokenizer(vocabulary: ["<unk>", "▁he", "llo", "▁wor", "ld", "<|endoftext|>"])
        #expect(tokenizer.decode(tokenIds: [1, 2, 3, 4], skipSpecialTokens: true) == "hello world")
        #expect(tokenizer.decode(tokenIds: [1, 2, 5, 3, 4, 9], skipSpecialTokens: true) == "hello world")
        #expect(tokenizer.decode(tokenIds: [1, 2, 5], skipSpecialTokens: false) == "hello<|endoftext|>")
        #expect(tokenizer.encode(text: "hello world", addSpecialTokens: false) == [1, 2, 3, 4])
        #expect(tokenizer.encode(text: "hex", addSpecialTokens: false) == [1, 0])
    }

    @Test("The engine routes /v1/audio/transcriptions through SpeechRecognitionModel and the loader reads a config.json plus model.safetensors with Parakeet's names")
    func engineRoutesParakeet() async throws {
        let model = try Self.tinyModel()
        Self.jointPassesThrough(model, encoderWeight: MLXArray.zeros([8, 8]), encoderBias: Self.oneHots([(1, 6)], width: 8)[0])
        let samples = Self.noise(count: 3_200)
        let direct = try model.transcribe(audio: samples, tokenizer: model.tokenizer, temperature: 0, language: nil)
        #expect(direct == "llollollo")
        let directory = FileManager.default.temporaryDirectory.appending(path: "parakeet-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(Self.tinyJSON.utf8).write(to: directory.appending(component: "config.json"))
        try save(arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: directory.appending(component: "model.safetensors"))
        let loaded = try await ParakeetLoader.loadContainer(from: directory)
        let container = try #require(loaded)
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 0))
        await engine.attach(container)
        #expect(await engine.transcribesAudio)
        let result = try await engine.transcribe(TranscriptionRequest(audio: AudioClip(samples: samples), language: nil, temperature: 0))
        #expect(result.text == direct)
        let missing = try await ParakeetLoader.loadContainer(from: directory.appending(component: "missing"))
        #expect(missing == nil)
    }
}
