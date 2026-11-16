import Foundation
import MLX
import MLXNN
import BasaltCore

public struct CosyVoice3Config: Sendable, Codable, Equatable {
    public var modelID: String = "Fun-CosyVoice3-0.5B-2512"
    public var textVocabSize: Int = 4_096
    public var fsqSpeechTokens: Int = 6_561
    public var talkerHiddenSize: Int = 512
    public var ditHiddenSize: Int = 256
    public var ditHeads: Int = 4
    public var melBins: Int = 80
    public var speakerDim: Int = 192
    public var preLookaheadLen: Int = 3
    public var tokenRateHz: Int = 25
    public var melRateHz: Int = 50
    public var cfgStrength: Float = 0.7
    public var odeSteps: Int = 4
    public var upsampleStride: Int = 120
    public var nFFT: Int = 16
    public var hopLength: Int = 4
    public var sampleRate: Int = SpeechRequest.defaultSampleRate

    public var sosToken: Int { fsqSpeechTokens }
    public var eosToken: Int { fsqSpeechTokens + 1 }

    public static var tiny: CosyVoice3Config {
        var cfg = CosyVoice3Config(
            modelID: "Fun-CosyVoice3-0.5B-Tiny",
            textVocabSize: 512,
            talkerHiddenSize: 32,
            ditHiddenSize: 32
        )
        cfg.odeSteps = 2
        cfg.upsampleStride = 30
        cfg.nFFT = 16
        cfg.hopLength = 4
        return cfg
    }

    public static func make() -> CosyVoice3Config { .tiny }

    public init(
        modelID: String = "Fun-CosyVoice3-0.5B-2512",
        textVocabSize: Int = 4_096,
        talkerHiddenSize: Int = 512,
        ditHiddenSize: Int = 256,
        sampleRate: Int = SpeechRequest.defaultSampleRate
    ) {
        self.modelID = modelID
        self.textVocabSize = max(64, textVocabSize)
        self.talkerHiddenSize = max(16, talkerHiddenSize)
        self.ditHiddenSize = max(16, ditHiddenSize)
        self.sampleRate = max(8_000, sampleRate)
    }

    public static func from(directory: URL, fallbackID: String = "Fun-CosyVoice3-0.5B-2512") throws(BasaltError) -> CosyVoice3Config {
        _ = try TTSModelFamily.validateSupported(identifier: fallbackID, directory: directory)
        let url = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: url),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return CosyVoice3Config(modelID: fallbackID)
        }
        _ = try TTSModelFamily.validateSupported(identifier: fallbackID, config: raw, directory: directory)
        var cfg = CosyVoice3Config(
            modelID: raw["model_id"] as? String ?? fallbackID,
            textVocabSize: raw["text_vocab_size"] as? Int ?? raw["vocab_size"] as? Int ?? 4_096,
            talkerHiddenSize: raw["talker_hidden_size"] as? Int ?? raw["hidden_size"] as? Int ?? 512,
            ditHiddenSize: raw["dit_hidden_size"] as? Int ?? 256,
            sampleRate: raw["sample_rate"] as? Int ?? SpeechRequest.defaultSampleRate
        )
        cfg.fsqSpeechTokens = max(64, raw["fsq_speech_tokens"] as? Int ?? 6_561)
        cfg.melBins = max(16, raw["mel_bins"] as? Int ?? 80)
        cfg.speakerDim = max(16, raw["speaker_dim"] as? Int ?? 192)
        cfg.preLookaheadLen = max(1, raw["pre_lookahead_len"] as? Int ?? 3)
        cfg.odeSteps = max(1, raw["ode_steps"] as? Int ?? 4)
        cfg.cfgStrength = Float(raw["cfg_strength"] as? Double ?? 0.7)
        return cfg
    }
}

public typealias TinyCosyVoice3Config = CosyVoice3Config

final class CosyVoice3TalkerModule: Module {
    let config: CosyVoice3Config
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "speech_embedding") var speechEmbedding: Embedding
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "llm_decoder") var llmDecoder: Linear

    init(_ config: CosyVoice3Config) {
        self.config = config
        let d = config.talkerHiddenSize
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.textVocabSize, dimensions: d)
        self._speechEmbedding.wrappedValue = Embedding(embeddingCount: config.fsqSpeechTokens + 3, dimensions: d)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: d)
        self._qkvProj.wrappedValue = Linear(d, d * 3, bias: false)
        self._outProj.wrappedValue = Linear(d, d, bias: false)
        self._mlpNorm.wrappedValue = RMSNorm(dimensions: d)
        self._gateProj.wrappedValue = Linear(d, d * 2, bias: false)
        self._upProj.wrappedValue = Linear(d, d * 2, bias: false)
        self._downProj.wrappedValue = Linear(d * 2, d, bias: false)
        self._llmDecoder.wrappedValue = Linear(d, config.fsqSpeechTokens + 2)
        super.init()
    }

    func generateFSQTokens(promptTokens: [Int32], targetSteps: Int) -> [Int32] {
        let d = config.talkerHiddenSize
        let promptH = textEmbedding(MLXArray(promptTokens).reshaped(1, promptTokens.count))
        let sosH = speechEmbedding(MLXArray([Int32(config.sosToken)]).reshaped(1, 1))
        var seq = concatenated([promptH, sosH], axis: 1)
        var emitted: [Int32] = []
        emitted.reserveCapacity(targetSteps)
        for step in 0 ..< targetSteps {
            let normed = attnNorm(seq)
            let qkv = qkvProj(normed)
            let q = qkv[0..., (seq.dim(1) - 1) ..< seq.dim(1), ..<d]
            let k = qkv[0..., 0..., d ..< (2 * d)]
            let v = qkv[0..., 0..., (2 * d)...]
            let scale = 1.0 / Float(d).squareRoot()
            let scores = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)
            let attended = seq[0..., (seq.dim(1) - 1) ..< seq.dim(1), 0...] + outProj(matmul(scores, v))
            let mNorm = mlpNorm(attended)
            let hidden = attended + downProj(MLXNN.silu(gateProj(mNorm)) * upProj(mNorm))
            let logits = llmDecoder(hidden).reshaped(-1)[..<config.fsqSpeechTokens].asArray(Float.self)
            let nextToken = Self.selectRepetitionAware(logits: logits, recent: emitted.suffix(10), step: step)
            emitted.append(nextToken)
            let nextEmb = speechEmbedding(MLXArray([nextToken]).reshaped(1, 1))
            seq = concatenated([seq, nextEmb], axis: 1)
        }
        return emitted
    }

    private static func selectRepetitionAware(logits: [Float], recent: ArraySlice<Int32>, step: Int) -> Int32 {
        guard !logits.isEmpty else { return 0 }
        let recentSet = Set(recent)
        var bestIdx = step % logits.count
        var bestVal = -Float.greatestFiniteMagnitude
        for (idx, raw) in logits.enumerated() {
            let penalty: Float = recentSet.contains(Int32(idx)) ? 1.5 : 0.0
            let adjusted = raw - penalty + Float((idx + step * 7) % 13) * 1e-3
            if adjusted > bestVal {
                bestVal = adjusted
                bestIdx = idx
            }
        }
        return Int32(bestIdx)
    }
}

final class CosyVoice3CausalDiTModule: Module {
    let config: CosyVoice3Config
    @ModuleInfo(key: "fsq_emb") var fsqEmbedding: Embedding
    @ModuleInfo(key: "pre_lookahead_1") var preLookahead1: Conv1d
    @ModuleInfo(key: "pre_lookahead_2") var preLookahead2: Conv1d
    @ModuleInfo(key: "mel_in") var melIn: Linear
    @ModuleInfo(key: "spk_cond") var spkCond: Linear
    @ModuleInfo(key: "time_cond") var timeCond: Linear
    @ModuleInfo(key: "adaln") var adaLN: Linear
    @ModuleInfo(key: "dit_norm") var ditNorm: LayerNorm
    @ModuleInfo(key: "dit_qkv") var ditQKV: Linear
    @ModuleInfo(key: "dit_out") var ditOut: Linear
    @ModuleInfo(key: "dit_mlp_1") var ditMLP1: Linear
    @ModuleInfo(key: "dit_mlp_2") var ditMLP2: Linear
    @ModuleInfo(key: "vel_out") var velOut: Linear

    init(_ config: CosyVoice3Config) {
        self.config = config
        let d = config.ditHiddenSize
        let kSize = config.preLookaheadLen + 1
        self._fsqEmbedding.wrappedValue = Embedding(embeddingCount: config.fsqSpeechTokens + 3, dimensions: d)
        self._preLookahead1.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: kSize)
        self._preLookahead2.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 3, padding: 1)
        self._melIn.wrappedValue = Linear(config.melBins, d)
        self._spkCond.wrappedValue = Linear(config.speakerDim, d)
        self._timeCond.wrappedValue = Linear(16, d)
        self._adaLN.wrappedValue = Linear(d, d * 6)
        self._ditNorm.wrappedValue = LayerNorm(dimensions: d, affine: false)
        self._ditQKV.wrappedValue = Linear(d, d * 3, bias: false)
        self._ditOut.wrappedValue = Linear(d, d)
        self._ditMLP1.wrappedValue = Linear(d, d * 2)
        self._ditMLP2.wrappedValue = Linear(d * 2, d)
        self._velOut.wrappedValue = Linear(d, config.melBins)
        super.init()
    }

    func solveCausalFlowMel(fsqTokens: [Int32], speakerVec: [Float]) -> MLXArray {
        let tokenH = fsqEmbedding(MLXArray(fsqTokens).reshaped(1, fsqTokens.count))
        let paddedLookahead = padded(
            tokenH,
            widths: [IntOrPair((0, 0)), IntOrPair((0, config.preLookaheadLen)), IntOrPair((0, 0))]
        )
        let looked = tokenH + MLXNN.silu(preLookahead2(MLXNN.silu(preLookahead1(paddedLookahead))))
        let frames50Hz = looked.dim(1) * 2
        let muCond = stacked([looked, looked], axis: 2).reshaped(1, frames50Hz, config.ditHiddenSize)
        let spkH = spkCond(MLXArray(speakerVec).reshaped(1, 1, config.speakerDim))
        let zeroSpk = MLXArray.zeros([1, 1, config.ditHiddenSize], dtype: .float32)
        let zeroMu = MLXArray.zeros(muCond.shape, dtype: .float32)
        var x = tanh(velOut(muCond)) * 0.25
        let steps = max(1, config.odeSteps)
        for step in 0 ..< steps {
            let s0 = Double(step) / Double(steps)
            let s1 = Double(step + 1) / Double(steps)
            let t0 = Float(1.0 - cos(0.5 * .pi * s0))
            let t1 = Float(1.0 - cos(0.5 * .pi * s1))
            let dt = t1 - t0
            let vCond = predictVelocity(x: x, mu: muCond, spk: spkH, t: t0)
            let vUncond = predictVelocity(x: x, mu: zeroMu, spk: zeroSpk, t: t0)
            let guided = (1.0 + config.cfgStrength) * vCond - config.cfgStrength * vUncond
            x = x + dt * guided
        }
        return x
    }

    private func predictVelocity(x: MLXArray, mu: MLXArray, spk: MLXArray, t: Float) -> MLXArray {
        let d = config.ditHiddenSize
        let tEmbed = Self.sinusoidalTime(t: t)
        let cond = MLXNN.silu(timeCond(tEmbed) + spk)
        let gates = adaLN(cond)
        let shiftMSA = gates[0..., 0..., ..<d]
        let scaleMSA = gates[0..., 0..., d ..< (2 * d)]
        let gateMSA = tanh(gates[0..., 0..., (2 * d) ..< (3 * d)])
        let shiftMLP = gates[0..., 0..., (3 * d) ..< (4 * d)]
        let scaleMLP = gates[0..., 0..., (4 * d) ..< (5 * d)]
        let gateMLP = tanh(gates[0..., 0..., (5 * d)...])
        let hIn = melIn(x) + mu
        let mod1 = ditNorm(hIn) * (1.0 + scaleMSA) + shiftMSA
        let attended = causalAttention(mod1)
        let hMid = hIn + (0.5 + 0.5 * gateMSA) * ditOut(attended)
        let mod2 = ditNorm(hMid) * (1.0 + scaleMLP) + shiftMLP
        let hOut = hMid + (0.5 + 0.5 * gateMLP) * ditMLP2(MLXNN.silu(ditMLP1(mod2)))
        return velOut(hOut)
    }

    private func causalAttention(_ x: MLXArray) -> MLXArray {
        let d = config.ditHiddenSize
        let len = x.dim(1)
        let qkv = ditQKV(x)
        let q = qkv[0..., 0..., ..<d]
        let k = qkv[0..., 0..., d ..< (2 * d)]
        let v = qkv[0..., 0..., (2 * d)...]
        let scale = 1.0 / Float(d).squareRoot()
        let rawScores = matmul(q, k.transposed(0, 2, 1)) * scale
        let idx = MLXArray(Int32(0) ..< Int32(len))
        let mask = idx.expandedDimensions(axis: 1) .>= idx.expandedDimensions(axis: 0)
        let masked = which(mask.expandedDimensions(axis: 0), rawScores, MLXArray(Float(-1e9)))
        return matmul(softmax(masked, axis: -1), v)
    }

    private static func sinusoidalTime(t: Float) -> MLXArray {
        let vals = (0 ..< 8).flatMap { i -> [Float] in
            let freq = Float(pow(2.0, Double(i))) * Float.pi
            return [sin(t * freq), cos(t * freq)]
        }
        return MLXArray(vals).reshaped(1, 1, 16)
    }
}

final class CosyVoice3CausalHiFTModule: Module {
    let config: CosyVoice3Config
    @ModuleInfo(key: "f0_predictor") var f0Predictor: Conv1d
    @ModuleInfo(key: "mel_upsample") var melUpsample: ConvTransposed1d
    @ModuleInfo(key: "nsf_proj") var nsfProj: Linear
    @ModuleInfo(key: "res_conv_1") var resConv1: Conv1d
    @ModuleInfo(key: "res_conv_2") var resConv2: Conv1d
    @ModuleInfo(key: "stft_proj") var stftProj: Linear

    init(_ config: CosyVoice3Config) {
        self.config = config
        let ch = max(16, config.melBins / 2)
        self._f0Predictor.wrappedValue = Conv1d(inputChannels: config.melBins, outputChannels: 1, kernelSize: 3, padding: 1)
        self._melUpsample.wrappedValue = ConvTransposed1d(
            inputChannels: config.melBins,
            outputChannels: ch,
            kernelSize: config.upsampleStride,
            stride: config.upsampleStride
        )
        self._nsfProj.wrappedValue = Linear(3, ch)
        self._resConv1.wrappedValue = Conv1d(inputChannels: ch, outputChannels: ch, kernelSize: 3)
        self._resConv2.wrappedValue = Conv1d(inputChannels: ch, outputChannels: ch, kernelSize: 3)
        self._stftProj.wrappedValue = Linear(ch, (config.nFFT / 2 + 1) * 2)
        super.init()
    }

    func synthesizeHiFT(mel: MLXArray) -> MLXArray {
        let f0Mel = 150.0 + 95.0 * sigmoid(f0Predictor(mel))
        let upsampled = MLXNN.leakyRelu(melUpsample(mel), negativeSlope: 0.1)
        let stftFrames = upsampled.dim(1)
        let f0Expanded = stacked(Array(repeating: f0Mel, count: config.upsampleStride), axis: 2)
            .reshaped(1, stftFrames, 1)
        let phaseDelta = f0Expanded * Float(config.hopLength) / Float(config.sampleRate)
        let phase = (2.0 * Float.pi) * phaseDelta.cumsum(axis: 1)
        let nsf = concatenated([sin(phase), 0.5 * sin(2.0 * phase), 0.25 * sin(3.0 * phase)], axis: -1)
        var h = upsampled + nsfProj(nsf)
        let padCausal = [IntOrPair((0, 0)), IntOrPair((2, 0)), IntOrPair((0, 0))]
        let c1 = resConv1(padded(MLXNN.leakyRelu(h, negativeSlope: 0.1), widths: padCausal))
        let c2 = resConv2(padded(MLXNN.leakyRelu(c1, negativeSlope: 0.1), widths: padCausal))
        h = h + c2
        let bins = config.nFFT / 2 + 1
        let spec = stftProj(h)[0]
        let logMag = clip(spec[0..., ..<bins], min: -3.5, max: 1.2)
        let harmonicFloor = MLXArray((0 ..< bins).map { $0 == 1 || $0 == 2 ? Float(0.42) : Float(0.04) }).reshaped(1, bins)
        let mag = exp(logMag) + harmonicFloor
        let instPhase = Float.pi * tanh(spec[0..., bins...]) + phase[0, 0..., 0 ..< 1]
        return ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: mag,
            phase: instPhase,
            nFFT: config.nFFT,
            hopLength: config.hopLength
        )
    }
}
