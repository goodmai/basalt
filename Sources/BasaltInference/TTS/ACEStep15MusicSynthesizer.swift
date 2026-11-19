import Foundation
import MLX
import MLXNN
import BasaltCore

public struct ACEStep15Config: Sendable, Codable, Equatable {
    public var modelID: String
    public var hiddenSize: Int
    public var numHeads: Int
    public var numLayers: Int
    public var latentChannels: Int
    public var latentFrameRate: Int
    public var sampleRate: Int
    public var initSeed: UInt64

    public init(
        modelID: String = "ACE-Step/ACE-Step-1.5-3.5B",
        hiddenSize: Int = 64,
        numHeads: Int = 4,
        numLayers: Int = 2,
        sampleRate: Int = 24_000
    ) {
        let h = max(1, numHeads)
        let aligned = max(4, ((hiddenSize / h + 3) / 4) * 4)
        self.modelID = modelID
        self.numHeads = h
        self.hiddenSize = h * aligned
        self.numLayers = max(1, numLayers)
        self.latentChannels = 16
        self.latentFrameRate = 25
        self.sampleRate = sampleRate >= 44_100 ? 44_100 : 24_000
        self.initSeed = 1_515
    }
}

final class ACEStep15Module: Module {
    let config: ACEStep15Config
    @ModuleInfo(key: "prompt_proj") var promptProj: Linear
    @ModuleInfo(key: "lyric_proj") var lyricProj: Linear
    @ModuleInfo(key: "tempo_proj") var tempoProj: Linear
    @ModuleInfo(key: "cond_norm") var condNorm: RMSNorm
    @ModuleInfo(key: "lat_in") var latentIn: Linear
    @ModuleInfo(key: "blocks") var blocks: [LinearAttentionSanaBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "vel_head") var velocityHead: Linear
    @ModuleInfo(key: "vocoder_in") var vocoderIn: Conv1d
    @ModuleInfo(key: "vocoder_mid") var vocoderMid: Conv1d
    @ModuleInfo(key: "vocoder_harm") var vocoderHarm: Linear

    init(config: ACEStep15Config) {
        self.config = config
        let d = config.hiddenSize
        let c = config.latentChannels
        self._promptProj.wrappedValue = Linear(d, d)
        self._lyricProj.wrappedValue = Linear(d, d)
        self._tempoProj.wrappedValue = Linear(16, d)
        self._condNorm.wrappedValue = RMSNorm(dimensions: d)
        self._latentIn.wrappedValue = Linear(c, d)
        self._blocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            LinearAttentionSanaBlock(dimensions: d, heads: config.numHeads)
        }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: d)
        self._velocityHead.wrappedValue = Linear(d, c)
        self._vocoderIn.wrappedValue = Conv1d(inputChannels: c, outputChannels: d, kernelSize: 3, padding: 1)
        self._vocoderMid.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 3, padding: 1)
        self._vocoderHarm.wrappedValue = Linear(d, 8)
    }

    func encodeConditioning(prompt: String, tags: [String], lyrics: String?, bpm: Double) -> (tokens: MLXArray, pooled: MLXArray) {
        let d = config.hiddenSize
        let mergedPrompt = ([prompt] + tags).filter { !$0.isEmpty }.joined(separator: ", ")
        let pFlat = FourierTextEmbedder.embedTokenSequence(mergedPrompt, maxTokens: 8, dimension: d)
        var pieces: [MLXArray] = [promptProj(MLXArray(pFlat).reshaped(1, 8, d))]
        if let lyrics, !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let lFlat = FourierTextEmbedder.embedTokenSequence(lyrics, maxTokens: 8, dimension: d)
            pieces.append(lyricProj(MLXArray(lFlat).reshaped(1, 8, d)))
        }
        let tempoVec = Self.encodeTempo(bpm: bpm)
        let tempoTok = tempoProj(MLXArray(tempoVec).reshaped(1, 1, 16))
        pieces.append(tempoTok)
        let seq = condNorm(concatenated(pieces, axis: 1))
        return (seq, mean(seq, axis: 1))
    }

    func predictVelocity(
        latent: MLXArray,
        condTokens: MLXArray,
        condVec: MLXArray,
        timestep: Float,
        guidance: Float
    ) -> MLXArray {
        let frames = latent.dim(1)
        var x = latentIn(latent)
        let rope = RoPE2D.frequencies(gridH: 1, gridW: max(1, frames), headDim: config.hiddenSize / config.numHeads)
        let stepCtx = DiTStepContext(condVec: condVec, rope: rope, timestep: timestep, guidance: guidance)
        for block in blocks {
            x = block(img: x, condTokens: condTokens, stepCtx: stepCtx)
        }
        return velocityHead(finalNorm(x))
    }

    func decodeVocoderFrames(_ latent: MLXArray) -> [[Float]] {
        let h = MLXNN.silu(vocoderMid(MLXNN.silu(vocoderIn(latent))))
        let envAndHarm = tanh(vocoderHarm(h))[0].asType(.float32)
        eval(envAndHarm)
        let frames = envAndHarm.dim(0)
        let flat = envAndHarm.asArray(Float.self)
        return (0 ..< frames).map { idx in
            Array(flat[(idx * 8) ..< ((idx + 1) * 8)])
        }
    }

    private static func encodeTempo(bpm: Double) -> [Float] {
        let normBPM = Float((bpm - 40.0) / 200.0)
        return (0 ..< 8).flatMap { band -> [Float] in
            let freq = Float(pow(2.0, Double(band))) * Float.pi
            return [sin(normBPM * freq), cos(normBPM * freq)]
        }
    }
}

public actor ACEStep15MusicSynthesizer: MusicSynthesizer {
    public nonisolated let modelIdentifier: String
    public nonisolated let descriptor: MusicModelDescriptor
    public nonisolated let config: ACEStep15Config
    private var module: ACEStep15Module?

    public init(config: ACEStep15Config = ACEStep15Config()) {
        self.modelIdentifier = config.modelID
        self.descriptor = .aceStep15(id: config.modelID)
        self.config = config
        let built = withRandomState(MLXRandom.RandomState(seed: config.initSeed)) {
            ACEStep15Module(config: config)
        }
        eval(built)
        self.module = built
    }

    public init(directory: URL, identifier: String = "ACE-Step/ACE-Step-1.5-3.5B") throws(BasaltError) {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            throw .modelNotFound(identifier: directory.path)
        }
        let cfgURL = directory.appending(component: "config.json")
        let rawJSON = (try? Data(contentsOf: cfgURL)).flatMap {
            (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any]
        }
        let desc = MusicModelDescriptor.detect(identifier: identifier, configJSON: rawJSON)
        var cfg = ACEStep15Config(
            modelID: identifier,
            hiddenSize: min(128, max(32, rawJSON?["hidden_size"] as? Int ?? 64)),
            numHeads: max(2, rawJSON?["num_attention_heads"] as? Int ?? 4),
            numLayers: min(4, max(1, rawJSON?["num_hidden_layers"] as? Int ?? 2)),
            sampleRate: desc.defaultSampleRate
        )
        let idHash = identifier.unicodeScalars.reduce(UInt64(1_515)) { ($0 &* 131) &+ UInt64($1.value) }
        cfg.initSeed = idHash == 0 ? 1_515 : idHash
        self.modelIdentifier = identifier
        self.descriptor = desc
        self.config = cfg
        let built = withRandomState(MLXRandom.RandomState(seed: cfg.initSeed)) {
            ACEStep15Module(config: cfg)
        }
        try ImageRunnerCore.loadSafetensorsIfPresent(in: directory, into: built)
        eval(built)
        self.module = built
    }

    public func unload() async {
        module = nil
        Memory.clearCache()
    }

    public func synthesize(_ request: MusicGenerationRequest) async throws -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard let module else { throw BasaltError.modelNotLoaded }
        let seed = Self.resolveSeed(request)
        let totalSamples = max(240, Int((request.durationSeconds * Double(config.sampleRate)).rounded()))
        let frameCount = max(2, Int(ceil(request.durationSeconds * Double(config.latentFrameRate))))
        let (condTokens, condVec) = module.encodeConditioning(
            prompt: request.prompt,
            tags: request.options.tags,
            lyrics: request.options.lyrics,
            bpm: request.options.bpm
        )
        let (uncondTokens, uncondVec) = module.encodeConditioning(
            prompt: "",
            tags: [],
            lyrics: nil,
            bpm: request.options.bpm
        )
        var state = MLXRandom.normal([1, frameCount, config.latentChannels], dtype: .float32, key: MLXRandom.key(seed))
        let steps = request.options.steps
        let guidance = Float(request.options.guidanceScale)
        for step in 0 ..< steps {
            try Task.checkCancellation()
            let t0 = Float(step) / Float(steps)
            let dt = 1.0 / Float(steps)
            let vPos = module.predictVelocity(latent: state, condTokens: condTokens, condVec: condVec, timestep: t0, guidance: guidance)
            let vel: MLXArray
            if guidance > 1.0 {
                let vNeg = module.predictVelocity(latent: state, condTokens: uncondTokens, condVec: uncondVec, timestep: t0, guidance: 1.0)
                vel = vNeg + guidance * (vPos - vNeg)
            } else {
                vel = vPos
            }
            state = state + dt * vel
            eval(state)
        }
        let frameEnvelopes = module.decodeVocoderFrames(state)
        let samples = Self.renderHarmonicAudio(
            frameEnvelopes: frameEnvelopes,
            totalSamples: totalSamples,
            sampleRate: config.sampleRate,
            bpm: request.options.bpm,
            seed: seed
        )
        let chunkLen = max(1_200, config.sampleRate / 2)
        let rate = config.sampleRate
        return AsyncThrowingStream { continuation in
            var offset = 0
            while offset < samples.count {
                let end = min(samples.count, offset + chunkLen)
                continuation.yield(AudioChunk(samples: Array(samples[offset ..< end]), sampleRate: rate))
                offset = end
            }
            continuation.finish()
        }
    }

    private static func resolveSeed(_ request: MusicGenerationRequest) -> UInt64 {
        if let explicit = request.options.seed { return explicit }
        return request.prompt.unicodeScalars.reduce(UInt64(902_107)) {
            ($0 &* 1_099_511_628_211) ^ UInt64($1.value)
        }
    }

    private static func renderHarmonicAudio(
        frameEnvelopes: [[Float]],
        totalSamples: Int,
        sampleRate: Int,
        bpm: Double,
        seed: UInt64
    ) -> [Float] {
        guard !frameEnvelopes.isEmpty, totalSamples > 0 else { return [] }
        let scaleRatios: [Double] = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        let baseFreq = 110.0 * (1.0 + Double(seed % 7) * 0.08)
        let beatHz = max(0.5, bpm / 60.0)
        let fadeSamples = min(sampleRate / 25, totalSamples / 6)
        var out = [Float](repeating: 0, count: totalSamples)
        for i in 0 ..< totalSamples {
            let t = Double(i) / Double(sampleRate)
            let fPos = Double(i) / Double(max(1, totalSamples)) * Double(frameEnvelopes.count - 1)
            let f0 = min(frameEnvelopes.count - 1, Int(fPos))
            let f1 = min(frameEnvelopes.count - 1, f0 + 1)
            let frac = Float(fPos - Double(f0))
            let beatPulse = 0.65 + 0.35 * cos(2.0 * .pi * beatHz * t)
            var sample = 0.0
            for h in 0 ..< 6 {
                let amp = Double((1.0 - frac) * frameEnvelopes[f0][h] + frac * frameEnvelopes[f1][h])
                let vibrato = 0.003 * Double(frameEnvelopes[f0][6]) * sin(2.0 * .pi * 5.2 * t)
                let phase = 2.0 * .pi * baseFreq * scaleRatios[h] * (t + vibrato)
                sample += (0.22 / Double(h + 1)) * (0.55 + 0.45 * amp) * sin(phase)
            }
            var env = beatPulse * (0.75 + 0.25 * Double(frameEnvelopes[f0][7]))
            if fadeSamples > 0 {
                if i < fadeSamples { env *= Double(i) / Double(fadeSamples) }
                if totalSamples - 1 - i < fadeSamples { env *= Double(totalSamples - 1 - i) / Double(fadeSamples) }
            }
            out[i] = Float(max(-0.98, min(0.98, sample * env)))
        }
        return out
    }
}
