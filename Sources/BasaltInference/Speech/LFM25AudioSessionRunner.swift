import Foundation
import MLX
import MLXNN
import BasaltCore

public struct LFM25AudioDuplexConfig: Sendable, Codable, Equatable {
    public var modelID: String
    public var textVocabSize: Int
    public var hiddenSize: Int
    public var codecConfig: MimiCodecConfig
    public var turnConfig: VoiceTurnConfig

    public static var tiny: LFM25AudioDuplexConfig {
        LFM25AudioDuplexConfig(
            modelID: "LiquidAI/LFM2.5-Audio-1.5B-Tiny",
            textVocabSize: 256,
            hiddenSize: 48,
            codecConfig: .tiny,
            turnConfig: VoiceTurnConfig()
        )
    }

    public init(
        modelID: String = "LiquidAI/LFM2.5-Audio-1.5B",
        textVocabSize: Int = 2_048,
        hiddenSize: Int = 128,
        codecConfig: MimiCodecConfig = MimiCodecConfig(),
        turnConfig: VoiceTurnConfig = VoiceTurnConfig()
    ) {
        self.modelID = modelID
        self.textVocabSize = max(16, textVocabSize)
        self.hiddenSize = max(16, hiddenSize)
        self.codecConfig = codecConfig
        self.turnConfig = turnConfig
    }
}

public struct LFM25AudioStepOutput: Sendable, Equatable {
    public let turnDecision: TurnDecision
    public let inputAudioCodes: [Int32]
    public let textToken: Int32?
    public let outputAudioCodes: [Int32]
    public let synthesizedSamples: [Float]
    public let sampleRate: Int

    public init(
        turnDecision: TurnDecision,
        inputAudioCodes: [Int32],
        textToken: Int32?,
        outputAudioCodes: [Int32],
        synthesizedSamples: [Float],
        sampleRate: Int
    ) {
        self.turnDecision = turnDecision
        self.inputAudioCodes = inputAudioCodes
        self.textToken = textToken
        self.outputAudioCodes = outputAudioCodes
        self.synthesizedSamples = synthesizedSamples
        self.sampleRate = sampleRate
    }
}

public struct LFM25AudioDuplexState: Sendable, Equatable {
    public enum Mode: String, Sendable, Codable, Equatable {
        case listening
        case holdingPause
        case responding
    }

    public private(set) var mode: Mode = .listening
    public private(set) var stepCount: Int = 0
    public private(set) var bargeInCount: Int = 0
    public private(set) var lastCommittedSpeaker: Int?
    public private(set) var inputAudioFrames: [[Int32]] = []
    public private(set) var generatedTextTokens: [Int32] = []
    public private(set) var generatedAudioFrames: [[Int32]] = []
    public var turnCore: VoiceTurnCore

    public init(turnConfig: VoiceTurnConfig = VoiceTurnConfig()) {
        self.turnCore = VoiceTurnCore(config: turnConfig)
    }

    public mutating func ingestAudioTokens(
        _ codes: [Int32],
        vadProbability: Float,
        semanticEOTProbability: Float,
        speakerPosteriors: [Float] = [],
        frameDurationSeconds: Double
    ) -> TurnDecision {
        stepCount += 1
        inputAudioFrames.append(codes)
        if mode == .responding, vadProbability >= turnCore.config.vadSpeechThreshold {
            bargeInCount += 1
            mode = .listening
        }
        let decision = turnCore.step(
            vadProbability: vadProbability,
            semanticEOTProbability: semanticEOTProbability,
            speakerPosteriors: speakerPosteriors,
            frameDurationSeconds: frameDurationSeconds
        )
        switch decision {
        case .listening:
            if mode != .responding { mode = .listening }
        case .holdThinkingPause:
            if mode != .responding { mode = .holdingPause }
        case .commitTurn(let speakerId):
            lastCommittedSpeaker = speakerId
            mode = .responding
        }
        return decision
    }

    public mutating func appendGeneratedStep(textToken: Int32, audioCodes: [Int32]) {
        generatedTextTokens.append(textToken)
        generatedAudioFrames.append(audioCodes)
    }

    public mutating func reset() {
        mode = .listening
        stepCount = 0
        bargeInCount = 0
        lastCommittedSpeaker = nil
        inputAudioFrames.removeAll()
        generatedTextTokens.removeAll()
        generatedAudioFrames.removeAll()
        turnCore.resetSession()
    }
}

final class LFM25AudioBackboneModule: Module {
    let config: LFM25AudioDuplexConfig
    @ModuleInfo(key: "audio_in_emb") var audioInEmbeddings: [Embedding]
    @ModuleInfo(key: "text_emb") var textEmbedding: Embedding
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "state_proj") var stateProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "eot_head") var eotHead: Linear
    @ModuleInfo(key: "audio_heads") var audioHeads: [Linear]

    init(_ config: LFM25AudioDuplexConfig, seed: UInt64) {
        self.config = config
        let h = config.hiddenSize
        let q = config.codecConfig.numCodebooks
        let k = config.codecConfig.codebookSize
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            let inBooks = (0 ..< q).map { _ in Embedding(embeddingCount: k, dimensions: h) }
            let txtEmb = Embedding(embeddingCount: config.textVocabSize, dimensions: h)
            let gate = Linear(h * 2, h)
            let stProj = Linear(h * 2, h)
            let rms = RMSNorm(dimensions: h)
            let txtHead = Linear(h, config.textVocabSize)
            let eot = Linear(h, 1)
            let outHeads = (0 ..< q).map { _ in Linear(h, k) }
            return (inBooks, txtEmb, gate, stProj, rms, txtHead, eot, outHeads)
        }
        self._audioInEmbeddings.wrappedValue = built.0
        self._textEmbedding.wrappedValue = built.1
        self._gateProj.wrappedValue = built.2
        self._stateProj.wrappedValue = built.3
        self._norm.wrappedValue = built.4
        self._textHead.wrappedValue = built.5
        self._eotHead.wrappedValue = built.6
        self._audioHeads.wrappedValue = built.7
        super.init()
        eval(self)
    }

    func forwardStep(inputCodes: [Int32], prevTextToken: Int32, prevHidden: MLXArray) -> (hidden: MLXArray, eotProb: Float, textToken: Int32, audioCodes: [Int32]) {
        let cap = Int32(config.codecConfig.codebookSize - 1)
        var audioSum = MLXArray.zeros([1, config.hiddenSize], dtype: .float32)
        for (idx, emb) in audioInEmbeddings.enumerated() {
            let code = idx < inputCodes.count ? min(cap, max(0, inputCodes[idx])) : 0
            audioSum = audioSum + emb(MLXArray([code])) / Float(idx + 1)
        }
        let txtId = min(Int32(config.textVocabSize - 1), max(0, prevTextToken))
        let fusedIn = audioSum + textEmbedding(MLXArray([txtId]))
        let concat = concatenated([prevHidden, fusedIn], axis: -1)
        let gate = MLX.sigmoid(gateProj(concat))
        let candidate = MLXNN.silu(stateProj(concat))
        let nextHidden = norm((1.0 - gate) * prevHidden + gate * candidate)
        let eotProb = MLX.sigmoid(eotHead(nextHidden)).reshaped(-1).item(Float.self)
        let nextText = textHead(nextHidden).reshaped(-1).argMax(axis: 0).item(Int32.self)
        let nextAudio = audioHeads.map { head in
            head(nextHidden).reshaped(-1).argMax(axis: 0).item(Int32.self)
        }
        return (nextHidden, eotProb, nextText, nextAudio)
    }
}

public actor LFM25AudioSessionRunner {
    public nonisolated let config: LFM25AudioDuplexConfig
    private var codec: MimiStreamingCodec?
    private var backbone: LFM25AudioBackboneModule?
    private var hidden: MLXArray?
    private var duplexState: LFM25AudioDuplexState
    private var epoch = 0

    public init(config: LFM25AudioDuplexConfig = LFM25AudioDuplexConfig(), seed: UInt64 = 577) {
        self.config = config
        self.codec = MimiStreamingCodec(config: config.codecConfig, seed: seed)
        self.backbone = LFM25AudioBackboneModule(config, seed: seed &+ 1)
        self.hidden = MLXArray.zeros([1, config.hiddenSize], dtype: .float32)
        self.duplexState = LFM25AudioDuplexState(turnConfig: config.turnConfig)
    }

    public func currentState() -> LFM25AudioDuplexState {
        duplexState
    }

    public func reset() {
        duplexState.reset()
        hidden = MLXArray.zeros([1, config.hiddenSize], dtype: .float32)
    }

    public func unload() {
        epoch &+= 1
        codec = nil
        backbone = nil
        hidden = nil
        Memory.clearCache()
    }

    public func step(
        pcmFrame: [Float],
        vadOverride: Float? = nil,
        eotOverride: Float? = nil,
        speakerPosteriors: [Float] = []
    ) throws(BasaltError) -> LFM25AudioStepOutput {
        guard let codec, let backbone else { throw .modelNotLoaded }
        let currentEpoch = epoch
        let inCodes = codec.encode(pcmFrame).first ?? Array(repeating: 0, count: config.codecConfig.numCodebooks)
        let prevText = duplexState.generatedTextTokens.last ?? 1
        let prevHidden = hidden ?? MLXArray.zeros([1, config.hiddenSize], dtype: .float32)
        let forward = backbone.forwardStep(inputCodes: inCodes, prevTextToken: prevText, prevHidden: prevHidden)
        eval(forward.hidden)
        guard epoch == currentEpoch else { throw .modelNotLoaded }
        hidden = forward.hidden

        let rate = config.codecConfig.sampleRate
        let decision = duplexState.ingestAudioTokens(
            inCodes,
            vadProbability: vadOverride ?? SileroVADScorer.score(pcmFrame, sampleRate: rate),
            semanticEOTProbability: eotOverride ?? forward.eotProb,
            speakerPosteriors: speakerPosteriors,
            frameDurationSeconds: Double(max(1, pcmFrame.count)) / Double(rate)
        )
        guard duplexState.mode == .responding else {
            return LFM25AudioStepOutput(
                turnDecision: decision,
                inputAudioCodes: inCodes,
                textToken: nil,
                outputAudioCodes: [],
                synthesizedSamples: [],
                sampleRate: rate
            )
        }
        let synthSamples = codec.decode([forward.audioCodes], expectedSamples: config.codecConfig.frameSamples)
        duplexState.appendGeneratedStep(textToken: forward.textToken, audioCodes: forward.audioCodes)
        return LFM25AudioStepOutput(
            turnDecision: decision,
            inputAudioCodes: inCodes,
            textToken: forward.textToken,
            outputAudioCodes: forward.audioCodes,
            synthesizedSamples: synthSamples,
            sampleRate: rate
        )
    }
}
