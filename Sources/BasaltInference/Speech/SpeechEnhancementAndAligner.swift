import Foundation
import MLX
import MLXNN
import BasaltCore

public struct MossFormer2Config: Sendable, Codable, Equatable {
    public var nFFT: Int
    public var hopLength: Int
    public var hiddenDim: Int
    public var lowCutHz: Double
    public var highCutHz: Double
    public var outOfBandFloor: Float = 0.015
    public var maskFloor: Float = 0.02

    public static var tiny: MossFormer2Config {
        MossFormer2Config(nFFT: 256, hopLength: 64, hiddenDim: 32)
    }

    public init(
        nFFT: Int = 512,
        hopLength: Int = 128,
        hiddenDim: Int = 64,
        lowCutHz: Double = 70.0,
        highCutHz: Double = 3_800.0
    ) {
        let evenFFT = max(32, nFFT.isMultiple(of: 2) ? nFFT : nFFT + 1)
        self.nFFT = evenFFT
        self.hopLength = max(8, min(evenFFT / 2, hopLength))
        self.hiddenDim = max(16, hiddenDim)
        self.lowCutHz = max(10.0, lowCutHz)
        self.highCutHz = max(self.lowCutHz + 200.0, highCutHz)
    }
}

final class MossFormer2MaskModule: Module {
    let bins: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "temporal_conv") var temporalConv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "residual_scale") var residualScale: MLXArray

    init(_ config: MossFormer2Config, seed: UInt64) {
        let b = config.nFFT / 2 + 1
        let d = config.hiddenDim
        self.bins = b
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            let inp = Linear(b, d)
            let conv = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 3, padding: 1, groups: d)
            let ln = LayerNorm(dimensions: d)
            let gate = Linear(d, d)
            let out = Linear(d, b)
            let scale = MLXArray.zeros([1], dtype: .float32)
            return (inp, conv, ln, gate, out, scale)
        }
        self._inProj.wrappedValue = built.0
        self._temporalConv.wrappedValue = built.1
        self._norm.wrappedValue = built.2
        self._gateProj.wrappedValue = built.3
        self._outProj.wrappedValue = built.4
        self._residualScale.wrappedValue = built.5
        super.init()
        eval(self)
    }

    func refineMask(logMagnitude: MLXArray, baseMask: MLXArray, floor: Float) -> MLXArray {
        let x = logMagnitude.expandedDimensions(axis: 0)
        let h0 = MLXNN.silu(inProj(x))
        let hConv = norm(h0 + temporalConv(h0))
        let gated = hConv * MLX.sigmoid(gateProj(hConv))
        let delta = MLX.tanh(outProj(gated)[0]) * residualScale
        return clip(baseMask + delta, min: floor, max: 1.0)
    }
}

public actor MossFormer2Enhancer {
    public nonisolated let config: MossFormer2Config
    private var module: MossFormer2MaskModule?
    private var epoch = 0

    public init(config: MossFormer2Config = MossFormer2Config(), seed: UInt64 = 578) {
        self.config = config
        self.module = MossFormer2MaskModule(config, seed: seed)
    }

    public func unload() {
        epoch &+= 1
        module = nil
        Memory.clearCache()
    }

    public func enhance(_ samples: [Float], sampleRate: Int = 16_000) throws(BasaltError) -> [Float] {
        guard sampleRate > 0 else {
            throw .invalidRequestStructure(details: "MossFormer2 sample rate must be positive")
        }
        guard !samples.isEmpty else { return [] }
        let sanitized = samples.map { $0.isFinite ? min(1, max(-1, $0)) : 0 }
        guard sanitized.count >= 16 else { return sanitized }
        guard let module else { throw .modelNotLoaded }
        let currentEpoch = epoch
        let enhanced = Self.runSTFTEnhancement(sanitized, sampleRate: sampleRate, config: config, module: module)
        guard epoch == currentEpoch else { throw .modelNotLoaded }
        return enhanced
    }

    private static func runSTFTEnhancement(
        _ samples: [Float],
        sampleRate: Int,
        config: MossFormer2Config,
        module: MossFormer2MaskModule
    ) -> [Float] {
        let n = samples.count
        let nFFT = config.nFFT
        let hop = config.hopLength
        let padLeft = nFFT / 2
        let totalNeeded = padLeft + n + nFFT
        let frames = max(1, (totalNeeded - nFFT) / hop + 1)
        let paddedLen = (frames - 1) * hop + nFFT
        var buf = Array(repeating: Float(0), count: padLeft)
        buf.append(contentsOf: samples)
        if buf.count < paddedLen {
            buf.append(contentsOf: repeatElement(Float(0), count: paddedLen - buf.count))
        }

        let window = MLXArray(ISTFTNetSynthesizer.hannWindow(length: nFFT)).reshaped(1, nFFT)
        let frameOffsets = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(hop)
        let sampleOffsets = MLXArray(Int32(0) ..< Int32(nFFT)).expandedDimensions(axis: 0)
        let flatIndices = (frameOffsets + sampleOffsets).reshaped(-1)
        let framed = MLXArray(buf).take(flatIndices, axis: 0).reshaped(frames, nFFT) * window

        let spec = MLXFFT.rfft(framed, n: nFFT, axis: -1)
        let real = spec.realPart()
        let imag = spec.imaginaryPart()
        let power = real * real + imag * imag
        let baseMask = spectralWienerMask(power: power, sampleRate: sampleRate, config: config)
        let logMag = log(maximum(sqrt(power), MLXArray(Float(1e-6))))
        let mask = module.refineMask(logMagnitude: logMag, baseMask: baseMask, floor: config.maskFloor)

        let maskedSpec = (real * mask).asType(.complex64) + MLXArray(real: 0, imaginary: 1) * (imag * mask).asType(.complex64)
        let timeFrames = MLXFFT.irfft(maskedSpec, n: nFFT, axis: -1) * window
        let winSq = broadcast(window * window, to: [frames, nFFT])
        let numer = MLXArray.zeros([paddedLen], dtype: .float32).at[flatIndices].add(timeFrames.reshaped(-1))
        let denom = MLXArray.zeros([paddedLen], dtype: .float32).at[flatIndices].add(winSq.reshaped(-1))
        let ola = (numer / maximum(denom, MLXArray(Float(1e-5))))[padLeft ..< (padLeft + n)]
        eval(ola)
        return ola.asArray(Float.self)
    }

    private static func spectralWienerMask(power: MLXArray, sampleRate: Int, config: MossFormer2Config) -> MLXArray {
        let bins = config.nFFT / 2 + 1
        let bandWeights = (0 ..< bins).map { bin -> Float in
            let hz = Double(bin * sampleRate) / Double(config.nFFT)
            if hz < config.lowCutHz * 0.5 || hz > config.highCutHz * 1.15 {
                return config.outOfBandFloor
            }
            if hz < config.lowCutHz {
                let t = (hz - config.lowCutHz * 0.5) / max(1.0, config.lowCutHz * 0.5)
                return config.outOfBandFloor + Float(t) * (1.0 - config.outOfBandFloor)
            }
            if hz > config.highCutHz {
                let t = (hz - config.highCutHz) / max(1.0, config.highCutHz * 0.15)
                return 1.0 - Float(min(1.0, t)) * (1.0 - config.outOfBandFloor)
            }
            return 1.0
        }
        let prior = MLXArray(bandWeights).reshaped(1, bins)
        let meanPow = power.mean(axis: -1, keepDims: true)
        let highBin = min(bins - 1, max(1, Int(config.highCutHz * Double(config.nFFT) / Double(sampleRate))))
        let outMean = power[0..., highBin...].mean(axis: -1, keepDims: true)
        let noiseFloor = maximum(MLXArray(Float(1e-9)), outMean * 1.35 + meanPow * 0.18)
        let pSq = power * power
        let nSq = noiseFloor * noiseFloor
        let wiener = pSq / maximum(pSq + nSq, MLXArray(Float(1e-12)))
        return clip(wiener * prior, min: config.maskFloor, max: 1.0)
    }
}

public enum Qwen3CTCAlignerCore {
    public static func align(
        logProbabilities: [[Float]],
        wordTokenIDs: [[Int]],
        words: [String],
        frameDurationSeconds: Double,
        blankID: Int = 0
    ) -> [WordTimestamp] {
        let cleanWords = words.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard !cleanWords.isEmpty else { return [] }
        let dt = max(0, frameDurationSeconds.isFinite ? frameDurationSeconds : 0.02)
        let frameCount = logProbabilities.count
        let totalDuration = Double(frameCount) * dt
        guard frameCount >= cleanWords.count, dt > 0 else {
            return proportionalFallback(words: cleanWords, totalSeconds: totalDuration)
        }
        let path = viterbiMonotonicPath(
            logProbabilities: logProbabilities,
            wordTokenIDs: wordTokenIDs,
            wordCount: cleanWords.count,
            blankID: blankID
        )
        return extractTimestamps(
            path: path,
            logProbabilities: logProbabilities,
            wordTokenIDs: wordTokenIDs,
            words: cleanWords,
            dt: dt
        )
    }

    private static func viterbiMonotonicPath(
        logProbabilities: [[Float]],
        wordTokenIDs: [[Int]],
        wordCount: Int,
        blankID: Int
    ) -> [Int] {
        let frames = logProbabilities.count
        let stateCount = 2 * wordCount + 1
        var prevScores = Array(repeating: -Double.infinity, count: stateCount)
        var backpointers = Array(repeating: Array(repeating: 0, count: stateCount), count: frames)
        prevScores[0] = emissionScore(row: logProbabilities[0], state: 0, wordTokens: wordTokenIDs, blankID: blankID)
        prevScores[1] = emissionScore(row: logProbabilities[0], state: 1, wordTokens: wordTokenIDs, blankID: blankID)

        for t in 1 ..< frames {
            var nextScores = Array(repeating: -Double.infinity, count: stateCount)
            for s in 0 ..< stateCount {
                var bestScore = prevScores[s]
                var bestPrev = s
                if s >= 1, prevScores[s - 1] > bestScore {
                    bestScore = prevScores[s - 1]
                    bestPrev = s - 1
                }
                if s % 2 == 1, s >= 3, prevScores[s - 2] > bestScore {
                    bestScore = prevScores[s - 2]
                    bestPrev = s - 2
                }
                if bestScore > -.infinity {
                    let emit = emissionScore(row: logProbabilities[t], state: s, wordTokens: wordTokenIDs, blankID: blankID)
                    nextScores[s] = bestScore + emit
                    backpointers[t][s] = bestPrev
                }
            }
            prevScores = nextScores
        }

        var current = prevScores[stateCount - 1] >= prevScores[stateCount - 2] ? stateCount - 1 : stateCount - 2
        var path = Array(repeating: 0, count: frames)
        for t in stride(from: frames - 1, through: 0, by: -1) {
            path[t] = current
            if t > 0 { current = backpointers[t][current] }
        }
        return path
    }

    private static func emissionScore(row: [Float], state: Int, wordTokens: [[Int]], blankID: Int) -> Double {
        guard !row.isEmpty else { return -1.0 }
        if state.isMultiple(of: 2) {
            let idx = min(row.count - 1, max(0, blankID))
            let val = row[idx]
            return val.isFinite ? Double(val) : -10.0
        }
        let wordIdx = state / 2
        let ids = wordIdx < wordTokens.count && !wordTokens[wordIdx].isEmpty ? wordTokens[wordIdx] : [1]
        var acc = 0.0
        for token in ids {
            let idx = min(row.count - 1, max(0, token))
            let val = row[idx]
            acc += val.isFinite ? Double(val) : -10.0
        }
        return acc / Double(ids.count)
    }

    private static func extractTimestamps(
        path: [Int],
        logProbabilities: [[Float]],
        wordTokenIDs: [[Int]],
        words: [String],
        dt: Double
    ) -> [WordTimestamp] {
        let totalSeconds = Double(path.count) * dt
        var output: [WordTimestamp] = []
        output.reserveCapacity(words.count)
        var cursorSeconds = 0.0

        for (wIdx, word) in words.enumerated() {
            let targetState = 2 * wIdx + 1
            var firstFrame: Int?
            var lastFrame: Int?
            var logProbSum = 0.0
            var count = 0
            for (t, s) in path.enumerated() where s == targetState {
                if firstFrame == nil { firstFrame = t }
                lastFrame = t
                logProbSum += emissionScore(row: logProbabilities[t], state: targetState, wordTokens: wordTokenIDs, blankID: 0)
                count += 1
            }
            let rawStart = firstFrame.map { Double($0) * dt } ?? cursorSeconds
            let rawEnd = lastFrame.map { Double($0 + 1) * dt } ?? rawStart
            let clampedStart = min(totalSeconds, max(cursorSeconds, rawStart))
            let clampedEnd = min(totalSeconds, max(clampedStart, rawEnd))
            let prob: Float = count > 0 ? Float(exp(min(0.0, logProbSum / Double(count)))) : 0.5
            output.append(WordTimestamp(word: word, start: clampedStart, end: clampedEnd, probability: prob))
            cursorSeconds = clampedEnd
        }
        return output
    }

    private static func proportionalFallback(words: [String], totalSeconds: Double) -> [WordTimestamp] {
        let safeTotal = max(0, totalSeconds.isFinite ? totalSeconds : 0)
        let count = Double(max(1, words.count))
        var cursor = 0.0
        return words.enumerated().map { idx, word in
            let start = max(cursor, safeTotal * Double(idx) / count)
            let end = max(start, safeTotal * Double(idx + 1) / count)
            cursor = end
            return WordTimestamp(word: word, start: start, end: end, probability: 0.5)
        }
    }
}

final class Qwen3ForcedAlignerModule: Module {
    let vocabSize: Int
    let frameHop: Int
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "ctc_head") var ctcHead: Linear

    init(vocabSize: Int, hiddenDim: Int, frameHop: Int, seed: UInt64) {
        self.vocabSize = vocabSize
        self.frameHop = frameHop
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            let c = Conv1d(inputChannels: 1, outputChannels: hiddenDim, kernelSize: frameHop * 2, stride: frameHop, padding: frameHop / 2)
            let ln = LayerNorm(dimensions: hiddenDim)
            let head = Linear(hiddenDim, vocabSize)
            return (c, ln, head)
        }
        self._conv.wrappedValue = built.0
        self._norm.wrappedValue = built.1
        self._ctcHead.wrappedValue = built.2
        super.init()
        eval(self)
    }

    func forwardLogProbs(_ waveform: MLXArray) -> MLXArray {
        let h = norm(MLXNN.gelu(conv(waveform)))
        return MLXNN.logSoftmax(ctcHead(h), axis: -1)[0]
    }
}

public actor Qwen3ForcedAligner {
    public nonisolated let vocabSize: Int
    public nonisolated let frameDurationSeconds: Double
    private var module: Qwen3ForcedAlignerModule?

    public init(vocabSize: Int = 128, hiddenDim: Int = 48, frameDurationSeconds: Double = 0.02, seed: UInt64 = 582) {
        let v = max(16, vocabSize)
        let dt = max(0.005, frameDurationSeconds)
        self.vocabSize = v
        self.frameDurationSeconds = dt
        let hop = max(80, Int((16_000.0 * dt).rounded()))
        self.module = Qwen3ForcedAlignerModule(vocabSize: v, hiddenDim: max(16, hiddenDim), frameHop: hop, seed: seed)
    }

    public func unload() {
        module = nil
        Memory.clearCache()
    }

    public func align(samples: [Float], sampleRate: Int = 16_000, transcript: String) throws(BasaltError) -> [WordTimestamp] {
        guard sampleRate > 0 else {
            throw .invalidRequestStructure(details: "Qwen3ForcedAligner sample rate must be positive")
        }
        let words = transcript.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !words.isEmpty, !samples.isEmpty else { return [] }
        guard let module else { throw .modelNotLoaded }

        let hopSamples = max(1, Int((Double(sampleRate) * frameDurationSeconds).rounded()))
        let resampled = sampleRate == 16_000
            ? samples
            : TTSWaveformMetrics.resampleLinear(samples, from: sampleRate, to: 16_000)
        let sanitized = resampled.map { $0.isFinite ? min(1, max(-1, $0)) : 0 }
        let logProbsTensor = module.forwardLogProbs(MLXArray(sanitized).reshaped(1, sanitized.count, 1))
        eval(logProbsTensor)
        let frameCount = logProbsTensor.dim(0)
        let flat = logProbsTensor.reshaped(-1).asArray(Float.self)
        let v = vocabSize
        let actualDt = Double(samples.count) / Double(sampleRate * max(1, frameCount))

        let rows: [[Float]] = (0 ..< frameCount).map { f in
            var row = Array(flat[(f * v) ..< ((f + 1) * v)])
            let startIdx = min(samples.count, f * hopSamples)
            let endIdx = min(samples.count, startIdx + hopSamples)
            let energy = startIdx < endIdx ? TTSWaveformMetrics.rms(Array(samples[startIdx ..< endIdx])) : 0
            if energy < 0.015 {
                row[0] += 2.5
            } else {
                row[0] -= 2.5
            }
            return row
        }
        let wordTokens: [[Int]] = words.map { word in
            let ids = word.unicodeScalars.prefix(16).map { (Int($0.value) % max(1, v - 1)) + 1 }
            return ids.isEmpty ? [1] : ids
        }
        return Qwen3CTCAlignerCore.align(
            logProbabilities: rows,
            wordTokenIDs: wordTokens,
            words: words,
            frameDurationSeconds: actualDt,
            blankID: 0
        )
    }

    public func alignTranscription(_ transcription: Transcription, audio: AudioClip) throws(BasaltError) -> Transcription {
        let aligned = try align(samples: audio.samples, sampleRate: Int(AudioClip.sampleRate), transcript: transcription.text)
        return Transcription(
            text: transcription.text,
            language: transcription.language,
            segments: transcription.segments,
            words: aligned
        )
    }
}
