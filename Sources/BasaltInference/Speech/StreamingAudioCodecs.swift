import Foundation
import MLX
import MLXNN
import BasaltCore

public struct MimiCodecConfig: Sendable, Codable, Equatable {
    public static let defaultStrides = [8, 6, 5, 4]

    public var sampleRate: Int
    public var strides: [Int]
    public var numCodebooks: Int
    public var codebookSize: Int
    public var latentDim: Int

    public var frameSamples: Int {
        max(1, strides.reduce(1, *))
    }

    public var frameStrideMs: Double {
        Double(frameSamples) / Double(max(1, sampleRate)) * 1_000.0
    }

    public static var tiny: MimiCodecConfig {
        MimiCodecConfig(sampleRate: 24_000, strides: defaultStrides, numCodebooks: 8, codebookSize: 64, latentDim: 32)
    }

    public init(
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        strides: [Int] = Self.defaultStrides,
        numCodebooks: Int = 8,
        codebookSize: Int = 256,
        latentDim: Int = 64
    ) {
        self.sampleRate = max(8_000, sampleRate)
        let validStrides = strides.map { max(2, $0) }
        self.strides = validStrides.isEmpty ? Self.defaultStrides : validStrides
        self.numCodebooks = max(1, numCodebooks)
        self.codebookSize = max(8, codebookSize)
        self.latentDim = max(8, latentDim)
    }
}

enum RVQQuantizerMath {
    static func padWaveform(_ samples: [Float], multipleOf frameSamples: Int) -> MLXArray {
        let unit = max(1, frameSamples)
        let sanitized = samples.map { $0.isFinite ? min(1, max(-1, $0)) : 0 }
        let count = max(unit, ((max(1, sanitized.count) + unit - 1) / unit) * unit)
        let paddedSamples = sanitized.count < count
            ? sanitized + Array(repeating: Float(0), count: count - sanitized.count)
            : sanitized
        return MLXArray(paddedSamples).reshaped(1, count, 1)
    }

    static func quantize(_ x: MLXArray, table: Embedding) -> (codes: MLXArray, quantized: MLXArray) {
        let b = x.dim(0), t = x.dim(1), d = x.dim(2)
        let flat = x.asType(.float32).reshaped(b * t, d)
        let weight = table.weight.asType(.float32)
        let xNorm = (flat * flat).sum(axis: -1, keepDims: true)
        let wNorm = (weight * weight).sum(axis: -1).reshaped(1, weight.dim(0))
        let distances = xNorm - 2.0 * matmul(flat, weight.T) + wNorm
        let indices = distances.argMin(axis: -1).asType(.int32).reshaped(b, t)
        return (indices, table(indices))
    }

    static func causalAudioHead(_ h: MLXArray, conv: Conv1d) -> MLXArray {
        let paddedH = padded(h, widths: [IntOrPair((0, 0)), IntOrPair((2, 0)), IntOrPair((0, 0))])
        return MLX.tanh(conv(paddedH)).reshaped(-1)
    }
}

final class CausalConv1dStage: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let padLeft: Int

    init(inChannels: Int, outChannels: Int, stride: Int) {
        let s = max(1, stride)
        let kernel = max(2, s * 2)
        self.padLeft = kernel - s
        self._conv.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernel, stride: s, padding: 0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xPad = padLeft > 0
            ? padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padLeft, 0)), IntOrPair((0, 0))])
            : x
        return MLXNN.elu(conv(xPad))
    }
}

public final class MimiStreamingCodec: Module {
    public let config: MimiCodecConfig
    @ModuleInfo(key: "encoder_stages") var encoderStages: [CausalConv1dStage]
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "codebooks") var codebooks: [Embedding]
    @ModuleInfo(key: "decoder_proj") var decoderProj: Linear
    @ModuleInfo(key: "decoder_stages") var decoderStages: [ConvTransposed1d]
    @ModuleInfo(key: "audio_head") var audioHead: Conv1d

    public init(config: MimiCodecConfig = MimiCodecConfig(), seed: UInt64 = 575) {
        self.config = config
        let dim = config.latentDim
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            var encStages: [CausalConv1dStage] = []
            var inCh = 1
            for (idx, s) in config.strides.enumerated() {
                let outCh = idx == config.strides.count - 1 ? dim : max(8, dim / max(1, 1 << (config.strides.count - 1 - idx)))
                encStages.append(CausalConv1dStage(inChannels: inCh, outChannels: outCh, stride: s))
                inCh = outCh
            }
            let encProj = Linear(inCh, dim)
            let books = (0 ..< config.numCodebooks).map { _ in
                Embedding(embeddingCount: config.codebookSize, dimensions: dim)
            }
            let decProj = Linear(dim, dim)
            let reversed = Array(config.strides.reversed())
            var decStages: [ConvTransposed1d] = []
            var decIn = dim
            for (idx, s) in reversed.enumerated() {
                let decOut = idx == reversed.count - 1 ? max(8, dim / 4) : max(8, dim / max(1, 1 << (idx + 1)))
                decStages.append(ConvTransposed1d(inputChannels: decIn, outputChannels: decOut, kernelSize: s, stride: s))
                decIn = decOut
            }
            let head = Conv1d(inputChannels: decIn, outputChannels: 1, kernelSize: 3, padding: 0)
            return (encStages, encProj, books, decProj, decStages, head)
        }
        self._encoderStages.wrappedValue = built.0
        self._encoderProj.wrappedValue = built.1
        self._codebooks.wrappedValue = built.2
        self._decoderProj.wrappedValue = built.3
        self._decoderStages.wrappedValue = built.4
        self._audioHead.wrappedValue = built.5
        super.init()
        eval(self)
    }

    public func encodeTensor(_ waveform: MLXArray) -> MLXArray {
        var h = waveform
        for stage in encoderStages { h = stage(h) }
        var residual = encoderProj(h)
        var frameCodes: [MLXArray] = []
        frameCodes.reserveCapacity(codebooks.count)
        for book in codebooks {
            let (codes, quantized) = RVQQuantizerMath.quantize(residual, table: book)
            frameCodes.append(codes[0])
            residual = residual - quantized
        }
        return stacked(frameCodes, axis: -1)
    }

    public func encode(_ samples: [Float]) -> [[Int32]] {
        let input = RVQQuantizerMath.padWaveform(samples, multipleOf: config.frameSamples)
        let codes = encodeTensor(input)
        eval(codes)
        let frames = codes.dim(0)
        let q = codes.dim(1)
        let flat = codes.reshaped(-1).asArray(Int32.self)
        return (0 ..< frames).map { f in Array(flat[(f * q) ..< ((f + 1) * q)]) }
    }

    public func decodeTensor(_ codes: MLXArray) -> MLXArray {
        let frames = codes.dim(0)
        let qCount = min(codebooks.count, codes.dim(1))
        var latent = MLXArray.zeros([1, frames, config.latentDim], dtype: .float32)
        for q in 0 ..< qCount {
            let clamped = clip(codes[0..., q], min: 0, max: config.codebookSize - 1).reshaped(1, frames)
            latent = latent + codebooks[q](clamped) / Float(q + 1)
        }
        var h = MLXNN.elu(decoderProj(latent))
        for stage in decoderStages {
            h = MLXNN.elu(stage(h))
        }
        return RVQQuantizerMath.causalAudioHead(h, conv: audioHead)
    }

    public func decode(_ frames: [[Int32]], expectedSamples: Int? = nil) -> [Float] {
        guard !frames.isEmpty else { return [] }
        let q = config.numCodebooks
        var flat: [Int32] = []
        flat.reserveCapacity(frames.count * q)
        for row in frames {
            for idx in 0 ..< q {
                let raw = idx < row.count ? row[idx] : 0
                flat.append(min(Int32(config.codebookSize - 1), max(0, raw)))
            }
        }
        let tensor = MLXArray(flat).reshaped(frames.count, q)
        let wave = decodeTensor(tensor)
        eval(wave)
        let out = wave.asArray(Float.self)
        if let expectedSamples, expectedSamples >= 0, expectedSamples <= out.count {
            return Array(out.prefix(expectedSamples))
        }
        return out
    }
}

public struct SNACCodecConfig: Sendable, Codable, Equatable {
    public static let defaultStrides = [6, 5, 4, 2]

    public var sampleRate: Int
    public var baseStrides: [Int]
    public var finestStepsPerFrame: Int
    public var codebookSize: Int
    public var latentDim: Int

    public var baseHopSamples: Int {
        max(1, baseStrides.reduce(1, *))
    }

    public var frameSamples: Int {
        baseHopSamples * finestStepsPerFrame
    }

    public var frameStrideMs: Double {
        Double(frameSamples) / Double(max(1, sampleRate)) * 1_000.0
    }

    public static var tiny: SNACCodecConfig {
        SNACCodecConfig(sampleRate: 24_000, baseStrides: defaultStrides, codebookSize: 64, latentDim: 32)
    }

    public init(
        sampleRate: Int = SpeechRequest.defaultSampleRate,
        baseStrides: [Int] = Self.defaultStrides,
        codebookSize: Int = 256,
        latentDim: Int = 64
    ) {
        self.sampleRate = max(8_000, sampleRate)
        let cleanStrides = baseStrides.map { max(2, $0) }
        self.baseStrides = cleanStrides.isEmpty ? Self.defaultStrides : cleanStrides
        self.finestStepsPerFrame = 4
        self.codebookSize = max(8, codebookSize)
        self.latentDim = max(8, latentDim)
    }
}

public struct SNACMultiScalePacket: Sendable, Codable, Equatable {
    public static let tokensPerFrame = 7

    public let frameCount: Int
    public let scale0: [Int32]
    public let scale1: [Int32]
    public let scale2: [Int32]

    public init(frameCount: Int, scale0: [Int32], scale1: [Int32], scale2: [Int32]) {
        let count = max(0, frameCount)
        self.frameCount = count
        self.scale0 = Array(scale0.prefix(count)) + Array(repeating: 0, count: max(0, count - scale0.count))
        self.scale1 = Array(scale1.prefix(count * 2)) + Array(repeating: 0, count: max(0, count * 2 - scale1.count))
        self.scale2 = Array(scale2.prefix(count * 4)) + Array(repeating: 0, count: max(0, count * 4 - scale2.count))
    }

    public var flattenedFrames: [[Int32]] {
        (0 ..< frameCount).map { f in
            [
                scale0[f],
                scale1[2 * f], scale1[2 * f + 1],
                scale2[4 * f], scale2[4 * f + 1], scale2[4 * f + 2], scale2[4 * f + 3],
            ]
        }
    }

    public static func fromFlattened(_ frames: [[Int32]]) throws(BasaltError) -> SNACMultiScalePacket {
        var s0: [Int32] = [], s1: [Int32] = [], s2: [Int32] = []
        s0.reserveCapacity(frames.count)
        s1.reserveCapacity(frames.count * 2)
        s2.reserveCapacity(frames.count * 4)
        for row in frames {
            guard row.count == tokensPerFrame else {
                throw .invalidRequestStructure(details: "SNAC frame requires \(tokensPerFrame) multi-scale tokens, got \(row.count)")
            }
            s0.append(row[0])
            s1.append(contentsOf: row[1 ... 2])
            s2.append(contentsOf: row[3 ... 6])
        }
        return SNACMultiScalePacket(frameCount: frames.count, scale0: s0, scale1: s1, scale2: s2)
    }
}

public final class SNACMultiScaleCodec: Module {
    public let config: SNACCodecConfig
    @ModuleInfo(key: "encoder_stages") var encoderStages: [CausalConv1dStage]
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "scale_codebooks") var scaleCodebooks: [Embedding]
    @ModuleInfo(key: "decoder_proj") var decoderProj: Linear
    @ModuleInfo(key: "decoder_stages") var decoderStages: [ConvTransposed1d]
    @ModuleInfo(key: "audio_head") var audioHead: Conv1d

    public init(config: SNACCodecConfig = SNACCodecConfig(), seed: UInt64 = 576) {
        self.config = config
        let dim = config.latentDim
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            var encStages: [CausalConv1dStage] = []
            var inCh = 1
            for (idx, s) in config.baseStrides.enumerated() {
                let outCh = idx == config.baseStrides.count - 1 ? dim : max(8, dim / max(1, 1 << (config.baseStrides.count - 1 - idx)))
                encStages.append(CausalConv1dStage(inChannels: inCh, outChannels: outCh, stride: s))
                inCh = outCh
            }
            let encProj = Linear(inCh, dim)
            let books = (0 ..< 3).map { _ in Embedding(embeddingCount: config.codebookSize, dimensions: dim) }
            let decProj = Linear(dim, dim)
            let reversed = Array(config.baseStrides.reversed())
            var decStages: [ConvTransposed1d] = []
            var decIn = dim
            for (idx, s) in reversed.enumerated() {
                let decOut = idx == reversed.count - 1 ? max(8, dim / 4) : max(8, dim / max(1, 1 << (idx + 1)))
                decStages.append(ConvTransposed1d(inputChannels: decIn, outputChannels: decOut, kernelSize: s, stride: s))
                decIn = decOut
            }
            let head = Conv1d(inputChannels: decIn, outputChannels: 1, kernelSize: 3, padding: 0)
            return (encStages, encProj, books, decProj, decStages, head)
        }
        self._encoderStages.wrappedValue = built.0
        self._encoderProj.wrappedValue = built.1
        self._scaleCodebooks.wrappedValue = built.2
        self._decoderProj.wrappedValue = built.3
        self._decoderStages.wrappedValue = built.4
        self._audioHead.wrappedValue = built.5
        super.init()
        eval(self)
    }

    public func encode(_ samples: [Float]) -> SNACMultiScalePacket {
        let input = RVQQuantizerMath.padWaveform(samples, multipleOf: config.frameSamples)
        var h = input
        for stage in encoderStages { h = stage(h) }
        let z = encoderProj(h)
        let fineSteps = z.dim(1)
        let frames = max(1, fineSteps / 4)
        let dim = config.latentDim

        let pooled0 = z.reshaped(1, frames, 4, dim).mean(axis: 2)
        let (c0, q0) = RVQQuantizerMath.quantize(pooled0, table: scaleCodebooks[0])
        let r1 = z - Self.repeatTime(q0, factor: 4)

        let pooled1 = r1.reshaped(1, frames * 2, 2, dim).mean(axis: 2)
        let (c1, q1) = RVQQuantizerMath.quantize(pooled1, table: scaleCodebooks[1])
        let r2 = r1 - Self.repeatTime(q1, factor: 2)

        let (c2, _) = RVQQuantizerMath.quantize(r2, table: scaleCodebooks[2])
        eval(c0, c1, c2)
        return SNACMultiScalePacket(
            frameCount: frames,
            scale0: c0.reshaped(-1).asArray(Int32.self),
            scale1: c1.reshaped(-1).asArray(Int32.self),
            scale2: c2.reshaped(-1).asArray(Int32.self)
        )
    }

    public func decode(_ packet: SNACMultiScalePacket, expectedSamples: Int? = nil) -> [Float] {
        guard packet.frameCount > 0 else { return [] }
        let cap = Int32(config.codebookSize - 1)
        let idx0 = clip(MLXArray(packet.scale0), min: 0, max: cap).reshaped(1, packet.frameCount)
        let idx1 = clip(MLXArray(packet.scale1), min: 0, max: cap).reshaped(1, packet.frameCount * 2)
        let idx2 = clip(MLXArray(packet.scale2), min: 0, max: cap).reshaped(1, packet.frameCount * 4)

        let z0 = Self.repeatTime(scaleCodebooks[0](idx0), factor: 4)
        let z1 = Self.repeatTime(scaleCodebooks[1](idx1), factor: 2)
        let z2 = scaleCodebooks[2](idx2)
        var h = MLXNN.elu(decoderProj(z0 + z1 * 0.5 + z2 * 0.25))
        for stage in decoderStages {
            h = MLXNN.elu(stage(h))
        }
        let wave = RVQQuantizerMath.causalAudioHead(h, conv: audioHead)
        eval(wave)
        let out = wave.asArray(Float.self)
        if let expectedSamples, expectedSamples >= 0, expectedSamples <= out.count {
            return Array(out.prefix(expectedSamples))
        }
        return out
    }

    private static func repeatTime(_ x: MLXArray, factor: Int) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), d = x.dim(2)
        return stacked(Array(repeating: x, count: factor), axis: 2).reshaped(b, t * factor, d)
    }
}
