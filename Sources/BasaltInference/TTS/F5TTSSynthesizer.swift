import Foundation
import MLX
import MLXNN
import BasaltCore

final class F5TimeEmbed: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let freqDim: Int = 256

    override init() {
        self._fc1.wrappedValue = Linear(256, 1024)
        self._fc2.wrappedValue = Linear(1024, 1024)
        super.init()
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = freqDim / 2
        let scale = Float(log(10_000.0)) / Float(half - 1)
        let freqs = exp(MLXArray((0 ..< half).map { -Float($0) * scale }))
        let args = t.asType(.float32).expandedDimensions(axis: -1) * Float(1000.0) * freqs.reshaped(1, half)
        let emb = concatenated([sin(args), cos(args)], axis: -1)
        return fc2(silu(fc1(emb)))
    }
}

final class F5ConvNeXtV2Block: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "grn_gamma") var grnGamma: MLXArray
    @ParameterInfo(key: "grn_beta") var grnBeta: MLXArray

    init(dim: Int = 512, intermediateDim: Int = 1024) {
        self._dwconv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim)
        self._grnGamma.wrappedValue = MLXArray.zeros([1, 1, intermediateDim], dtype: .float32)
        self._grnBeta.wrappedValue = MLXArray.zeros([1, 1, intermediateDim], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = norm(dwconv(x))
        h = gelu(pwconv1(h))
        let gx = sqrt(h.square().sum(axis: 1, keepDims: true) + Float(1e-6))
        let nx = gx / (gx.mean(axis: -1, keepDims: true) + Float(1e-6))
        h = grnGamma * (h * nx) + grnBeta + h
        h = pwconv2(h)
        return residual + h
    }
}

final class F5TextEmbedding: Module {
    @ModuleInfo(key: "text_embed") var textEmbed: Embedding
    @ModuleInfo(key: "text_blocks") var textBlocks: [F5ConvNeXtV2Block]
    var maskPadding: Bool

    init(vocabSize: Int = 2545, textDim: Int = 512, convLayers: Int = 4, maskPadding: Bool = true) {
        self.maskPadding = maskPadding
        self._textEmbed.wrappedValue = Embedding(embeddingCount: vocabSize + 1, dimensions: textDim)
        self._textBlocks.wrappedValue = (0 ..< convLayers).map { _ in
            F5ConvNeXtV2Block(dim: textDim, intermediateDim: textDim * 2)
        }
        super.init()
    }

    func callAsFunction(_ textIDs: MLXArray, seqLen: Int, dropText: Bool = false) -> MLXArray {
        var ids = textIDs + Int32(1)
        let curLen = ids.dim(1)
        if curLen < seqLen {
            let pad = MLXArray.zeros([ids.dim(0), seqLen - curLen], dtype: .int32)
            ids = concatenated([ids, pad], axis: 1)
        } else if curLen > seqLen {
            ids = ids[0..., ..<seqLen]
        }
        let padMask = (ids .== Int32(0)).expandedDimensions(axis: -1)
        if dropText {
            ids = MLXArray.zeros(like: ids)
        }
        var x = textEmbed(ids)
        let half = x.dim(-1) / 2
        let scale = Float(log(10_000.0)) / Float(half)
        let invFreq = exp(MLXArray((0 ..< half).map { -Float($0) * scale }))
        let pos = MLXArray((0 ..< seqLen).map(Float.init)).expandedDimensions(axis: 1)
        let angles = pos * invFreq.reshaped(1, half)
        let posEmb = concatenated([cos(angles), sin(angles)], axis: -1).expandedDimensions(axis: 0)
        x = x + posEmb
        if maskPadding {
            x = MLX.where(padMask, MLXArray.zeros(like: x), x)
        }
        for block in textBlocks {
            x = block(x)
            if maskPadding {
                x = MLX.where(padMask, MLXArray.zeros(like: x), x)
            }
        }
        return x
    }
}

final class F5InputEmbedding: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "conv0") var conv0: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(melDim: Int = 100, textDim: Int = 512, outDim: Int = 1024) {
        self._proj.wrappedValue = Linear(melDim * 2 + textDim, outDim)
        self._conv0.wrappedValue = Conv1d(inputChannels: outDim, outputChannels: outDim, kernelSize: 31, padding: 15, groups: 16)
        self._conv2.wrappedValue = Conv1d(inputChannels: outDim, outputChannels: outDim, kernelSize: 31, padding: 15, groups: 16)
        super.init()
    }

    func callAsFunction(x: MLXArray, cond: MLXArray, textEmbed: MLXArray) -> MLXArray {
        let h = proj(concatenated([x, cond, textEmbed], axis: -1))
        let pos = Self.mish(conv2(Self.mish(conv0(h))))
        return h + pos
    }

    private static func mish(_ x: MLXArray) -> MLXArray {
        x * tanh(softplus(x))
    }
}

final class F5DiTBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: Linear
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "ff_in") var ffIn: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear
    let dim: Int
    let heads: Int
    let headDim: Int

    init(dim: Int = 1024, heads: Int = 16, ffMult: Int = 2) {
        self.dim = dim
        self.heads = heads
        self.headDim = dim / heads
        self._attnNorm.wrappedValue = Linear(dim, dim * 6)
        self._toQ.wrappedValue = Linear(dim, dim)
        self._toK.wrappedValue = Linear(dim, dim)
        self._toV.wrappedValue = Linear(dim, dim)
        self._toOut.wrappedValue = Linear(dim, dim)
        self._ffIn.wrappedValue = Linear(dim, dim * ffMult)
        self._ffOut.wrappedValue = Linear(dim * ffMult, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tEmb: MLXArray, cosRot: MLXArray, sinRot: MLXArray) -> MLXArray {
        let mod = attnNorm(silu(tEmb)).expandedDimensions(axis: 1)
        let chunks = split(mod, parts: 6, axis: -1)
        let (shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP) = (
            chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5]
        )
        let norm1 = Self.layerNormNoAffine(x) * (1.0 + scaleMSA) + shiftMSA
        let attnOut = attention(norm1, cosRot: cosRot, sinRot: sinRot)
        let xMid = x + gateMSA * attnOut
        let norm2 = Self.layerNormNoAffine(xMid) * (1.0 + scaleMLP) + shiftMLP
        let ffOutVal = ffOut(geluApproximate(ffIn(norm2)))
        return xMid + gateMLP * ffOutVal
    }

    private func attention(_ x: MLXArray, cosRot: MLXArray, sinRot: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        var q = toQ(x).reshaped(batch, steps, heads, headDim).transposed(0, 2, 1, 3)
        var k = toK(x).reshaped(batch, steps, heads, headDim).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped(batch, steps, heads, headDim).transposed(0, 2, 1, 3)
        q = Self.applyRoPE(q, cosRot: cosRot, sinRot: sinRot)
        k = Self.applyRoPE(k, cosRot: cosRot, sinRot: sinRot)
        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        return toOut(attn.transposed(0, 2, 1, 3).reshaped(batch, steps, dim))
    }

    private static func layerNormNoAffine(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return (x - mean) * rsqrt(variance + Float(1e-6))
    }

    private static func applyRoPE(_ x: MLXArray, cosRot: MLXArray, sinRot: MLXArray) -> MLXArray {
        let x1 = x[0..., 0..., 0..., .stride(by: 2)]
        let x2 = x[0..., 0..., 0..., .stride(from: 1, by: 2)]
        let rot = stacked([-x2, x1], axis: -1).reshaped(x.shape)
        return x * cosRot + rot * sinRot
    }
}

package final class F5DiTModel: Module {
    @ModuleInfo(key: "time_embed") var timeEmbed: F5TimeEmbed
    @ModuleInfo(key: "text_embed") var textEmbed: F5TextEmbedding
    @ModuleInfo(key: "input_embed") var inputEmbed: F5InputEmbedding
    @ParameterInfo(key: "rotary_inv_freq") var rotaryInvFreq: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [F5DiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    package init(vocabSize: Int = 2545, dim: Int = 1024, depth: Int = 22, heads: Int = 16, melDim: Int = 100) {
        self._timeEmbed.wrappedValue = F5TimeEmbed()
        self._textEmbed.wrappedValue = F5TextEmbedding(vocabSize: vocabSize)
        self._inputEmbed.wrappedValue = F5InputEmbedding(melDim: melDim, textDim: 512, outDim: dim)
        self._rotaryInvFreq.wrappedValue = MLXArray.zeros([dim / heads / 2], dtype: .float32)
        self._blocks.wrappedValue = (0 ..< depth).map { _ in F5DiTBlock(dim: dim, heads: heads) }
        self._normOut.wrappedValue = Linear(dim, dim * 2)
        self._projOut.wrappedValue = Linear(dim, melDim)
        super.init()
    }

    package func callAsFunction(
        x: MLXArray,
        cond: MLXArray,
        textIDs: MLXArray,
        time: MLXArray,
        dropAudioCond: Bool = false,
        dropText: Bool = false
    ) -> MLXArray {
        let seqLen = x.dim(1)
        let tEmb = timeEmbed(time)
        let txtEmb = textEmbed(textIDs, seqLen: seqLen, dropText: dropText)
        let condIn = dropAudioCond ? MLXArray.zeros(like: cond) : cond
        var h = inputEmbed(x: x, cond: condIn, textEmbed: txtEmb)
        let (cosRot, sinRot) = ropeAngles(seqLen: seqLen)
        for block in blocks {
            h = block(h, tEmb: tEmb, cosRot: cosRot, sinRot: sinRot)
        }
        let finalMod = normOut(silu(tEmb)).expandedDimensions(axis: 1)
        let parts = split(finalMod, parts: 2, axis: -1)
        let mean = h.mean(axis: -1, keepDims: true)
        let variance = h.variance(axis: -1, keepDims: true)
        let normed = (h - mean) * rsqrt(variance + Float(1e-6))
        h = normed * (1.0 + parts[0]) + parts[1]
        return projOut(h)
    }

    private func ropeAngles(seqLen: Int) -> (cosRot: MLXArray, sinRot: MLXArray) {
        let positions = MLXArray((0 ..< seqLen).map(Float.init)).expandedDimensions(axis: 1)
        let freqs = positions * rotaryInvFreq.reshaped(1, rotaryInvFreq.dim(0))
        let repeatedFreqs = stacked([freqs, freqs], axis: -1).reshaped(1, 1, seqLen, rotaryInvFreq.dim(0) * 2)
        return (cos(repeatedFreqs), sin(repeatedFreqs))
    }
}

package enum F5TTSWeightLoader {
    package struct StrictLoadReport: Sendable, Equatable {
        package let checkpointKeyCount: Int
        package let modelParameterCount: Int
    }

    package static func loadStrict(
        from directory: URL,
        into model: F5DiTModel
    ) throws(BasaltError) -> StrictLoadReport {
        let (fileURL, isV1) = try resolveWeightsFile(in: directory)
        model.textEmbed.maskPadding = isV1
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: fileURL)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "f5-tts: failed to read safetensors: \(error.localizedDescription)")
        }
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let updates = try mapAndValidate(raw: raw, expected: expected, path: fileURL.path)
        do {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "f5-tts: strict update failed: \(error.localizedDescription)")
        }
        eval(model)
        return StrictLoadReport(checkpointKeyCount: updates.count, modelParameterCount: expected.count)
    }

    private static func resolveWeightsFile(in directory: URL) throws(BasaltError) -> (url: URL, isV1: Bool) {
        let preferredV1 = ["model_v1.safetensors", "model_last_inference.safetensors", "model_last.safetensors"]
        for name in preferredV1 {
            let candidate = directory.appending(component: name)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return (candidate, true)
            }
        }
        let v0 = directory.appending(component: "model.safetensors")
        if FileManager.default.fileExists(atPath: v0.path) {
            return (v0, false)
        }
        throw .weightsCorrupted(path: directory.path, reason: "f5-tts: missing F5-TTS safetensors checkpoint")
    }

    private static func mapAndValidate(
        raw: [String: MLXArray],
        expected: [String: MLXArray],
        path: String
    ) throws(BasaltError) -> [(String, MLXArray)] {
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(expected.count)
        var seen = Set<String>()
        for (rawKey, tensor) in raw where !isMetadataKey(rawKey) {
            let targetKey = remapKey(rawKey)
            guard let target = expected[targetKey] else {
                throw .weightsCorrupted(path: path, reason: "f5-tts: unexpected tensor '\(rawKey)' -> '\(targetKey)'")
            }
            let f32 = alignedTensor(tensor, targetShape: target.shape, key: targetKey)
            guard f32.shape == target.shape else {
                throw .weightsCorrupted(path: path, reason: "f5-tts: shape mismatch for '\(targetKey)': expected \(target.shape), got \(f32.shape)")
            }
            seen.insert(targetKey)
            updates.append((targetKey, f32))
        }
        let missing = Set(expected.keys).subtracting(seen)
        guard missing.isEmpty && updates.count == expected.count else {
            throw .weightsCorrupted(path: path, reason: "f5-tts: missing \(missing.count) tensors: \(missing.sorted().prefix(5).joined(separator: ", "))")
        }
        return updates
    }

    private static func isMetadataKey(_ key: String) -> Bool {
        key == "initted" || key == "step" || key.contains("mel_spec.")
    }

    private static func alignedTensor(_ tensor: MLXArray, targetShape: [Int], key: String) -> MLXArray {
        let f32 = tensor.asType(.float32)
        if f32.ndim == 3 && f32.shape != targetShape && key.hasSuffix(".weight") {
            return f32.transposed(0, 2, 1)
        }
        return f32
    }

    package static func loadVocab(from directory: URL) throws(BasaltError) -> [Character: Int32] {
        let vocabURL = directory.appending(component: "vocab.txt")
        guard let rawData = try? Data(contentsOf: vocabURL) else {
            throw .weightsCorrupted(path: vocabURL.path, reason: "f5-tts: missing vocab.txt")
        }
        let lfByte = UInt8(ascii: "\n")
        let crByte = UInt8(ascii: "\r")
        var map: [Character: Int32] = [:]
        for (idx, slice) in rawData.split(separator: lfByte, omittingEmptySubsequences: false).enumerated() {
            let trimmed = slice.last == crByte ? slice.dropLast() : slice
            let line = String(decoding: trimmed, as: UTF8.self)
            if line.count == 1, let ch = line.first, map[ch] == nil {
                map[ch] = Int32(idx)
            }
        }
        return map
    }

    private static func remapKey(_ key: String) -> String {
        var trimmed = key
        if trimmed.hasPrefix("ema_model.") { trimmed = String(trimmed.dropFirst("ema_model.".count)) }
        if trimmed.hasPrefix("transformer.") { trimmed = String(trimmed.dropFirst("transformer.".count)) }
        if trimmed == "rotary_embed.inv_freq" { return "rotary_inv_freq" }
        if trimmed.hasPrefix("norm_out.linear.") { return "norm_out." + trimmed.dropFirst("norm_out.linear.".count) }
        if trimmed.hasPrefix("time_embed.time_mlp.layers.0.") { return "time_embed.fc1." + trimmed.dropFirst("time_embed.time_mlp.layers.0.".count) }
        if trimmed.hasPrefix("time_embed.time_mlp.layers.2.") { return "time_embed.fc2." + trimmed.dropFirst("time_embed.time_mlp.layers.2.".count) }
        if trimmed.hasPrefix("time_embed.time_mlp.0.") { return "time_embed.fc1." + trimmed.dropFirst("time_embed.time_mlp.0.".count) }
        if trimmed.hasPrefix("time_embed.time_mlp.2.") { return "time_embed.fc2." + trimmed.dropFirst("time_embed.time_mlp.2.".count) }
        if trimmed.hasPrefix("input_embed.conv_pos_embed.conv1d.layers.0.") { return "input_embed.conv0." + trimmed.dropFirst("input_embed.conv_pos_embed.conv1d.layers.0.".count) }
        if trimmed.hasPrefix("input_embed.conv_pos_embed.conv1d.layers.2.") { return "input_embed.conv2." + trimmed.dropFirst("input_embed.conv_pos_embed.conv1d.layers.2.".count) }
        if trimmed.hasPrefix("input_embed.conv_pos_embed.conv1d.0.") { return "input_embed.conv0." + trimmed.dropFirst("input_embed.conv_pos_embed.conv1d.0.".count) }
        if trimmed.hasPrefix("input_embed.conv_pos_embed.conv1d.2.") { return "input_embed.conv2." + trimmed.dropFirst("input_embed.conv_pos_embed.conv1d.2.".count) }
        if trimmed.hasPrefix("text_embed.text_blocks.") {
            let stripped = trimmed
                .replacingOccurrences(of: "text_embed.text_blocks.layers.", with: "text_embed.text_blocks.")
                .replacingOccurrences(of: ".grn.gamma", with: ".grn_gamma")
                .replacingOccurrences(of: ".grn.beta", with: ".grn_beta")
            return stripped
        }
        if trimmed.hasPrefix("transformer_blocks.") {
            let suffix = trimmed.dropFirst("transformer_blocks.".count)
            return "blocks." + suffix
                .replacingOccurrences(of: ".attn_norm.linear.", with: ".attn_norm.")
                .replacingOccurrences(of: ".attn.to_out.layers.0.", with: ".to_out.")
                .replacingOccurrences(of: ".attn.to_out.0.", with: ".to_out.")
                .replacingOccurrences(of: ".attn.", with: ".")
                .replacingOccurrences(of: ".ff.ff.layers.0.layers.0.", with: ".ff_in.")
                .replacingOccurrences(of: ".ff.ff.layers.2.", with: ".ff_out.")
                .replacingOccurrences(of: ".ff.ff.0.0.", with: ".ff_in.")
                .replacingOccurrences(of: ".ff.ff.2.", with: ".ff_out.")
        }
        return trimmed
    }
}

package struct F5SynthesisOptions: Sendable, Equatable {
    package var steps: Int
    package var cfgStrength: Float
    package var swayCoef: Float
    package var seed: UInt64

    package static let `default` = F5SynthesisOptions(steps: 8, cfgStrength: 2.0, swayCoef: -1.0, seed: 42)

    package init(steps: Int = 8, cfgStrength: Float = 2.0, swayCoef: Float = -1.0, seed: UInt64 = 42) {
        self.steps = steps
        self.cfgStrength = cfgStrength
        self.swayCoef = swayCoef
        self.seed = seed
    }
}

package actor F5TTSSynthesizer {
    package nonisolated let strictLoadReport: F5TTSWeightLoader.StrictLoadReport
    private let dit: F5DiTModel
    private let vocos: VocosModel
    private let vocab: [Character: Int32]

    package init(modelDirectory: URL, vocosDirectory: URL) throws(BasaltError) {
        let vocabMap = try F5TTSWeightLoader.loadVocab(from: modelDirectory)
        let model = F5DiTModel()
        let report = try F5TTSWeightLoader.loadStrict(from: modelDirectory, into: model)
        let vocoder = try VocosModel.loadStrict(from: vocosDirectory)
        self.vocab = vocabMap
        self.dit = model
        self.vocos = vocoder
        self.strictLoadReport = report
    }

    package func synthesize(
        text: String,
        referenceAudio: [Float],
        referenceText: String,
        options: F5SynthesisOptions = .default
    ) -> [Float] {
        var refScaled = referenceAudio
        let refRMS = TTSWaveformMetrics.rms(refScaled)
        if refRMS > 1e-6 && refRMS < 0.1 {
            let gain = Float(0.1 / refRMS)
            refScaled = refScaled.map { $0 * gain }
        }
        let refMel = vocos.logMelSpectrogram(audio: MLXArray(refScaled)).expandedDimensions(axis: 0)
        let refFrames = refMel.dim(1)
        let fullText = referenceText.trimmingCharacters(in: .whitespacesAndNewlines) + " " + text.trimmingCharacters(in: .whitespacesAndNewlines)
        let refChars = max(1, referenceText.utf8.count)
        let genChars = max(1, text.utf8.count)
        let genFrames = max(32, Int((Double(refFrames) * Double(genChars) / Double(refChars)).rounded()))
        let totalFrames = refFrames + genFrames

        let tokenIDs = fullText.map { vocab[$0] ?? 0 }
        let idsArray = MLXArray(tokenIDs).reshaped(1, tokenIDs.count)

        let condPad = MLXArray.zeros([1, genFrames, 100], dtype: .float32)
        let condMel = concatenated([refMel, condPad], axis: 1)

        var y = withRandomState(MLXRandom.RandomState(seed: options.seed)) {
            MLXRandom.normal([1, totalFrames, 100], dtype: .float32)
        }
        let stepCount = max(4, options.steps)
        let times: [Float] = (0 ... stepCount).map { i in
            let u = Float(i) / Float(stepCount)
            return u + options.swayCoef * (cos(Float.pi * 0.5 * u) - 1.0 + u)
        }
        func velocity(_ tVal: Float, _ state: MLXArray) -> MLXArray {
            let tArr = MLXArray([tVal])
            let predCond = dit(x: state, cond: condMel, textIDs: idsArray, time: tArr, dropAudioCond: false, dropText: false)
            let predUncond = dit(x: state, cond: condMel, textIDs: idsArray, time: tArr, dropAudioCond: true, dropText: true)
            return predCond + options.cfgStrength * (predCond - predUncond)
        }
        for i in 0 ..< stepCount {
            let t0 = times[i]
            let dt = times[i + 1] - t0
            let k1 = velocity(t0, y)
            let k2 = velocity(t0 + 0.5 * dt, y + 0.5 * dt * k1)
            let k3 = velocity(t0 + 0.5 * dt, y + 0.5 * dt * k2)
            let k4 = velocity(t0 + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            eval(y)
        }
        let genMel = y[0..., refFrames..., 0...]
        let wave = vocos.decode(mel: genMel)
        eval(wave)
        return wave.asType(.float32).asArray(Float.self)
    }
}
