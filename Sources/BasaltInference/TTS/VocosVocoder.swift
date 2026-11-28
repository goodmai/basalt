import Foundation
import MLX
import MLXNN
import BasaltCore

final class VocosConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int = 512, intermediateDim: Int = 1536) {
        self._dwconv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim)
        self._gamma.wrappedValue = MLXArray.ones([dim], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = dwconv(x)
        h = norm(h)
        h = gelu(pwconv1(h))
        h = pwconv2(h) * gamma
        return residual + h
    }
}

final class VocosBackbone: Module {
    @ModuleInfo(key: "embed") var embed: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "convnext") var convnext: [VocosConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(inputChannels: Int = 100, dim: Int = 512, intermediateDim: Int = 1536, numLayers: Int = 8) {
        self._embed.wrappedValue = Conv1d(inputChannels: inputChannels, outputChannels: dim, kernelSize: 7, padding: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._convnext.wrappedValue = (0 ..< numLayers).map { _ in
            VocosConvNeXtBlock(dim: dim, intermediateDim: intermediateDim)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = norm(embed(mel))
        for block in convnext {
            x = block(x)
        }
        return finalLayerNorm(x)
    }
}

final class VocosISTFTHead: Module {
    @ModuleInfo(key: "out") var out: Linear
    let nFFT: Int
    let hopLength: Int

    init(dim: Int = 512, nFFT: Int = 1024, hopLength: Int = 256) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self._out.wrappedValue = Linear(dim, nFFT + 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = out(x)[0]
        let bins = nFFT / 2 + 1
        let mag = clip(exp(projected[0..., ..<bins]), max: Float(1e2))
        let phase = projected[0..., bins...]
        let rawWave = ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: mag,
            phase: phase,
            nFFT: nFFT,
            hopLength: hopLength
        )
        let pad = nFFT / 2
        if rawWave.dim(0) > 2 * pad {
            return rawWave[pad ..< (rawWave.dim(0) - pad)]
        }
        return rawWave
    }
}

package final class VocosModel: Module {
    @ModuleInfo(key: "backbone") var backbone: VocosBackbone
    @ModuleInfo(key: "head") var head: VocosISTFTHead
    @ParameterInfo(key: "mel_fb") var melFilterbank: MLXArray

    package init(nMels: Int = 100, dim: Int = 512, intermediateDim: Int = 1536, numLayers: Int = 8, nFFT: Int = 1024, hopLength: Int = 256) {
        self._backbone.wrappedValue = VocosBackbone(inputChannels: nMels, dim: dim, intermediateDim: intermediateDim, numLayers: numLayers)
        self._head.wrappedValue = VocosISTFTHead(dim: dim, nFFT: nFFT, hopLength: hopLength)
        self._melFilterbank.wrappedValue = MLXArray.zeros([nFFT / 2 + 1, nMels], dtype: .float32)
        super.init()
    }

    package func decode(mel: MLXArray) -> MLXArray {
        head(backbone(mel))
    }

    package func logMelSpectrogram(audio: MLXArray) -> MLXArray {
        let wave = audio.asType(.float32)
        let nFFT = head.nFFT
        let hop = head.hopLength
        let pad = nFFT / 2
        let padded = MLX.padded(wave, widths: [IntOrPair((pad, pad))], mode: .edge)
        let frames = max(1, (padded.dim(0) - nFFT) / hop + 1)
        let starts = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(hop)
        let offsets = MLXArray(Int32(0) ..< Int32(nFFT)).expandedDimensions(axis: 0)
        let framed = padded[(starts + offsets).reshaped(-1)].reshaped(frames, nFFT)
        let window = MLXArray(ISTFTNetSynthesizer.hannWindow(length: nFFT)).reshaped(1, nFFT)
        let spec = MLXFFT.rfft(framed * window, n: nFFT, axis: -1)
        let mag = sqrt(spec.realPart().square() + spec.imaginaryPart().square() + Float(1e-9))
        let mel = matmul(mag, melFilterbank)
        return log(maximum(mel, MLXArray(Float(1e-5))))
    }

    package static func loadStrict(from directory: URL) throws(BasaltError) -> VocosModel {
        let fileURL = directory.appending(component: "model.safetensors")
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw .weightsCorrupted(path: fileURL.path, reason: "vocos: missing model.safetensors")
        }
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: fileURL)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "vocos: failed to load safetensors: \(error.localizedDescription)")
        }
        let model = VocosModel()
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(expected.count)
        let skipWindowKeys: Set<String> = ["feature_extractor.mel_spec.spectrogram.window", "head.istft.window"]
        var consumed = skipWindowKeys.intersection(raw.keys)

        for (rawKey, tensor) in raw {
            if skipWindowKeys.contains(rawKey) { continue }
            let targetKey = rawKey == "feature_extractor.mel_spec.mel_scale.fb" ? "mel_fb" : rawKey
            guard let target = expected[targetKey] else {
                throw .weightsCorrupted(path: fileURL.path, reason: "vocos: unexpected tensor '\(rawKey)'")
            }
            let converted: MLXArray
            if rawKey.hasSuffix(".weight") && tensor.ndim == 3 && tensor.shape != target.shape {
                converted = tensor.asType(.float32).transposed(0, 2, 1)
            } else {
                converted = tensor.asType(.float32)
            }
            guard converted.shape == target.shape else {
                throw .weightsCorrupted(path: fileURL.path, reason: "vocos: shape mismatch for '\(targetKey)': expected \(target.shape), got \(converted.shape)")
            }
            consumed.insert(rawKey)
            updates.append((targetKey, converted))
        }
        guard updates.count == expected.count, consumed.count == raw.count else {
            throw .weightsCorrupted(path: fileURL.path, reason: "vocos: parameter count mismatch (loaded \(updates.count)/\(expected.count))")
        }
        do {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "vocos: strict update failed: \(error.localizedDescription)")
        }
        eval(model)
        return model
    }
}
