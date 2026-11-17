import Foundation
import MLX
import MLXNN
import BasaltCore

final class Zonos3ConvNeXt: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear

    init(width: Int, hidden: Int) {
        self._dwconv.wrappedValue = Conv1d(inputChannels: width, outputChannels: width, kernelSize: 7, groups: width, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(width, hidden, bias: true)
        self._pwconv2.wrappedValue = Linear(hidden, width, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padLeft = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((6, 0)), IntOrPair((0, 0))])
        let h = norm(dwconv(padLeft))
        return x + pwconv2(MLXNN.gelu(pwconv1(h)))
    }
}

final class Zonos3MossLayer: Module {
    let heads: Int
    let headDim: Int
    let context: Int
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "attention_out") var attentionOut: Linear
    @ModuleInfo(key: "ffn_in") var ffnIn: Linear
    @ModuleInfo(key: "ffn_out") var ffnOut: Linear
    @ParameterInfo(key: "attention_scale") var attentionScale: MLXArray
    @ParameterInfo(key: "ffn_scale") var ffnScale: MLXArray

    init(_ s: Zonos3MiniConfig.MossStageSpec) {
        let d = s.dModel
        self.heads = s.numHeads
        self.headDim = max(2, d / max(1, s.numHeads))
        self.context = max(1, s.contextFrames)
        self._norm1.wrappedValue = LayerNorm(dimensions: d, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: d, eps: 1e-5)
        self._qkv.wrappedValue = Linear(d, 3 * d, bias: false)
        self._attentionOut.wrappedValue = Linear(d, d, bias: false)
        self._ffnIn.wrappedValue = Linear(d, s.dimFeedforward, bias: false)
        self._ffnOut.wrappedValue = Linear(s.dimFeedforward, d, bias: false)
        self._attentionScale.wrappedValue = MLXArray.ones([d], dtype: .float32) * 0.1
        self._ffnScale.wrappedValue = MLXArray.ones([d], dtype: .float32) * 0.1
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h1 = x + attentionScale.asType(x.dtype) * slidingAttention(norm1(x))
        let ffn = ffnOut(MLXNN.gelu(ffnIn(norm2(h1))))
        return h1 + ffnScale.asType(h1.dtype) * ffn
    }

    private func slidingAttention(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), d = x.dim(2)
        let projected = qkv(x).reshaped(b, t, 3, heads, headDim)
        let positions = MLXArray((0 ..< t).map(Float.init))
        let q = Self.ropeBatch(projected[0..., 0..., 0, 0..., 0...], positions: positions)
            .transposed(0, 2, 1, 3)
        let k = Self.ropeBatch(projected[0..., 0..., 1, 0..., 0...], positions: positions)
            .transposed(0, 2, 1, 3)
        let v = projected[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let scale = 1.0 / Float(headDim).squareRoot()
        var blocks: [MLXArray] = []
        for start in stride(from: 0, to: t, by: 512) {
            let end = min(t, start + 512)
            let left = max(0, start - context + 1)
            let mask = Self.windowMask(qStart: start, qEnd: end, kLeft: left, context: context, dtype: q.dtype)
            let out = MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., start ..< end, 0...],
                keys: k[0..., 0..., left ..< end, 0...],
                values: v[0..., 0..., left ..< end, 0...],
                scale: scale,
                mask: .array(mask)
            )
            blocks.append(out)
        }
        let merged = concatenated(blocks, axis: 2).transposed(0, 2, 1, 3).reshaped(b, t, d)
        return attentionOut(merged)
    }

    private static func ropeBatch(_ z: MLXArray, positions: MLXArray) -> MLXArray {
        let b = z.dim(0), t = z.dim(1), h = z.dim(2), hd = z.dim(3)
        let half = hd / 2
        let freqs = (0 ..< half).map { idx in Float(exp(-Double(idx) * log(10_000.0) * 2.0 / Double(hd))) }
        let angles = matmul(positions.reshaped(t, 1), MLXArray(freqs).reshaped(1, half))
        let cosA = cos(angles).reshaped(1, t, 1, half)
        let sinA = sin(angles).reshaped(1, t, 1, half)
        let zf = z.asType(.float32).reshaped(b, t, h, half, 2)
        let r = zf[0..., 0..., 0..., 0..., 0]
        let im = zf[0..., 0..., 0..., 0..., 1]
        return stacked([r * cosA - im * sinA, r * sinA + im * cosA], axis: -1).reshaped(b, t, h, hd).asType(z.dtype)
    }

    private static func windowMask(qStart: Int, qEnd: Int, kLeft: Int, context: Int, dtype: DType) -> MLXArray {
        let qLen = qEnd - qStart
        let kLen = qEnd - kLeft
        var values = [Float]()
        values.reserveCapacity(qLen * kLen)
        for qPos in qStart ..< qEnd {
            for kPos in kLeft ..< qEnd {
                let delta = qPos - kPos
                values.append((delta >= 0 && delta < context) ? 0.0 : -1e9)
            }
        }
        return MLXArray(values).reshaped(1, 1, qLen, kLen).asType(dtype)
    }
}

final class Zonos3MossStage: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "layers") var layers: [Zonos3MossLayer]
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

    init(_ s: Zonos3MiniConfig.MossStageSpec) {
        self._inputProj.wrappedValue = Linear(s.inputDimension, s.dModel, bias: false)
        self._layers.wrappedValue = (0 ..< s.numLayers).map { _ in Zonos3MossLayer(s) }
        if let out = s.outputDimension {
            self._outputProj.wrappedValue = Linear(s.dModel, out, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputProj(x)
        for layer in layers { h = layer(h) }
        if let outputProj { h = outputProj(h) }
        return h
    }
}

final class Zonos3Quantizer: Module {
    @ParameterInfo(key: "codebook") var codebook: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(size: Int, dim: Int, out: Int) {
        self._codebook.wrappedValue = MLXRandom.normal([size, dim], dtype: .float32) * 0.05
        self._outProj.wrappedValue = Linear(dim, out, bias: true)
        super.init()
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        outProj(codebook[codes])
    }
}

final class Zonos3ISTFTHead: Module {
    let nFFTs: [Int]
    let hop: Int
    @ModuleInfo(key: "blocks") var blocks: [Zonos3ConvNeXt]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "outs") var outs: [Linear]
    @ModuleInfo(key: "mixer") var mixer: Conv1d

    init(_ c: Zonos3MiniConfig.Codec, dim: Int) {
        self.nFFTs = c.nFFTs
        self.hop = c.headHop
        self._blocks.wrappedValue = (0 ..< c.headDepth).map { _ in Zonos3ConvNeXt(width: dim, hidden: c.headHidden) }
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._outs.wrappedValue = c.nFFTs.map { Linear(dim, $0 + 2, bias: true) }
        let k = c.firKernel
        self._mixer.wrappedValue = Conv1d(inputChannels: c.nFFTs.count, outputChannels: 1, kernelSize: k, padding: (k - 1) / 2, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.asType(.float32)
        for block in blocks { h = block(h) }
        h = norm(h)
        let frames = h.dim(1)
        let logCap = Float(log(100.0))
        var branches: [MLXArray] = []
        for (n, lin) in zip(nFFTs, outs) {
            let specRaw = lin(h).squeezed(axis: 0).asType(.float32)
            let half = (n + 2) / 2
            let logMag = minimum(specRaw[0..., ..<half], MLXArray(logCap))
            let phase = specRaw[0..., half...]
            let mag = exp(logMag)
            let wave = Zonos3CodecMath.mlxCenterInverseSTFT(magnitude: mag, phase: phase, nFFT: n, hopLength: hop, targetLength: frames * hop)
            branches.append(wave)
        }
        let stackedBranches = stacked(branches, axis: -1).expandedDimensions(axis: 0)
        return mixer(stackedBranches)
    }
}

final class Zonos3MiniCodecModule: Module {
    let config: Zonos3MiniConfig.Codec
    @ModuleInfo(key: "quantizers") var quantizers: [Zonos3Quantizer]
    @ModuleInfo(key: "semantic_blocks") var semanticBlocks: [Zonos3ConvNeXt]
    @ModuleInfo(key: "semantic_proj") var semanticProj: Linear
    @ModuleInfo(key: "moss_stages") var mossStages: [Zonos3MossStage]
    @ModuleInfo(key: "head") var head: Zonos3ISTFTHead
    @ParameterInfo(key: "to_external_kernel") var toExternalKernel: MLXArray

    init(_ c: Zonos3MiniConfig.Codec) {
        self.config = c
        var qs = [Zonos3Quantizer(size: c.codebookSize, dim: c.codebookDim, out: c.semanticWidth)]
        qs += (0 ..< c.residualCodebooks).map { _ in
            Zonos3Quantizer(size: c.codebookSize, dim: c.codebookDim, out: c.latentDim)
        }
        self._quantizers.wrappedValue = qs
        self._semanticBlocks.wrappedValue = (0 ..< c.semanticDepth).map { _ in
            Zonos3ConvNeXt(width: c.semanticWidth, hidden: c.semanticHidden)
        }
        self._semanticProj.wrappedValue = Linear(c.semanticWidth, c.latentDim, bias: true)
        self._mossStages.wrappedValue = c.stages.map { Zonos3MossStage($0) }
        let lastDim = c.stages.last?.dModel ?? 768
        self._head.wrappedValue = Zonos3ISTFTHead(c, dim: lastDim)
        self._toExternalKernel.wrappedValue = Zonos3CodecMath.makeSincResampleKernel(
            orig: c.resampleOrig, new: c.resampleNew, width: c.resampleWidth
        )
        super.init()
    }

    func decode(codes: MLXArray) -> MLXArray {
        let frames = codes.dim(0)
        let batchCodes = codes.transposed(1, 0).expandedDimensions(axis: 0)
        var sem = quantizers[0](batchCodes[0..., 0, 0...]).asType(.float32)
        for block in semanticBlocks { sem = block(sem) }
        var z = semanticProj(sem).asType(.float32)
        for i in 0 ..< config.residualCodebooks {
            z = z + quantizers[1 + i](batchCodes[0..., 1 + i, 0...]).asType(.float32)
        }
        var x = z
        for (idx, stage) in mossStages.enumerated() {
            if idx > 0 { x = Self.unpatch2x(x) }
            x = stage(x)
        }
        let internalAudio = head(x).squeezed(axis: -1)
        let resampled = resampleToExternal(internalAudio)
        let expected = min(resampled.dim(1), frames * config.hop)
        return resampled[0, ..<expected]
    }

    static func unpatch2x(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), c = x.dim(2)
        return x.reshaped(b, t, c / 2, 2).transposed(0, 1, 3, 2).reshaped(b, t * 2, c / 2)
    }

    func resampleToExternal(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let orig = config.resampleOrig, width = config.resampleWidth
        let paddedIn = padded(
            x.expandedDimensions(axis: -1),
            widths: [IntOrPair((0, 0)), IntOrPair((width, width + orig)), IntOrPair((0, 0))]
        )
        let conv = MLX.conv1d(paddedIn, toExternalKernel.asType(.float32), stride: orig)
        let flat = conv.reshaped(batch, -1)
        let outLen = (config.resampleNew * length + orig - 1) / orig
        return flat[0..., ..<min(flat.dim(1), outLen)]
    }
}

package enum Zonos3CodecMath {
    package static func mlxCenterInverseSTFT(
        magnitude: MLXArray, phase: MLXArray, nFFT: Int, hopLength: Int, targetLength: Int
    ) -> MLXArray {
        let uncentered = ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: magnitude, phase: phase, nFFT: nFFT, hopLength: hopLength
        )
        let start = nFFT / 2
        let needed = start + max(0, targetLength)
        let buf = uncentered.dim(0) < needed
            ? padded(uncentered, widths: [IntOrPair((0, needed - uncentered.dim(0)))])
            : uncentered
        return buf[start ..< needed]
    }

    package static func cpuCenterInverseSTFT(
        magnitude: [[Float]], phase: [[Float]], nFFT: Int, hopLength: Int, targetLength: Int
    ) -> [Float] {
        let uncentered = ISTFTNetSynthesizer.cpuInverseSTFT(
            magnitude: magnitude, phase: phase, nFFT: nFFT, hopLength: hopLength
        )
        let start = nFFT / 2
        let needed = start + max(0, targetLength)
        let paddedBuf = uncentered.count < needed
            ? uncentered + Array(repeating: Float(0), count: needed - uncentered.count)
            : uncentered
        return Array(paddedBuf[start ..< needed])
    }

    package static func makeSincResampleKernel(
        orig: Int, new: Int, width: Int, lowpassWidth: Double = 16.0, rolloff: Double = 0.99
    ) -> MLXArray {
        let baseFreq = Double(min(orig, new)) * rolloff
        let tapCount = 2 * width + orig
        var values = [Float]()
        values.reserveCapacity(new * tapCount)
        for phase in 0 ..< new {
            let phaseShift = -Double(phase) / Double(new)
            for idx in (-width) ..< (width + orig) {
                var t = (phaseShift + Double(idx) / Double(orig)) * baseFreq
                t = max(-lowpassWidth, min(lowpassWidth, t))
                let window = pow(cos(t * .pi / lowpassWidth / 2.0), 2.0)
                let arg = t * .pi
                let sinc = abs(arg) < 1e-12 ? 1.0 : sin(arg) / arg
                values.append(Float(sinc * window * (baseFreq / Double(orig))))
            }
        }
        return MLXArray(values).reshaped(new, tapCount, 1)
    }

    package static func cpuSincResample(
        _ samples: [Float], kernel: [Float], orig: Int, new: Int, width: Int
    ) -> [Float] {
        guard !samples.isEmpty else { return [] }
        let tapCount = 2 * width + orig
        let padded = Array(repeating: Float(0), count: width) + samples + Array(repeating: Float(0), count: width + orig)
        let steps = max(0, (padded.count - tapCount) / orig + 1)
        let outLen = (new * samples.count + orig - 1) / orig
        var out = [Float]()
        out.reserveCapacity(steps * new)
        for step in 0 ..< steps {
            let base = step * orig
            for p in 0 ..< new {
                let kBase = p * tapCount
                var acc: Double = 0
                for k in 0 ..< tapCount {
                    acc += Double(padded[base + k]) * Double(kernel[kBase + k])
                }
                out.append(Float(acc))
                if out.count == outLen { return out }
            }
        }
        return Array(out.prefix(outLen))
    }
}
