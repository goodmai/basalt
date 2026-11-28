import Foundation
import MLX
import MLXNN

final class KokoroAdaINResBlock1: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "adain1") var adain1: [KokoroAdaIN1d]
    @ModuleInfo(key: "adain2") var adain2: [KokoroAdaIN1d]
    @ParameterInfo(key: "alpha1") var alpha1: [MLXArray]
    @ParameterInfo(key: "alpha2") var alpha2: [MLXArray]

    init(channels: Int, kernelSize: Int, dilations: [Int] = [1, 3, 5], styleDim: Int = 128) {
        self._convs1.wrappedValue = dilations.map { dilation in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize * dilation - dilation) / 2,
                dilation: dilation
            )
        }
        self._convs2.wrappedValue = dilations.map { _ in
            Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                padding: (kernelSize - 1) / 2
            )
        }
        self._adain1.wrappedValue = dilations.map { _ in KokoroAdaIN1d(styleDim: styleDim, channels: channels) }
        self._adain2.wrappedValue = dilations.map { _ in KokoroAdaIN1d(styleDim: styleDim, channels: channels) }
        self._alpha1.wrappedValue = dilations.map { _ in MLXArray.ones([1, 1, channels], dtype: .float32) }
        self._alpha2.wrappedValue = dilations.map { _ in MLXArray.ones([1, 1, channels], dtype: .float32) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray) -> MLXArray {
        var h = x
        for index in convs1.indices {
            var xt = adain1[index](h, style: style)
            let a1 = alpha1[index]
            xt = xt + (1.0 / a1) * (sin(a1 * xt).square())
            xt = convs1[index](xt)
            xt = adain2[index](xt, style: style)
            let a2 = alpha2[index]
            xt = xt + (1.0 / a2) * (sin(a2 * xt).square())
            xt = convs2[index](xt)
            h = h + xt
        }
        return h
    }
}

final class KokoroSourceModuleHnNSF: Module {
    @ModuleInfo(key: "l_linear") var lLinear: Linear
    let sampleRate: Int
    let harmonicNum: Int
    let sineAmp: Float = 0.1
    let voicedThreshold: Float = 10.0

    init(sampleRate: Int = 24_000, harmonicNum: Int = 8) {
        self.sampleRate = sampleRate
        self.harmonicNum = harmonicNum
        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
        super.init()
    }

    func callAsFunction(_ f0Upsampled: MLXArray) -> MLXArray {
        let harmonics = MLXArray((1 ... (harmonicNum + 1)).map(Float.init)).reshaped(1, 1, harmonicNum + 1)
        let f0Mat = f0Upsampled * harmonics
        let radValues = (f0Mat / Float(sampleRate)) % Float(1.0)
        let phase = radValues.cumsum(axis: 1) * (2.0 * Float.pi)
        let uv = (f0Upsampled .> Float(voicedThreshold)).asType(.float32)
        let sines = sin(phase) * sineAmp * uv
        return tanh(lLinear(sines))
    }
}

final class KokoroISTFTGenerator: Module {
    @ModuleInfo(key: "m_source") var mSource: KokoroSourceModuleHnNSF
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "noise_convs") var noiseConvs: [Conv1d]
    @ModuleInfo(key: "noise_res") var noiseRes: [KokoroAdaINResBlock1]
    @ModuleInfo(key: "resblocks") var resblocks: [KokoroAdaINResBlock1]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let upsampleRates: [Int]
    let nFFT: Int
    let hopLength: Int
    let numKernels: Int
    let totalUpsampleFactor: Int

    init(
        styleDim: Int = 128,
        upsampleInitialChannel: Int = 512,
        nFFT: Int = 20,
        hopLength: Int = 5
    ) {
        let resblockKernelSizes = [3, 7, 11]
        let upsampleRates = [10, 6]
        self.upsampleRates = upsampleRates
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.numKernels = resblockKernelSizes.count
        self.totalUpsampleFactor = upsampleRates.reduce(1, *) * hopLength
        self._mSource.wrappedValue = KokoroSourceModuleHnNSF(sampleRate: 24_000, harmonicNum: 8)
        let stages = Self.buildStages(
            styleDim: styleDim,
            upsampleInitialChannel: upsampleInitialChannel,
            stftChannels: nFFT + 2
        )
        self._ups.wrappedValue = stages.ups
        self._noiseConvs.wrappedValue = stages.noiseConvs
        self._noiseRes.wrappedValue = stages.noiseRes
        self._resblocks.wrappedValue = stages.resblocks
        let finalCh = upsampleInitialChannel >> upsampleRates.count
        self._convPost.wrappedValue = Conv1d(inputChannels: finalCh, outputChannels: nFFT + 2, kernelSize: 7, padding: 3)
        super.init()
    }

    private struct GeneratorStages {
        let ups: [ConvTransposed1d]
        let noiseConvs: [Conv1d]
        let noiseRes: [KokoroAdaINResBlock1]
        let resblocks: [KokoroAdaINResBlock1]
    }

    private static func buildStages(
        styleDim: Int,
        upsampleInitialChannel: Int,
        stftChannels: Int
    ) -> GeneratorStages {
        let upsampleRates = [10, 6]
        let upsampleKernelSizes = [20, 12]
        let resblockKernelSizes = [3, 7, 11]
        let resblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        var upLayers: [ConvTransposed1d] = []
        var nConvs: [Conv1d] = []
        var nRes: [KokoroAdaINResBlock1] = []
        var blocks: [KokoroAdaINResBlock1] = []
        for index in upsampleRates.indices {
            let u = upsampleRates[index]
            let k = upsampleKernelSizes[index]
            let inCh = upsampleInitialChannel >> index
            let outCh = upsampleInitialChannel >> (index + 1)
            upLayers.append(ConvTransposed1d(
                inputChannels: inCh, outputChannels: outCh, kernelSize: k, stride: u, padding: (k - u) / 2
            ))
            let isLast = index + 1 == upsampleRates.count
            let stride = isLast ? 1 : upsampleRates[(index + 1)...].reduce(1, *)
            nConvs.append(Conv1d(
                inputChannels: stftChannels, outputChannels: outCh,
                kernelSize: isLast ? 1 : stride * 2, stride: stride, padding: isLast ? 0 : (stride + 1) / 2
            ))
            nRes.append(KokoroAdaINResBlock1(channels: outCh, kernelSize: isLast ? 11 : 7, dilations: [1, 3, 5], styleDim: styleDim))
            for kIdx in resblockKernelSizes.indices {
                blocks.append(KokoroAdaINResBlock1(
                    channels: outCh, kernelSize: resblockKernelSizes[kIdx], dilations: resblockDilations[kIdx], styleDim: styleDim
                ))
            }
        }
        return GeneratorStages(ups: upLayers, noiseConvs: nConvs, noiseRes: nRes, resblocks: blocks)
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray, f0: MLXArray) -> MLXArray {
        let f0Up = repeated(f0.expandedDimensions(axis: -1), count: totalUpsampleFactor, axis: 1)
        let harSource = mSource(f0Up).squeezed(axis: -1)
        let har = Self.forwardSTFT(harSource, nFFT: nFFT, hopLength: hopLength)
        var h = x
        for index in ups.indices {
            h = leakyRelu(h, negativeSlope: 0.1)
            h = ups[index](h)
            let xSource = noiseRes[index](noiseConvs[index](har), style: style)
            let targetLen = min(h.dim(1), xSource.dim(1))
            h = h[0..., ..<targetLen, 0...] + xSource[0..., ..<targetLen, 0...]
            var xs: MLXArray?
            for kIdx in 0 ..< numKernels {
                let out = resblocks[index * numKernels + kIdx](h, style: style)
                xs = xs.map { $0 + out } ?? out
            }
            h = (xs ?? h) / Float(numKernels)
        }
        h = leakyRelu(h, negativeSlope: 0.01)
        let post = convPost(h)[0]
        let bins = nFFT / 2 + 1
        let spec = exp(clip(post[0..., ..<bins], min: -10.0, max: 10.0))
        let phase = sin(post[0..., bins...])
        let rawWave = ISTFTNetSynthesizer.mlxInverseSTFT(
            magnitude: spec,
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

    static func forwardSTFT(_ signal: MLXArray, nFFT: Int, hopLength: Int) -> MLXArray {
        let wave = signal[0].asType(.float32)
        let pad = nFFT / 2
        let padded = MLX.padded(wave, widths: [IntOrPair((pad, pad))], mode: .edge)
        let totalLen = padded.dim(0)
        let frames = max(1, (totalLen - nFFT) / hopLength + 1)
        let starts = MLXArray(Int32(0) ..< Int32(frames)).expandedDimensions(axis: 1) * Int32(hopLength)
        let offsets = MLXArray(Int32(0) ..< Int32(nFFT)).expandedDimensions(axis: 0)
        let indices = (starts + offsets).reshaped(-1)
        let framed = padded[indices].reshaped(frames, nFFT)
        let window = MLXArray(ISTFTNetSynthesizer.hannWindow(length: nFFT)).reshaped(1, nFFT)
        let spec = MLXFFT.rfft(framed * window, n: nFFT, axis: -1)
        let real = spec.realPart()
        let imag = spec.imaginaryPart()
        let mag = sqrt(real * real + imag * imag + Float(1e-9))
        let phase = atan2(imag, real)
        return concatenated([mag, phase], axis: -1).expandedDimensions(axis: 0)
    }
}

final class KokoroDecoder: Module {
    @ModuleInfo(key: "F0_conv") var f0Conv: Conv1d
    @ModuleInfo(key: "N_conv") var nConv: Conv1d
    @ModuleInfo(key: "asr_res") var asrRes: Conv1d
    @ModuleInfo(key: "encode") var encode: KokoroAdainResBlk1d
    @ModuleInfo(key: "decode") var decode: [KokoroAdainResBlk1d]
    @ModuleInfo(key: "generator") var generator: KokoroISTFTGenerator

    init(dimIn: Int = 512, f0Channel: Int = 512, styleDim: Int = 128, nFFT: Int = 20, hopLength: Int = 5) {
        self._f0Conv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        self._nConv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        self._asrRes.wrappedValue = Conv1d(inputChannels: dimIn, outputChannels: 64, kernelSize: 1)
        self._encode.wrappedValue = KokoroAdainResBlk1d(dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim, upsample: false)
        self._decode.wrappedValue = [
            KokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, upsample: false),
            KokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, upsample: false),
            KokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim, upsample: false),
            KokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: f0Channel, styleDim: styleDim, upsample: true),
        ]
        self._generator.wrappedValue = KokoroISTFTGenerator(
            styleDim: styleDim,
            upsampleInitialChannel: f0Channel,
            nFFT: nFFT,
            hopLength: hopLength
        )
        super.init()
    }

    func callAsFunction(asr: MLXArray, f0Curve: MLXArray, nCurve: MLXArray, style: MLXArray) -> MLXArray {
        let f0Down = f0Conv(f0Curve.expandedDimensions(axis: -1))
        let nDown = nConv(nCurve.expandedDimensions(axis: -1))
        let steps = min(asr.dim(1), min(f0Down.dim(1), nDown.dim(1)))
        let asrTrim = asr[0..., ..<steps, 0...]
        let f0Trim = f0Down[0..., ..<steps, 0...]
        let nTrim = nDown[0..., ..<steps, 0...]
        var x = encode(concatenated([asrTrim, f0Trim, nTrim], axis: -1), style: style)
        let res = asrRes(asrTrim)
        for block in decode {
            let cat = concatenated([x, res, f0Trim, nTrim], axis: -1)
            x = block(cat, style: style)
        }
        return generator(x, style: style, f0: f0Curve)
    }
}

package final class KokoroStyleTTS2Model: Module {
    @ModuleInfo(key: "bert") var bert: KokoroALBERT
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo(key: "text_encoder") var textEncoder: KokoroTextEncoder
    @ModuleInfo(key: "predictor") var predictor: KokoroProsodyPredictor
    @ModuleInfo(key: "decoder") var decoder: KokoroDecoder

    init(vocabSize: Int = 178, hiddenDim: Int = 512, styleDim: Int = 128, maxDur: Int = 50, nFFT: Int = 20, hopLength: Int = 5) {
        self._bert.wrappedValue = KokoroALBERT(vocabSize: vocabSize)
        self._bertEncoder.wrappedValue = Linear(768, hiddenDim)
        self._textEncoder.wrappedValue = KokoroTextEncoder(vocabSize: vocabSize, channels: hiddenDim)
        self._predictor.wrappedValue = KokoroProsodyPredictor(dModel: hiddenDim, styleDim: styleDim, maxDur: maxDur)
        self._decoder.wrappedValue = KokoroDecoder(dimIn: hiddenDim, f0Channel: hiddenDim, styleDim: styleDim, nFFT: nFFT, hopLength: hopLength)
        super.init()
    }

    func synthesize(tokenIDs: [Int32], refStyle: MLXArray, speed: Double) -> MLXArray {
        let ids = MLXArray(tokenIDs).reshaped(1, tokenIDs.count)
        let styleAcoustic = refStyle[0..., ..<128]
        let styleProsody = refStyle[0..., 128...]
        let bertDur = bert(ids)
        let dEn = bertEncoder(bertDur)
        let d = predictor.textEncoder(dEn, style: styleProsody)
        let durHidden = predictor.lstm(d)
        let durLogits = predictor.durationProj(durHidden)
        let durRaw = sigmoid(durLogits).sum(axis: -1) / Float(max(0.25, speed))
        let durInts = durRaw[0].asArray(Float.self).map { max(1, Int($0.rounded())) }
        var gatherIdx: [Int32] = []
        gatherIdx.reserveCapacity(durInts.reduce(0, +))
        for (idx, count) in durInts.enumerated() {
            gatherIdx.append(contentsOf: repeatElement(Int32(idx), count: count))
        }
        let alignIndices = MLXArray(gatherIdx)
        let en = d.take(alignIndices, axis: 1)
        let (f0Pred, nPred) = predictor.predictF0N(en, style: styleProsody)
        let tEn = textEncoder(ids)
        let asr = tEn.take(alignIndices, axis: 1)
        return decoder(asr: asr, f0Curve: f0Pred, nCurve: nPred, style: styleAcoustic)
    }
}
