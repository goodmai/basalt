import Foundation
import MLX
import MLXNN

final class NemotronSpeakerModules: Module {
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ModuleInfo(key: "first_hidden_to_hidden") var firstHiddenToHidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var singleHiddenToSpks: Linear
    @ModuleInfo(key: "hidden_to_spks") var hiddenToSpks: Linear
    @ParameterInfo(key: "learnable_sil_emb") var learnableSilEmb: MLXArray

    init(_ config: NemotronSortformerModulesConfig) {
        self._encoderProj.wrappedValue = Linear(config.fcDModel, config.tfDModel, bias: true)
        self._subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: config.tfDModel,
            outputChannels: config.tfDModel * config.subsamplingFactor,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._firstHiddenToHidden.wrappedValue = Linear(config.tfDModel, config.tfDModel, bias: true)
        self._singleHiddenToSpks.wrappedValue = Linear(config.tfDModel, config.numSpks, bias: true)
        self._hiddenToSpks.wrappedValue = Linear(2 * config.tfDModel, config.numSpks, bias: true)
        self._learnableSilEmb.wrappedValue = MLXArray.zeros([config.fcDModel])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = encoderProj(x)
        let b = projected.dim(0)
        let h = projected.dim(2)
        let upsampled = subpixelUpsample(projected).reshaped(b, -1, h)
        let hidden = relu(firstHiddenToHidden(relu(upsampled)))
        return sigmoid(singleHiddenToSpks(hidden))
    }
}

struct NemotronDiarizationStreamingState {
    var spkcache: MLXArray
    var spkcachePreds: MLXArray
    var fifo: MLXArray
    var fifoPreds: MLXArray
}

final class NemotronDiarizationModel: Module {
    let config: NemotronDiarizationConfig
    @ModuleInfo(key: "encoder") var encoder: NemotronConformerEncoder
    @ModuleInfo(key: "sortformer_modules") var sortformerModules: NemotronSpeakerModules
    @ModuleInfo(key: "preprocessor") var preprocessor: NemotronMelFeatures

    init(_ config: NemotronDiarizationConfig) {
        self.config = config
        self._encoder.wrappedValue = NemotronConformerEncoder(config.encoder)
        self._sortformerModules.wrappedValue = NemotronSpeakerModules(config.sortformerModules)
        self._preprocessor.wrappedValue = NemotronMelFeatures(config.preprocessor)
        super.init()
    }

    var parameterDType: DType {
        encoder.preEncode.parameterDType
    }

    func makeStreamingState() -> NemotronDiarizationStreamingState {
        let dtype = parameterDType
        let dModel = config.encoder.dModel
        let spks = config.maxNumOfSpks
        return NemotronDiarizationStreamingState(
            spkcache: MLXArray.zeros([1, 0, dModel], dtype: dtype),
            spkcachePreds: MLXArray.zeros([1, 0, spks], dtype: .float32),
            fifo: MLXArray.zeros([1, 0, dModel], dtype: dtype),
            fifoPreds: MLXArray.zeros([1, 0, spks], dtype: .float32)
        )
    }

    func streamingStep(
        features: MLXArray,
        state: inout NemotronDiarizationStreamingState,
        centralFrames: Int,
        featureLength: Int,
        leftContextFrames: Int
    ) -> MLXArray {
        let cfg = config.sortformerModules
        let factor = config.encoder.subsamplingFactor
        let (chunk, lengths) = encoder.preEncode(
            features.asType(parameterDType), lengths: MLXArray([Int32(featureLength)])
        )
        let cacheLen = state.spkcache.dim(1)
        let fifoLen = state.fifo.dim(1)
        let combined = concatenated([state.spkcache, state.fifo, chunk], axis: 1)
        let totalLengths = lengths + Int32(cacheLen + fifoLen)
        var high = sortformerModules(encoder(combined, lengths: totalLengths))
        let validLimit = (lengths[0].item(Int.self) + cacheLen + fifoLen) * factor
        let valid = MLXArray(Int32(0) ..< Int32(high.dim(1))) .< Int32(validLimit)
        high = high * valid.expandedDimensions(axes: [0, 2]).asType(high.dtype)
        let low = high.asType(.float32).reshaped(1, -1, factor, config.maxNumOfSpks).mean(axis: 2)
        let n = (centralFrames + factor - 1) / factor
        let left = leftContextFrames / factor
        let start = cacheLen + fifoLen + left
        let frameStart = start * factor
        let result = high[0, frameStart ..< (frameStart + centralFrames), 0...].asType(.float32)
        let chunkSlice = chunk[0..., left ..< min(chunk.dim(1), left + n), 0...]
        state.fifo = concatenated([state.fifo, chunkSlice], axis: 1)
        let prevFifoPreds = low[0..., cacheLen ..< (cacheLen + fifoLen), 0...]
        let newFifoPreds = low[0..., start ..< min(low.dim(1), start + n), 0...]
        state.fifoPreds = concatenated([prevFifoPreds, newFifoPreds], axis: 1)
        if state.fifo.dim(1) > cfg.fifoLen {
            let pop = min(state.fifo.dim(1), max(cfg.spkcacheUpdatePeriod, state.fifo.dim(1) - cfg.fifoLen))
            state.spkcache = concatenated([state.spkcache, state.fifo[0..., ..<pop, 0...]], axis: 1)
            state.spkcachePreds = concatenated([state.spkcachePreds, state.fifoPreds[0..., ..<pop, 0...]], axis: 1)
            state.fifo = state.fifo[0..., pop..., 0...]
            state.fifoPreds = state.fifoPreds[0..., pop..., 0...]
            if state.spkcache.dim(1) > cfg.spkcacheLen {
                let trim = state.spkcache.dim(1) - cfg.spkcacheLen
                state.spkcache = state.spkcache[0..., trim..., 0...]
                state.spkcachePreds = state.spkcachePreds[0..., trim..., 0...]
            }
        }
        eval(result, state.spkcache, state.spkcachePreds, state.fifo, state.fifoPreds)
        return result
    }
}

private struct SinusoidalKernel {
    let matrix: MLXArray
}

final class NemotronParallelExpertEncoder: Module {
    let peConfig: NemotronPEEncoderConfig
    let factor: Int
    private let diarKernel: SinusoidalKernel

    @ModuleInfo(key: "asr_encoder") var asrEncoder: NemotronConformerEncoder
    @ModuleInfo(key: "diarization_model") var diarizationModel: NemotronDiarizationModel
    @ModuleInfo(key: "asr_norm") var asrNorm: LayerNorm
    @ModuleInfo(key: "diar_norm") var diarNorm: LayerNorm

    init(_ config: NemotronTranscribeConfiguration) {
        let asrCfg = config.asrConfig
        let diarCfg = config.peEncoderConfig.diarizationModelCfg
        self.peConfig = config.peEncoderConfig
        self.factor = asrCfg.subsamplingFactor
        self._asrEncoder.wrappedValue = NemotronConformerEncoder(asrCfg)
        self._diarizationModel.wrappedValue = NemotronDiarizationModel(diarCfg)
        self._asrNorm.wrappedValue = LayerNorm(dimensions: asrCfg.dModel)
        self._diarNorm.wrappedValue = LayerNorm(dimensions: diarCfg.maxNumOfSpks)

        let spks = diarCfg.maxNumOfSpks
        let width = asrCfg.dModel
        let position = MLXArray(Int32(0) ..< Int32(spks)).asType(.float32).expandedDimensions(axis: 1)
        let evenIndices = MLXArray(stride(from: 0, to: width, by: 2).map(Int32.init)).asType(.float32)
        let frequencies = exp(evenIndices * -(Float(log(10_000.0)) / Float(width)))
        let angles = position * frequencies.expandedDimensions(axis: 0)
        self.diarKernel = SinusoidalKernel(
            matrix: stacked([sin(angles), cos(angles)], axis: -1).reshaped(spks, width)
        )
        super.init()
    }

    static func normalizeFeatures(_ features: MLXArray) -> MLXArray {
        let frames = features.dim(-1)
        guard frames > 0 else { return features }
        let reference = features[0..., 0..., 0 ..< 1]
        let mean = reference + (features - reference).mean(axis: -1, keepDims: true)
        let centered = features - mean
        let denom = Float(max(1, frames - 1))
        let variance = (centered * centered).sum(axis: -1, keepDims: true) / denom
        return centered / (sqrt(variance) + Float(1e-5))
    }

    func fuse(states: MLXArray, probabilities: MLXArray) -> MLXArray {
        var probs = probabilities.asType(states.dtype)
        if peConfig.speakerFeatureMode == "thresholded" {
            probs = (probs .> peConfig.speakerActivityThreshold).asType(states.dtype)
        }
        let targets = diarNorm(probs)
        let infusion = matmul(targets, diarKernel.matrix.asType(targets.dtype))
        return asrNorm(states) + peConfig.spkKernelScale * infusion
    }

    func callAsFunction(_ features: MLXArray) throws -> MLXArray {
        let total = features.dim(-1)
        guard total > 0 else {
            return MLXArray.zeros([1, 0, asrNorm.weight?.dim(0) ?? 1280], dtype: asrNorm.weight?.dtype ?? .float32)
        }
        let size = peConfig.onlineInferenceLength * factor
        let left = max(0, peConfig.chunkLeftContext) * factor
        let right = max(0, peConfig.chunkRightContext) * factor
        let dtype = asrNorm.weight?.dtype ?? .float32
        let normalized = Self.normalizeFeatures(features).asType(dtype)
        var state = diarizationModel.makeStreamingState()
        var outputs = [MLXArray]()

        for start in stride(from: 0, to: total, by: size) {
            try Task.checkCancellation()
            let end = min(start + size, total)
            let contextStart = max(0, start - left)
            let contextEnd = min(total, end + right)
            let length = contextEnd - contextStart
            let window = normalized[0..., 0..., contextStart ..< contextEnd]
            let (x, lengths) = asrEncoder.preEncode(window, lengths: MLXArray([Int32(length)]))
            let fullEncoded = asrEncoder(x, lengths: lengths)
            let first = (start - contextStart) / factor
            let count = (end + factor - 1) / factor - (start + factor - 1) / factor
            let encoded = fullEncoded[0..., first ..< (first + count), 0...]
            let central = count * factor
            let rawWindow = features[0..., 0..., contextStart ..< contextEnd]
            let nativeProbs = diarizationModel.streamingStep(
                features: rawWindow,
                state: &state,
                centralFrames: central,
                featureLength: length,
                leftContextFrames: start - contextStart
            )
            let probabilities = nativeProbs.reshaped(1, count, factor, -1).mean(axis: 2)
            let fused = fuse(states: encoded, probabilities: probabilities)
            eval(fused)
            outputs.append(fused)
        }
        return concatenated(outputs, axis: 1)
    }
}

final class NemotronPerception: Module {
    @ModuleInfo(key: "preprocessor") var preprocessor: NemotronMelFeatures
    @ModuleInfo(key: "encoder") var encoder: NemotronParallelExpertEncoder
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ config: NemotronTranscribeConfiguration) {
        self._preprocessor.wrappedValue = NemotronMelFeatures(config.perception.preprocessor)
        self._encoder.wrappedValue = NemotronParallelExpertEncoder(config)
        self._proj.wrappedValue = Linear(
            config.asrConfig.dModel, config.llmConfig.hiddenSize, bias: true
        )
        super.init()
    }

    func callAsFunction(_ audio: MLXArray) throws -> MLXArray {
        let mel = preprocessor(audio)
        let encoded = try encoder(mel)
        return proj(encoded)
    }
}
