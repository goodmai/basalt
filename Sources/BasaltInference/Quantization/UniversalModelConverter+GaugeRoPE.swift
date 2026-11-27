import BasaltCore
import Foundation
import MLX
import MLXLMCommon

extension UniversalModelConverter {
    static func exportGaugeRoPE(
        request: ModelConversionRequest,
        detected: DetectedSourceCheckpoint,
        sourceURL: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (BasaltCore::ModelConversionProgress) -> Void
    ) async throws -> BasaltCore::ModelConversionResult {
        let layout = try copyForGaugeRoPE(request: request, detected: detected, sourceURL: sourceURL, outputURL: outputURL)
        onProgress(.init(
            phase: "calibrating", completed: 1, total: 2,
            message: "Weights copied unchanged; calibrating the GaugeRoPE codec on up to \(max(layout.calibrationTokens, GaugeRoPECalibration.storedCalibrationTokens)) tokens"))
        let engine = MLXInferenceEngine(.conversion)
        defer { await withTaskCancellationShield { await engine.unload(); Memory.clearCache() } }
        try await engine.load(modelPath: outputURL.path)
        guard let container = await engine.container else { throw BasaltError.modelNotLoaded }
        let layers = try await writeGaugeRoPECodec(container, layout: layout, calibrationText: request.calibrationText, to: outputURL)
        onProgress(.init(
            phase: "writing", completed: 2, total: 2,
            message: "Stored the GaugeRoPE codec of \(layers) full-attention layers in \(GaugeRoPECodecStore.fileName)"))
        let shards = (try? FileManager.default.contentsOfDirectory(atPath: outputURL.path).filter { $0.hasSuffix(".safetensors") }.count) ?? 1
        return ModelConversionResult(
            outputId: request.outputId, bytesBefore: detected.totalBytes,
            bytesAfter: UniversalSourceReader.directoryWeightBytes(outputURL), shards: shards)
    }

    static func copyForGaugeRoPE(
        request: ModelConversionRequest, detected: DetectedSourceCheckpoint, sourceURL: URL, outputURL: URL
    ) throws -> GaugeRoPELayout {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: sourceURL.path)) ?? [])
            .filter { !$0.hasPrefix(".") && $0 != GaugeRoPECodecStore.fileName }
        guard !detected.format.hasPrefix("gguf"), !detected.format.hasPrefix("modelopt:"),
              names.contains(where: { $0.hasSuffix(".safetensors") }) else {
            throw BasaltError.invalidRequestStructure(
                details: "convert: gaugerope keeps the weights as they are and adds a KV codec, so the source must be an MLX checkpoint directory (config.json beside safetensors weights), and this one is \(detected.format); convert it to an MLX format first (#647)")
        }
        guard request.adapter == nil, !request.pvTune, !request.macaqRefine else {
            throw BasaltError.invalidRequestStructure(
                details: "convert: gaugerope leaves the weights unchanged, so adapter, pv_tune and macaq_refine do not apply to it (#647)")
        }
        guard let layout = try ModelCapabilities.gaugeRoPELayout(
            .gaugeRoPE, config: ModelCache.config(at: sourceURL), ropeYarn: nil, path: sourceURL.path) else {
            throw BasaltError.modelConfigurationInvalid(path: sourceURL.path, reason: "no GaugeRoPE layout for this checkpoint (#647)")
        }
        try QTIPNativeLoader.validateOutput(source: sourceURL, output: outputURL)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        for name in names {
            try FileManager.default.copyItem(
                at: sourceURL.appending(component: name).resolvingSymlinksInPath(), to: outputURL.appending(component: name))
        }
        return layout
    }

    static func writeGaugeRoPECodec(
        _ container: ModelContainer, layout: GaugeRoPELayout, calibrationText: String?, to directory: URL
    ) async throws -> Int {
        let codecs = try await GaugeRoPECalibration.codecs(container, layout: layout, calibrationText: calibrationText)
        try GaugeRoPECodecStore.save(
            codecs, layout: layout, fingerprint: GaugeRoPECodecStore.fingerprint(modelDirectory: directory), to: directory)
        return codecs.count
    }
}
