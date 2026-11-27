import Foundation
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func expertCacheStats() async -> (stats: ExpertCacheStats, capacity: ByteCount)? {
        guard let container else { return nil }
        return await container.performEvaluating(nonSendable: ()) { context, _ in
            ExpertStreamingLoader.store(in: context.model).map { ($0.stats, $0.capacity) }
        }
    }

    public var expertTelemetry: ExpertTelemetry? {
        get async {
            guard let container else { return nil }
            return await container.performEvaluating(nonSendable: ()) { context, _ in
                ExpertStreamingLoader.store(in: context.model).map {
                    ExpertTelemetry(cache: $0.stats, capacityBytes: $0.capacity.rawValue, layers: $0.routing.layers(expertCounts: $0.expertCountsByLayer))
                }
            }
        }
    }

    func loadContainer(from directory: URL) async throws -> ModelContainer {
        if let parakeet = try await ParakeetLoader.loadContainer(from: directory) {
            logger.info("Parakeet TDT speech recognition: \(directory.lastPathComponent) (NeMo config without model_type) serves /v1/audio/transcriptions only")
            return parakeet
        }
        if let macaq = try await MACaQNativeLoader.loadContainer(from: directory, tokenizerLoader: DefaultTokenizerLoader()) {
            logger.info("Experimental MACaQ native weights: \(directory.lastPathComponent) runs fused Hadamard + channel-balanced low-bit decode on basalt's Metal kernels")
            return macaq
        }
        if let basis = try await BasisTransformNativeLoader.loadContainer(from: directory, tokenizerLoader: DefaultTokenizerLoader()) {
            logger.info("Experimental FlatQuant/LIRQuant native weights: \(directory.lastPathComponent) runs Kronecker/Givens prologue decode on basalt's Metal kernels")
            return basis
        }
        if let qtip = try await QTIPNativeLoader.loadContainer(from: directory, tokenizerLoader: DefaultTokenizerLoader()) {
            logger.info("Experimental QTIP native weights: \(directory.lastPathComponent) runs bitshift trellis blocks on basalt's Metal decode kernel")
            return qtip
        }
        if let nf4 = try await NF4NativeLoader.loadContainer(from: directory, tokenizerLoader: DefaultTokenizerLoader()) {
            logger.info("Experimental NF4 native weights: \(directory.lastPathComponent) runs NormalFloat 4-bit blocks on basalt's Metal kernels")
            return nf4
        }
        if let native = try await GGUFNativeLoader.loadContainer(from: directory, tokenizerLoader: DefaultTokenizerLoader()) {
            logger.info("Experimental GGUF native weights: \(directory.lastPathComponent) runs its i-quant and k-quant blocks on basalt's Metal kernels")
            return native
        }
        let resolvedStreaming = ModelFitAnalyzer.autoExpertStreaming(
            directory: directory,
            workingSetBudget: PromptCacheBudget.gpuWorkingSet(),
            explicit: configuration.expertStreaming,
            disabled: configuration.disableExpertStreaming
        )
        guard let expertStreaming = resolvedStreaming else { return try await loadModelContainer(from: directory, using: DefaultTokenizerLoader()) }
        guard let loaded = try await ExpertStreamingLoader.loadContainer(from: directory, streaming: expertStreaming, tokenizerLoader: DefaultTokenizerLoader()) else {
            logger.warn("--expert-streaming: \(directory.lastPathComponent) has no routed experts to stream; it loads resident")
            return try await loadModelContainer(from: directory, using: DefaultTokenizerLoader())
        }
        if let capacity = await loaded.perform({ ExpertStreamingLoader.store(in: $0.model)?.capacity }) {
            logger.info("Expert streaming: routed experts stay in the safetensors shards and load on demand into \(formatRAMBytes(capacity.rawValue)) of expert slots")
        }
        return loaded
    }

    static func expertReserve(_ container: ModelContainer) async -> ByteCount {
        await container.perform { ExpertStreamingLoader.store(in: $0.model)?.unclaimedReserve } ?? .zero
    }
}
