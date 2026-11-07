import Foundation

public protocol InferenceEngine: AudioTranscriber {
    func load(modelPath: String) async throws(BasaltError)
    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse
    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk>
    var isLoaded: Bool { get async }
    var kvMemoryBudget: ByteCount? { get async }
    var supportsKVZones: Bool { get async }
    var supportsKVEviction: Bool { get async }
    var supportsKVQuantization: Bool { get async }
    var evictsKVInFullPrecision: Bool { get async }
    var hybridAnchorUsage: (count: Int, bytes: Int64)? { get async }
    var activeDeltas: [OverlayInfo] { get async }
    func promptTokenCount(for request: GenerationRequest) async -> Int?
    func persistPromptCache(within deadline: Duration) async
    func apply(_ directive: MemoryDirective) async -> MemoryLimits?
    func pace(with pacer: PrefillPacer) async
    var adapterNames: [String] { get async }
    var activeAdapter: String? { get async }
    func loadAdapters(_ directories: [String: URL]) async throws(BasaltError)
    func useAdapter(_ name: String?) async throws(BasaltError)
    var transcribesAudio: Bool { get async }
}

extension InferenceEngine {
    public var kvMemoryBudget: ByteCount? { get async { nil } }
    public var supportsKVZones: Bool { get async { true } }
    public var supportsKVEviction: Bool { get async { true } }
    public var supportsKVQuantization: Bool { get async { true } }
    public var evictsKVInFullPrecision: Bool { get async { false } }
    public var hybridAnchorUsage: (count: Int, bytes: Int64)? { get async { nil } }
    public var activeDeltas: [OverlayInfo] { get async { [] } }
    public func promptTokenCount(for request: GenerationRequest) async -> Int? { nil }
    public func persistPromptCache(within deadline: Duration) async {}
    public func apply(_ directive: MemoryDirective) async -> MemoryLimits? { nil }
    public func pace(with pacer: PrefillPacer) async {}
    public var adapterNames: [String] { get async { [] } }
    public var activeAdapter: String? { get async { nil } }
    public func useAdapter(_ name: String?) async throws(BasaltError) {}
    public var transcribesAudio: Bool { get async { false } }

    public func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        throw .noAudioEncoder("")
    }

    public func loadAdapters(_ directories: [String: URL]) async throws(BasaltError) {
        guard directories.isEmpty else { throw .invalidRequestStructure(details: "this engine serves no LoRA adapters") }
    }
}
