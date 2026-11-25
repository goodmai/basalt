import Foundation

public struct ComposeSpec: Sendable, Equatable {
    public enum Source: Sendable, Equatable {
        case local(String)
        case cloud(provider: String, model: String)

        public static let cloudPrefix = "cloud:"
        public static let providers = ["openrouter", "gemini"]

        public init(_ text: String) {
            guard text.hasPrefix(Self.cloudPrefix) else {
                self = .local(text)
                return
            }
            let rest = text.dropFirst(Self.cloudPrefix.count)
            let provider = rest.prefix { $0 != "/" }
            self = .cloud(provider: String(provider), model: String(rest.dropFirst(provider.count + 1)))
        }
    }

    public enum Keep: String, Sendable, Equatable, CaseIterable {
        case resident, sequential
        case onDemand = "on-demand"
    }

    public struct Model: Sendable, Equatable {
        public var source: Source
        public var keep: Keep?
        public var idleUnload: Int?
        public var apiKeyEnv: String?
        public var data: String?
        public var budget: Double?
        public var speculative: ModelSpeculativeSpec?
        public var diffusionText: DiffusionTextSpec?

        public init(source: Source, keep: Keep? = nil, idleUnload: Int? = nil, apiKeyEnv: String? = nil, data: String? = nil) {
            (self.source, self.keep, self.idleUnload, self.apiKeyEnv, self.data) = (source, keep, idleUnload, apiKeyEnv, data)
        }

        public var isCloud: Bool {
            if case .cloud = source { true } else { false }
        }
    }

    public struct Speak: Sendable, Codable, Equatable {
        public var chain: [String]
        public var voice: String?
        public var language: String?
        public var referenceAudio: String?
        public var onGPUBusy: SpeechOnGPUBusy?
        public var sentenceStreaming: Bool?
        public var format: SpeechAudioFormat?
        public var exaggeration: Double?
        public var maxTTFAMs: Int?

        public init(
            chain: [String],
            voice: String? = nil,
            onGPUBusy: SpeechOnGPUBusy? = nil,
            sentenceStreaming: Bool? = nil,
            format: SpeechAudioFormat? = nil,
            maxTTFAMs: Int? = nil
        ) {
            (self.chain, self.voice, self.onGPUBusy) = (chain, voice, onGPUBusy)
            (self.sentenceStreaming, self.format, self.maxTTFAMs) = (sentenceStreaming, format, maxTTFAMs)
        }

        enum CodingKeys: String, CodingKey {
            case chain, voice, language, format, exaggeration
            case referenceAudio = "reference_audio"
            case onGPUBusy = "on_gpu_busy"
            case sentenceStreaming = "sentence_streaming"
            case maxTTFAMs = "max_ttfa_ms"
        }
    }

    public typealias Pipeline = PipelineSpec

    public static let appleVision = "apple-vision"
    public static let speechAnalyzer = "speech-analyzer"
    public static let appleSpeech = "apple-speech"
    public static let defaultIdleUnload = 120
    public static let noPerceived = "no-perceived"
    public static let shareMedia = "media"

    public var version = 1
    public var schemaMinorVersion = 0
    public var name: String
    public var memory: PipelineMemory = .auto
    public var models: [String: Model]
    public var documents: [String] = []
    public var audio: [String] = []
    public var cascade: [String] = []
    public var escalateBelow: Double?
    public var abstainBelow: Double?
    public var minMargin: Double?
    public var epistemic: EpistemicGateSpec?
    public var answer: String
    public var answerAdapter: String?
    public var routes: [AnswerRoute] = []
    public var speak: Speak?
    public var imagine: Imagine?
    public var budget: Double?
    public var pipelines: [String: PipelineSpec] = [:]
    public var experimental = Experimental()
    public var mcp = MCP()
    public var adapters: [String: ComposeAdapterSpec] = [:]
    public var benchmarks: ComposeBenchmarksSpec?
    public var voice: VoiceDuplexSpec?
    public var music: MusicComposeSpec?
    public var video: VideoComposeSpec?

    public var unservedMediaModels: Set<String> {
        Set([music?.model, video?.model].compactMap { $0 }.filter { !$0.isEmpty })
    }
    public var train: ComposeTrainSpec?
    public var cluster: ComposeClusterSpec?
    public var semanticCache: SemanticCacheConfig?

    public init(name: String, models: [String: Model], answer: String) {
        (self.name, self.models, self.answer) = (name, models, answer)
    }

    public func local(_ name: String) -> String? {
        guard case .local(let source)? = models[name]?.source else { return nil }
        return source
    }

    public var answerModels: [String] {
        var names = [answer]
        for route in routes {
            for name in [route.model] + (route.fallback.map { [$0] } ?? []) where !names.contains(name) { names.append(name) }
        }
        return names
    }

    public var ocr: String? { documents.first { models[$0] != nil } }
    public var speech: String? { audio.first { models[$0] != nil } }
}

public struct PipelineSpec: Sendable, Equatable {
    public var documents: [String] = []
    public var audio: [String] = []
    public var cascade: [String] = []
    public var escalateBelow: Double?
    public var abstainBelow: Double?
    public var minMargin: Double?
    public var epistemic: EpistemicGateSpec?
    public var answer: String
    public var answerAdapter: String?
    public var routes: [AnswerRoute] = []
    public var speak: ComposeSpec.Speak?
    public var imagine: ComposeSpec.Imagine?

    public init(answer: String) { self.answer = answer }
}

extension ComposeSpec {
    public static let defaultPipeline = "default"

    public var pipelineNames: [String] { [Self.defaultPipeline] + pipelines.keys.sorted() }

    public var defaultSpec: PipelineSpec {
        var spec = PipelineSpec(answer: answer)
        (spec.documents, spec.audio, spec.cascade, spec.routes) = (documents, audio, cascade, routes)
        (spec.escalateBelow, spec.abstainBelow, spec.minMargin, spec.epistemic) = (escalateBelow, abstainBelow, minMargin, epistemic)
        (spec.answerAdapter, spec.speak, spec.imagine) = (answerAdapter, speak, imagine)
        return spec
    }

    public func view(_ name: String) -> ComposeSpec? {
        guard name != Self.defaultPipeline else { return self }
        guard let pipeline = pipelines[name] else { return nil }
        var view = self
        (view.documents, view.audio, view.cascade, view.routes, view.answer) = (pipeline.documents, pipeline.audio, pipeline.cascade, pipeline.routes, pipeline.answer)
        (view.escalateBelow, view.abstainBelow, view.minMargin, view.epistemic, view.pipelines) = (pipeline.escalateBelow, pipeline.abstainBelow, pipeline.minMargin, pipeline.epistemic, [:])
        (view.answerAdapter, view.speak, view.imagine) = (pipeline.answerAdapter, pipeline.speak, pipeline.imagine)
        return view
    }

    public var views: [(name: String, spec: ComposeSpec)] { pipelineNames.compactMap { name in view(name).map { (name, $0) } } }
}
