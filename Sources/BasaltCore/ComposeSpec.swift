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

        public init(source: Source, keep: Keep? = nil, idleUnload: Int? = nil, apiKeyEnv: String? = nil, data: String? = nil) {
            (self.source, self.keep, self.idleUnload, self.apiKeyEnv, self.data) = (source, keep, idleUnload, apiKeyEnv, data)
        }

        public var isCloud: Bool {
            if case .cloud = source { true } else { false }
        }
    }

    public static let appleVision = "apple-vision"
    public static let speechAnalyzer = "speech-analyzer"
    public static let defaultIdleUnload = 120
    public static let noPerceived = "no-perceived"
    public static let shareMedia = "media"

    public var version = 1
    public var name: String
    public var memory: PipelineMemory = .auto
    public var models: [String: Model]
    public var documents: [String] = []
    public var audio: [String] = []
    public var cascade: [String] = []
    public var escalateBelow: Double?
    public var abstainBelow: Double?
    public var minMargin: Double?
    public var answer: String
    public var routes: [AnswerRoute] = []
    public var budget: Double?
    public var pipelines: [String: PipelineSpec] = [:]

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
    public var answer: String
    public var routes: [AnswerRoute] = []

    public init(answer: String) { self.answer = answer }
}

extension ComposeSpec {
    public static let defaultPipeline = "default"

    public var pipelineNames: [String] { [Self.defaultPipeline] + pipelines.keys.sorted() }

    public var defaultSpec: PipelineSpec {
        var spec = PipelineSpec(answer: answer)
        (spec.documents, spec.audio, spec.cascade, spec.routes) = (documents, audio, cascade, routes)
        (spec.escalateBelow, spec.abstainBelow, spec.minMargin) = (escalateBelow, abstainBelow, minMargin)
        return spec
    }

    public func view(_ name: String) -> ComposeSpec? {
        guard name != Self.defaultPipeline else { return self }
        guard let pipeline = pipelines[name] else { return nil }
        var view = self
        (view.documents, view.audio, view.cascade, view.routes, view.answer) = (pipeline.documents, pipeline.audio, pipeline.cascade, pipeline.routes, pipeline.answer)
        (view.escalateBelow, view.abstainBelow, view.minMargin, view.pipelines) = (pipeline.escalateBelow, pipeline.abstainBelow, pipeline.minMargin, [:])
        return view
    }

    public var views: [(name: String, spec: ComposeSpec)] { pipelineNames.compactMap { name in view(name).map { (name, $0) } } }
}
