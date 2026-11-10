import Foundation

public enum PipelineCore {
    public static let questionMaxTokens = 16

    public enum System2Call: Sendable, Equatable {
        case question(id: String, prompt: String)
        case task(prompt: String)

        public var prompt: String {
            switch self {
            case .question(_, let prompt), .task(let prompt): prompt
            }
        }

        public var question: String? {
            if case .question(let id, _) = self { id } else { nil }
        }
    }

    public struct Route: Sendable, Equatable {
        public var answers: [String: PipelineAnswer] = [:]
        public var calls: [System2Call] = []
    }

    public struct System2Reply: Sendable, Equatable {
        public let call: System2Call
        public let model: String
        public let text: String
        public let tokens: Int
        public let ms: Int

        public init(call: System2Call, model: String, text: String, tokens: Int, ms: Int) {
            (self.call, self.model, self.text, self.tokens, self.ms) = (call, model, text, tokens, ms)
        }
    }

    public enum Perceiver: String, Sendable, Equatable {
        case ocrModel, appleVision, speechModel, chatModel, speechAnalyzer

        public var betweenGenerations: Bool {
            switch self {
            case .ocrModel, .speechModel: true
            case .appleVision, .chatModel, .speechAnalyzer: false
            }
        }
    }

    public struct Perceivers: Sendable, Equatable {
        public var ocrModel = false
        public var appleVision = false
        public var speechModel = false
        public var chatAudio = false
        public var speechAnalyzer = false

        public init(ocrModel: Bool = false, appleVision: Bool = false, speechModel: Bool = false, chatAudio: Bool = false, speechAnalyzer: Bool = false) {
            (self.ocrModel, self.appleVision, self.speechModel, self.chatAudio, self.speechAnalyzer) = (ocrModel, appleVision, speechModel, chatAudio, speechAnalyzer)
        }

        public func perceiver(for attachment: PipelineAttachment) -> Perceiver? {
            switch attachment {
            case .document: ocrModel ? .ocrModel : appleVision ? .appleVision : nil
            case .audio: speechModel ? .speechModel : chatAudio ? .chatModel : speechAnalyzer ? .speechAnalyzer : nil
            }
        }
    }

    public static let speechAnalyzerModel = "apple-speech-analyzer"

    public struct Perceiving: Sendable {
        public let index: Int
        public let attachment: PipelineAttachment
        public let perceiver: Perceiver?
    }

    public static func perceiveOrder(_ attachments: [PipelineAttachment], perceivers: Perceivers, grouped: Bool) -> [Perceiving] {
        let items = attachments.enumerated().map { Perceiving(index: $0.offset, attachment: $0.element, perceiver: perceivers.perceiver(for: $0.element)) }
        return grouped ? items.sorted { ($0.perceiver?.rawValue ?? "", $0.index) < ($1.perceiver?.rawValue ?? "", $1.index) } : items
    }

    public static func missingStage(_ request: PipelineRequest, perceivers: Perceivers, hasDecisions: Bool) -> String? {
        for attachment in request.attachments where perceivers.perceiver(for: attachment) == nil {
            switch attachment {
            case .document:
                return "A pdf or image attachment needs an OCR model; start basalt serve with --pipeline-ocr-model <GLM-OCR, MinerU2.5, PaddleOCR-VL or DeepSeek-OCR repo|dir>."
            case .audio:
                return "An audio attachment needs a speech model; start basalt serve with --pipeline-audio-model <Nemotron Transcribe repo|dir>, a chat model with an audio encoder, or --experimental-speech-analyzer."
            }
        }
        if !request.questions.isEmpty, !hasDecisions {
            return "Typed questions need a decision model; start basalt serve with --decision-model <self|repo|dir>."
        }
        return nil
    }

    public static func state(_ perceived: PipelinePerceived, text: String?) -> String {
        let pages = (perceived.pages ?? []).map { "[document \($0.attachment + 1), page \($0.page)]\n\($0.markdown)" }
        let transcript = perceived.transcript.map { "[transcript]\n\($0)" }
        return (pages + [transcript, text].compactMap { $0 }).joined(separator: "\n\n")
    }

    public static func route(questions: [String: SystemOneQuestion], decision: SystemOneResponse?, task: String?, state: String) -> Route {
        var route = Route()
        for (id, question) in questions.sorted(by: { $0.key < $1.key }) {
            if let answer = decision?.answers[id], answer.abstain != true {
                route.answers[id] = PipelineAnswer(answeredBy: .decide, label: label(of: answer, question), decision: answer)
            } else {
                route.calls.append(.question(id: id, prompt: prompt(question, state: state)))
            }
        }
        if let task { route.calls.append(.task(prompt: taskPrompt(task, state: state))) }
        return route
    }

    public static func prompt(_ question: SystemOneQuestion, state: String) -> String {
        let (keys, descriptions) = SystemOneCore.options(question)
        let options = zip(keys, descriptions).map { "\($0): \($1)" }.joined(separator: "\n")
        return state + "\n\nQuestion:\n" + question.instructions + "\n\nOptions:\n" + options
            + "\n\nAnswer with exactly one option label (" + keys.joined(separator: ", ") + ") and nothing else."
    }

    public static func taskPrompt(_ task: String, state: String) -> String {
        state.isEmpty ? task : "Context:\n" + state + "\n\nTask:\n" + task
    }

    public static func request(for call: System2Call, origin: RequestOrigin) -> GenerationRequest {
        let message = [ChatMessageInput(role: .user, text: call.prompt)]
        guard call.question != nil else {
            return GenerationRequest(prompt: call.prompt, temperature: nil, topP: nil, messages: message).withOrigin(origin)
        }
        var request = GenerationRequest(prompt: call.prompt, maxTokens: questionMaxTokens, temperature: 0, topP: nil, messages: message)
        request.enableThinking = false
        return request.withOrigin(origin)
    }

    public static func label(of answer: SystemOneAnswer, _ question: SystemOneQuestion) -> String? {
        switch question.type {
        case .choice: answer.choice
        case .noul: answer.noul.map { $0 >= 0.5 ? "true" : "false" }
        case .score: answer.probabilities?.max { ($0.value, $1.key) < ($1.value, $0.key) }?.key
        }
    }

    public static func label(in reply: String, keys: [String]) -> String? {
        let lowered = keys.map { $0.lowercased() }
        let whole = reply.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters)).lowercased()
        if let exact = lowered.firstIndex(of: whole) { return keys[exact] }
        let words = reply.lowercased().split { !($0.isLetter || $0.isNumber || $0 == "_" || $0 == "-") }
        return words.lazy.compactMap { lowered.firstIndex(of: String($0)) }.first.map { keys[$0] }
    }

    public static func answers(
        _ route: Route, replies: [System2Reply], questions: [String: SystemOneQuestion], decision: SystemOneResponse?
    ) -> (answers: [String: PipelineAnswer], system2: PipelineSystem2?) {
        var answers = route.answers
        var system2: PipelineSystem2?
        for reply in replies {
            guard let id = reply.call.question, let question = questions[id] else {
                system2 = PipelineSystem2(model: reply.model, text: reply.text)
                continue
            }
            let label = label(in: reply.text, keys: SystemOneCore.options(question).keys)
            answers[id] = PipelineAnswer(answeredBy: .answer, label: label, decision: decision?.answers[id], text: reply.text)
        }
        return (answers, system2)
    }

    public static let perceivedSource = "pipeline-perceived"

    public static func skipped(_ model: String, ms: Int) -> PipelineStageReceipt {
        PipelineStageReceipt(stage: .decide, model: model, ms: ms, tokens: 0, escalated: true)
    }

    public static func decided(_ response: SystemOneResponse, ms: Int) -> PipelineStageReceipt {
        PipelineStageReceipt(
            stage: .decide, model: response.model, ms: ms, tokens: response.usage.prefixTokens + response.usage.suffixTokens,
            confidence: response.answers.values.compactMap(\.confidence).min(), escalated: response.escalate)
    }

    public static func receipt(
        _ stages: [PipelineStageReceipt], replies: [System2Reply], answers: [String: PipelineAnswer], ms: Int
    ) -> PipelineReceipt {
        return PipelineReceipt(
            stages: stages + replies.map(PipelineStageReceipt.init), answeredBy: answers.mapValues(\.answeredBy), escalated: answers.values.contains { $0.answeredBy == .answer },
            system2Calls: replies.count, ms: ms)
    }

    public static func milliseconds(_ duration: Duration) -> Int {
        let parts = duration.components
        return Int(parts.seconds * 1000 + parts.attoseconds / 1_000_000_000_000_000)
    }
}
