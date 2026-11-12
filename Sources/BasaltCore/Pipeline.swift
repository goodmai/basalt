import Foundation

public enum PipelineAttachment: Sendable {
    case document(OCRDocument)
    case audio(AudioClip)

    public init(kind: String, data: Data, format: String?) throws(BasaltError) {
        switch kind {
        case "pdf", "image":
            self = .document(try OCRDocument(data: data))
        case "audio":
            guard let format else { throw .invalidRequestStructure(details: "pipeline: an audio attachment needs format, such as wav or m4a") }
            self = .audio(try AudioClip(encoded: data, format: format))
        default:
            throw Self.unknown(kind)
        }
    }

    public init(kind: String, encoded: String, format: String?) throws(BasaltError) {
        switch kind {
        case "pdf", "image":
            self = .document(try OCRDocument(encoded: encoded))
        case "audio":
            guard encoded.utf8.count <= (AudioClip.maxBytes + 2) / 3 * 4 else {
                throw .invalidRequestStructure(details: "pipeline: the audio is over the limit of \(AudioClip.maxBytes >> 20) MB")
            }
            guard let data = Data(base64Encoded: encoded, options: .ignoreUnknownCharacters) else {
                throw .invalidRequestStructure(details: "pipeline: attachment data must be base64")
            }
            try self.init(kind: kind, data: data, format: format)
        default:
            throw Self.unknown(kind)
        }
    }

    static func unknown(_ kind: String) -> BasaltError {
        .invalidRequestStructure(details: "pipeline: attachment kind must be pdf, image or audio, got '\(kind.prefix(16))'")
    }
}

public struct PipelineRequest: Sendable {
    public static let maxAttachments = 8

    public let attachments: [PipelineAttachment]
    public let text: String?
    public let questions: [String: SystemOneQuestion]
    public let task: String?
    public let pages: String?

    static func audioSeconds(_ attachments: [PipelineAttachment]) -> Double {
        attachments.reduce(0) { total, attachment in
            guard case .audio(let clip) = attachment else { return total }
            return total + Double(clip.samples.count) / AudioClip.sampleRate
        }
    }

    public init(
        attachments: [PipelineAttachment], text: String?, questions: [String: SystemOneQuestion]?, task: String?, pages: String?
    ) throws(BasaltError) {
        func refuse(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "pipeline: \(reason)") }
        let (text, task, questions) = (Self.nonBlank(text), Self.nonBlank(task), questions ?? [:])
        guard attachments.count <= Self.maxAttachments else { throw refuse("a request carries at most \(Self.maxAttachments) attachments") }
        guard !attachments.isEmpty || text != nil else { throw refuse("send attachments, text or both") }
        guard Self.audioSeconds(attachments) <= AudioClip.maxSeconds else { throw refuse("the audio attachments together last over \(Int(AudioClip.maxSeconds)) s") }
        guard !questions.isEmpty || task != nil else { throw refuse("send questions for System One, a task for System 2, or both") }
        guard [text, task].allSatisfy({ ($0?.count ?? 0) <= SystemOneCore.maxStateCharacters }) else {
            throw refuse("text and task hold at most \(SystemOneCore.maxStateCharacters) characters each")
        }
        guard questions.count <= SystemOneCore.maxQuestions else { throw refuse("at most \(SystemOneCore.maxQuestions) questions") }
        for (name, question) in questions {
            guard !name.trimmingCharacters(in: .whitespaces).isEmpty else { throw refuse("question IDs cannot be blank") }
            try SystemOneCore.validate(question, name: name)
        }
        if let pages { _ = try OCRPageSelection(pages) }
        (self.attachments, self.text, self.questions, self.task, self.pages) = (attachments, text, questions, task, pages)
    }

    static func nonBlank(_ text: String?) -> String? {
        text.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
    }
}

public enum PipelineStage: String, Codable, Sendable, Equatable {
    case perceive, decide, answer
}

public struct PipelinePage: Codable, Sendable, Equatable {
    public let attachment: Int
    public let page: Int
    public let markdown: String

    public init(attachment: Int, page: Int, markdown: String) {
        (self.attachment, self.page, self.markdown) = (attachment, page, markdown)
    }
}

public struct PipelinePerceived: Codable, Sendable, Equatable {
    public var pages: [PipelinePage]?
    public var transcript: String?

    public init(pages: [PipelinePage]? = nil, transcript: String? = nil) {
        (self.pages, self.transcript) = (pages, transcript)
    }
}

public struct PipelineAnswer: Codable, Sendable, Equatable {
    public let answeredBy: PipelineStage
    public let label: String?
    public let decision: SystemOneAnswer?
    public let text: String?

    public init(answeredBy: PipelineStage, label: String?, decision: SystemOneAnswer?, text: String? = nil) {
        (self.answeredBy, self.label, self.decision, self.text) = (answeredBy, label, decision, text)
    }

    enum CodingKeys: String, CodingKey {
        case answeredBy = "answered_by", label, decision, text
    }
}

public struct PipelineSystem2: Codable, Sendable, Equatable {
    public let model: String
    public let text: String
}

public struct PipelineStageReceipt: Codable, Sendable, Equatable {
    public let stage: PipelineStage
    public let model: String
    public let ms: Int
    public let tokens: Int
    public let confidence: Double?
    public let escalated: Bool
    public let question: String?
    public var route: Int?
    public var cost: Double?
    public var skipped: [String]?

    public init(stage: PipelineStage, model: String, ms: Int, tokens: Int, confidence: Double? = nil, escalated: Bool = false) {
        (self.stage, self.model, self.ms, self.tokens) = (stage, model, ms, tokens)
        (self.confidence, self.escalated, self.question) = (confidence, escalated, nil)
    }

    public init(_ reply: PipelineCore.System2Reply) {
        (stage, model, ms, tokens) = (.answer, reply.model, reply.ms, reply.tokens)
        (confidence, escalated, question) = (nil, false, reply.call.question)
        (route, cost, skipped) = (reply.route, reply.cost, reply.skipped.isEmpty ? nil : reply.skipped)
    }
}

public struct PipelineReceipt: Codable, Sendable, Equatable {
    public let stages: [PipelineStageReceipt]
    public let answeredBy: [String: PipelineStage]
    public let escalated: Bool
    public let system2Calls: Int
    public let ms: Int
    public var swapMs: Int?

    enum CodingKeys: String, CodingKey {
        case stages, escalated, ms
        case answeredBy = "answered_by", system2Calls = "system2_calls", swapMs = "swap_ms"
    }
}

public struct PipelineResponse: Codable, Sendable, Equatable {
    public let perceived: PipelinePerceived
    public let answers: [String: PipelineAnswer]
    public let system2: PipelineSystem2?
    public let receipt: PipelineReceipt

    public init(perceived: PipelinePerceived, answers: [String: PipelineAnswer], system2: PipelineSystem2?, receipt: PipelineReceipt) {
        (self.perceived, self.answers, self.system2, self.receipt) = (perceived, answers, system2, receipt)
    }
}

extension PipelineAttachment {
    public var media: DecisionMedia? {
        switch self {
        case .document(.pdf(let data)): DecisionMedia(kind: "pdf", mimeType: "application/pdf", data: data)
        case .document(.image(let image)): DecisionMedia(kind: "image", mimeType: image.mime, data: image.data)
        case .audio(let clip): DecisionMedia(kind: "audio", mimeType: "audio/wav", data: clip.wav)
        }
    }
}

extension AudioClip {
    public var wav: Data {
        let pcm = samples.map { Int16(max(-1, min(1, $0)) * Float(Int16.max)) }
        let rate = UInt32(Self.sampleRate)
        var data = Data("RIFF".utf8)
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        append(UInt32(36 + pcm.count * 2))
        data.append(contentsOf: Data("WAVEfmt ".utf8))
        append(UInt32(16)); append(UInt16(1)); append(UInt16(1)); append(rate); append(rate * 2); append(UInt16(2)); append(UInt16(16))
        data.append(contentsOf: Data("data".utf8))
        append(UInt32(pcm.count * 2))
        for sample in pcm { append(sample) }
        return data
    }
}
