import Foundation

public struct PipelineChoices: Sendable, Equatable, Codable {
    public var chat: String?
    public var ocr: String?
    public var audio: String?
    public var decide: String?
    public var abstainBelow: Double?
    public var minMargin: Double?

    public init(chat: String? = nil, ocr: String? = nil, audio: String? = nil, decide: String? = nil, abstainBelow: Double? = nil, minMargin: Double? = nil) {
        (self.chat, self.ocr, self.audio, self.decide, self.abstainBelow, self.minMargin) = (chat, ocr, audio, decide, abstainBelow, minMargin)
    }

    public func overridden(by explicit: PipelineChoices) -> PipelineChoices {
        PipelineChoices(
            chat: explicit.chat ?? chat, ocr: explicit.ocr ?? ocr, audio: explicit.audio ?? audio, decide: explicit.decide ?? decide,
            abstainBelow: explicit.abstainBelow ?? abstainBelow, minMargin: explicit.minMargin ?? minMargin)
    }

    public var summary: String {
        let models = [("chat", chat), ("ocr", ocr), ("audio", audio), ("decide", decide)].compactMap { role, model in model.map { "\(role) \($0)" } }
        let gate = [("abstain below", abstainBelow), ("min margin", minMargin)].compactMap { name, value in value.map { "\(name) \($0)" } }
        return (models + gate).joined(separator: ", ")
    }
}

public struct PipelinePresetPlan: Sendable, Equatable, Codable {
    public let preset: PipelinePreset
    public let choices: PipelineChoices
    public let fits: Bool
    public let plan: PlanVerdict

    public var lines: [String] {
        ["pipeline preset \(preset.rawValue): \(choices.summary)"] + ComputePlan.describe(plan) + (plan.refusal.map { ["refused: \($0)"] } ?? [])
    }
}

public enum PipelinePreset: String, CaseIterable, Sendable, Codable {
    case documentQA = "document-qa"
    case voiceAgent = "voice-agent"
    case supportTriage = "support-triage"

    public struct Model: Sendable, Equatable {
        public let repo: String
        public let bytes: Int64
    }

    public static let gemma = Model(repo: "mlx-community/gemma-4-12B-it-qat-4bit", bytes: 10_987_772_430)
    public static let qwen = Model(repo: "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly", bytes: 11_771_374_457)
    public static let paddleOCR = Model(repo: "mlx-community/PaddleOCR-VL-1.5-8bit", bytes: 1_088_866_628)
    public static let nemotronTranscribe = Model(repo: "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX", bytes: 14_900_664_231)
    public static let kvHeadroomBytes: Int64 = 8_000_000_000
    public static let abstainBelow = 0.75
    public static let minMargin = 0.1

    public static var names: String { allCases.map(\.rawValue).joined(separator: ", ") }

    var ocr: Model? { self == .documentQA ? Self.paddleOCR : nil }
    var audio: Model? { self == .voiceAgent ? Self.nemotronTranscribe : nil }
    var decides: Bool { self != .voiceAgent }

    public var questions: [String: SystemOneQuestion] { self == .supportTriage ? Self.triageQuestions : [:] }

    func chat(budgetBytes: Int64) -> Model {
        let sidecars = [ocr, audio].compactMap { $0?.bytes }.reduce(0, +)
        return Self.qwen.bytes + sidecars + Self.kvHeadroomBytes <= budgetBytes ? Self.qwen : Self.gemma
    }

    public func choices(budgetBytes: Int64) -> PipelineChoices {
        PipelineChoices(
            chat: chat(budgetBytes: budgetBytes).repo, ocr: ocr?.repo, audio: audio?.repo, decide: decides ? "self" : nil,
            abstainBelow: decides ? Self.abstainBelow : nil, minMargin: decides ? Self.minMargin : nil)
    }

    public func plan(budgetBytes: Int64, power: PowerState = .desktop) -> PipelinePresetPlan {
        let chat = chat(budgetBytes: budgetBytes)
        let workloads = [Workload(name: chat.repo, role: "chat", unit: .gpu, kind: .decode, residentBytes: chat.bytes)]
            + [ocr, audio].compactMap { $0.map { Workload(name: $0.repo, role: "perceive", unit: .gpu, kind: .encoder, residentBytes: $0.bytes) } }
        let verdict = ComputePlan.admit(workloads, budgetBytes: budgetBytes, power: power)
        return PipelinePresetPlan(preset: self, choices: choices(budgetBytes: budgetBytes), fits: verdict.refusal == nil, plan: verdict)
    }

    public static let triageQuestions: [String: SystemOneQuestion] = [
        "department": SystemOneQuestion(type: .choice, instructions: "Which department should handle this ticket?", criteria: .named([
            "account": "Account: login, password, profile settings.",
            "billing": "Billing: invoices, charges, refunds of payments.",
            "returns": "Returns: sending items back and exchanges.",
            "sales": "Sales: prices, quotes and new purchases.",
            "shipping": "Shipping: delivery status, tracking, couriers.",
            "technical": "Technical support: the app or device does not work.",
        ])),
        "late": SystemOneQuestion(type: .noul, instructions: "The order was delivered later than promised.", criteria: .named([
            "false": "The order was delivered on time or early.", "true": "The order was delivered later than promised.",
        ])),
        "severity": SystemOneQuestion(type: .score, instructions: "How severe is the incident?", criteria: .levels([
            "No impact: cosmetic or informational.", "Minor: one user affected, workaround exists.",
            "Major: many users affected or a key feature down.", "Critical: outage or data loss.",
        ])),
    ]
}
