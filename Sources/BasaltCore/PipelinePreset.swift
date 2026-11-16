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

public enum PipelineRole: String, Sendable, Equatable, Codable {
    case chat, ocr, audio, decide
}

public struct PipelineMember: Sendable, Equatable, Codable {
    public var role: PipelineRole
    public var model: String
    public var bytes: Int64
    public var keep: KeepPolicy?
    public var unit: ComputeUnit?

    public init(_ role: PipelineRole, _ model: String, bytes: Int64 = 0, keep: KeepPolicy? = nil, unit: ComputeUnit? = nil) {
        self.role = role
        self.model = model
        self.bytes = bytes
        self.keep = keep
        self.unit = unit
    }

    public var workload: Workload? {
        let policy = keep ?? .resident
        switch role {
        case .chat: return Workload(name: model, role: "chat", unit: unit ?? .gpu, kind: .decode, residentBytes: bytes, keep: policy)
        case .ocr, .audio: return Workload(name: model, role: Workload.perceiveRole, unit: unit ?? .gpu, kind: .encoder, residentBytes: bytes, keep: policy)
        case .decide:
            guard model != "self", bytes > 0 else { return nil }
            return Workload(name: model, role: "decide", unit: inferredDecideUnit, kind: .encoder, residentBytes: bytes, keep: policy)
        }
    }

    var inferredDecideUnit: ComputeUnit {
        if let unit { return unit }
        if FileManager.default.fileExists(atPath: URL(fileURLWithPath: model).appending(component: "laya_decision.json").path) {
            return .ane
        }
        let parts = model.lowercased().split { !$0.isLetter && !$0.isNumber }
        return parts.contains { $0 == "ane" || $0 == "laya" || $0 == "coreai" } ? .ane : .gpu
    }
}

public struct PipelineProfile: Sendable, Equatable, Codable {
    public static let kvHeadroomBytes: Int64 = 8_000_000_000

    public var members: [PipelineMember]
    public var largerChat: PipelineMember?
    public var abstainBelow: Double?
    public var minMargin: Double?
    public var questions: [String: SystemOneQuestion]

    func resolved(budgetBytes: Int64) -> [PipelineMember] {
        guard let larger = largerChat else { return members }
        let sidecars = members.filter { $0.role != .chat }.reduce(0) { $0 + $1.bytes }
        guard larger.bytes + sidecars + Self.kvHeadroomBytes <= budgetBytes else { return members }
        return members.map { $0.role == .chat ? larger : $0 }
    }

    public func choices(budgetBytes: Int64) -> PipelineChoices {
        let members = resolved(budgetBytes: budgetBytes)
        func model(_ role: PipelineRole) -> String? { members.first { $0.role == role }?.model }
        return PipelineChoices(chat: model(.chat), ocr: model(.ocr), audio: model(.audio), decide: model(.decide), abstainBelow: abstainBelow, minMargin: minMargin)
    }

    public func plan(budgetBytes: Int64, power: PowerState = .desktop, memory: PipelineMemory = .auto) -> PlanVerdict {
        ComputePlan.admit(resolved(budgetBytes: budgetBytes).compactMap(\.workload), budgetBytes: budgetBytes, power: power, memory: memory)
    }
}

public enum PipelinePreset: String, CaseIterable, Sendable, Codable {
    case documentQA = "document-qa"
    case voiceAgent = "voice-agent"
    case supportTriage = "support-triage"

    public static var names: String { allCases.map(\.rawValue).joined(separator: ", ") }

    public var profile: PipelineProfile {
        switch self {
        case .documentQA: Self.bundled.documentQA
        case .voiceAgent: Self.bundled.voiceAgent
        case .supportTriage: Self.bundled.supportTriage
        }
    }

    public var questions: [String: SystemOneQuestion] { profile.questions }

    public func choices(budgetBytes: Int64) -> PipelineChoices { profile.choices(budgetBytes: budgetBytes) }

    public func plan(budgetBytes: Int64, power: PowerState = .desktop, memory: PipelineMemory = .auto) -> PipelinePresetPlan {
        let verdict = profile.plan(budgetBytes: budgetBytes, power: power, memory: memory)
        return PipelinePresetPlan(preset: self, choices: choices(budgetBytes: budgetBytes), fits: verdict.refusal == nil, plan: verdict)
    }
}
