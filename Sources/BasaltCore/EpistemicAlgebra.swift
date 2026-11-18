import Foundation

public struct EvidencePair: Sendable, Codable, Equatable {
    public var support: Double
    public var refutation: Double

    public init(support: Double, refutation: Double) {
        self.support = Self.clamp(support)
        self.refutation = Self.clamp(refutation)
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawSupport = try container.decode(Double.self, forKey: .support)
        let rawRefutation = try container.decode(Double.self, forKey: .refutation)
        self.init(support: rawSupport, refutation: rawRefutation)
    }

    public static let none = EvidencePair(support: 0, refutation: 0)

    public static func clamp(_ value: Double) -> Double {
        guard value.isFinite else { return 0 }
        return min(max(value, 0), 1)
    }

    public var negated: EvidencePair {
        EvidencePair(support: refutation, refutation: support)
    }

    public func truthJoin(_ other: EvidencePair) -> EvidencePair {
        EvidencePair(support: max(support, other.support), refutation: min(refutation, other.refutation))
    }

    public func truthMeet(_ other: EvidencePair) -> EvidencePair {
        EvidencePair(support: min(support, other.support), refutation: max(refutation, other.refutation))
    }

    public func knowledgeJoin(_ other: EvidencePair) -> EvidencePair {
        EvidencePair(support: max(support, other.support), refutation: max(refutation, other.refutation))
    }

    public func knowledgeMeet(_ other: EvidencePair) -> EvidencePair {
        EvidencePair(support: min(support, other.support), refutation: min(refutation, other.refutation))
    }

    public func belnap(supportThreshold: Double = 0.5, refutationThreshold: Double = 0.5) -> BelnapState {
        BelnapState.project(self, supportThreshold: supportThreshold, refutationThreshold: refutationThreshold)
    }

    enum CodingKeys: String, CodingKey {
        case support, refutation
    }
}

public struct EpistemicThresholds: Sendable, Codable, Equatable, Hashable {
    public static let `default` = EpistemicThresholds()

    public var supportThreshold: Double
    public var refutationThreshold: Double
    public var entropyCeiling: Double

    public init(
        supportThreshold: Double = 0.5,
        refutationThreshold: Double = 0.5,
        entropyCeiling: Double = 4.0
    ) {
        self.supportThreshold = EvidencePair.clamp(supportThreshold)
        self.refutationThreshold = EvidencePair.clamp(refutationThreshold)
        self.entropyCeiling = entropyCeiling.isFinite && entropyCeiling > 0 ? entropyCeiling : 4.0
    }
}

public enum BelnapState: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    case neither
    case `true`
    case `false`
    case both

    public init(bitPair: UInt8) {
        switch bitPair & 0x03 {
        case 0: self = .neither
        case 1: self = .false
        case 2: self = .true
        default: self = .both
        }
    }

    public var bitPair: UInt8 {
        switch self {
        case .neither: 0
        case .false: 1
        case .true: 2
        case .both: 3
        }
    }

    public var flags: (supported: Bool, refuted: Bool) {
        switch self {
        case .neither: (false, false)
        case .true: (true, false)
        case .false: (false, true)
        case .both: (true, true)
        }
    }

    public static func from(supported: Bool, refuted: Bool) -> BelnapState {
        switch (supported, refuted) {
        case (false, false): .neither
        case (true, false): .true
        case (false, true): .false
        case (true, true): .both
        }
    }

    public static func project(
        _ evidence: EvidencePair,
        supportThreshold: Double = 0.5,
        refutationThreshold: Double = 0.5
    ) -> BelnapState {
        from(
            supported: evidence.support >= EvidencePair.clamp(supportThreshold),
            refuted: evidence.refutation >= EvidencePair.clamp(refutationThreshold)
        )
    }

    public static func project(
        _ evidence: EvidencePair,
        thresholds: EpistemicThresholds
    ) -> BelnapState {
        project(
            evidence,
            supportThreshold: thresholds.supportThreshold,
            refutationThreshold: thresholds.refutationThreshold
        )
    }

    public var negated: BelnapState {
        let pair = flags
        return Self.from(supported: pair.refuted, refuted: pair.supported)
    }

    public var truthNegated: BelnapState { negated }

    public func truthJoin(_ other: BelnapState) -> BelnapState {
        let (lhs, rhs) = (flags, other.flags)
        return Self.from(supported: lhs.supported || rhs.supported, refuted: lhs.refuted && rhs.refuted)
    }

    public func truthMeet(_ other: BelnapState) -> BelnapState {
        let (lhs, rhs) = (flags, other.flags)
        return Self.from(supported: lhs.supported && rhs.supported, refuted: lhs.refuted || rhs.refuted)
    }

    public func knowledgeJoin(_ other: BelnapState) -> BelnapState {
        BelnapState(bitPair: bitPair | other.bitPair)
    }

    public func knowledgeMeet(_ other: BelnapState) -> BelnapState {
        BelnapState(bitPair: bitPair & other.bitPair)
    }

    public func infoJoin(_ other: BelnapState) -> BelnapState {
        knowledgeJoin(other)
    }

    public func infoMeet(_ other: BelnapState) -> BelnapState {
        knowledgeMeet(other)
    }

    public func truthPrecedesOrEquals(_ other: BelnapState) -> Bool {
        let (lhs, rhs) = (flags, other.flags)
        return (!lhs.supported || rhs.supported) && (!rhs.refuted || lhs.refuted)
    }

    public func knowledgePrecedesOrEquals(_ other: BelnapState) -> Bool {
        (bitPair & other.bitPair) == bitPair
    }

    public func infoPrecedesOrEquals(_ other: BelnapState) -> Bool {
        knowledgePrecedesOrEquals(other)
    }
}

public struct BelnapGeneralizedState: Sendable, Codable, Equatable, Hashable, CaseIterable {
    private static let truthMask: UInt8 = 0x0C
    private static let nonTruthMask: UInt8 = 0x03
    private static let falsityMask: UInt8 = 0x0A
    private static let nonFalsityMask: UInt8 = 0x05

    public static let empty = BelnapGeneralizedState(rawBits: 0x0)
    public static let all = BelnapGeneralizedState(rawBits: 0xF)
    public static let neither = BelnapGeneralizedState(.neither)
    public static let `false` = BelnapGeneralizedState(.false)
    public static let `true` = BelnapGeneralizedState(.true)
    public static let both = BelnapGeneralizedState(.both)

    public static var allCases: [BelnapGeneralizedState] {
        (0 ..< 16).map { BelnapGeneralizedState(rawBits: UInt8($0)) }
    }

    public let rawBits: UInt8

    public init(rawBits: UInt8) {
        self.rawBits = rawBits & 0x0F
    }

    public init(_ single: BelnapState) {
        self.rawBits = 1 << single.bitPair
    }

    public init<S: Sequence>(_ states: S) where S.Element == BelnapState {
        var mask: UInt8 = 0
        for state in states { mask |= (1 << state.bitPair) }
        self.rawBits = mask & 0x0F
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        rawBits = (try container.decode(UInt8.self)) & 0x0F
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawBits)
    }

    public func contains(_ state: BelnapState) -> Bool {
        (rawBits & (1 << state.bitPair)) != 0
    }

    public func infoMeet(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        BelnapGeneralizedState(rawBits: rawBits & other.rawBits)
    }

    public func infoJoin(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        BelnapGeneralizedState(rawBits: rawBits | other.rawBits)
    }

    public static func & (lhs: BelnapGeneralizedState, rhs: BelnapGeneralizedState) -> BelnapGeneralizedState {
        lhs.infoMeet(rhs)
    }

    public static func | (lhs: BelnapGeneralizedState, rhs: BelnapGeneralizedState) -> BelnapGeneralizedState {
        lhs.infoJoin(rhs)
    }

    public func truthMeet(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        let positive = (rawBits & other.rawBits) & Self.truthMask
        let negative = (rawBits | other.rawBits) & Self.nonTruthMask
        return BelnapGeneralizedState(rawBits: positive | negative)
    }

    public func truthJoin(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        let positive = (rawBits | other.rawBits) & Self.truthMask
        let negative = (rawBits & other.rawBits) & Self.nonTruthMask
        return BelnapGeneralizedState(rawBits: positive | negative)
    }

    public func falsityMeet(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        let positive = (rawBits & other.rawBits) & Self.falsityMask
        let negative = (rawBits | other.rawBits) & Self.nonFalsityMask
        return BelnapGeneralizedState(rawBits: positive | negative)
    }

    public func falsityJoin(_ other: BelnapGeneralizedState) -> BelnapGeneralizedState {
        let positive = (rawBits | other.rawBits) & Self.falsityMask
        let negative = (rawBits & other.rawBits) & Self.nonFalsityMask
        return BelnapGeneralizedState(rawBits: positive | negative)
    }

    public var truthNegated: BelnapGeneralizedState {
        BelnapGeneralizedState(rawBits: ((rawBits & Self.nonTruthMask) << 2) | ((rawBits & Self.truthMask) >> 2))
    }

    public var falsityNegated: BelnapGeneralizedState {
        BelnapGeneralizedState(rawBits: ((rawBits & Self.nonFalsityMask) << 1) | ((rawBits & Self.falsityMask) >> 1))
    }

    public var infoNegated: BelnapGeneralizedState {
        BelnapGeneralizedState(rawBits: ~rawBits)
    }

    public func infoPrecedesOrEquals(_ other: BelnapGeneralizedState) -> Bool {
        infoJoin(other) == other
    }

    public func truthPrecedesOrEquals(_ other: BelnapGeneralizedState) -> Bool {
        truthJoin(other) == other
    }

    public func falsityPrecedesOrEquals(_ other: BelnapGeneralizedState) -> Bool {
        falsityJoin(other) == other
    }
}

public enum EpistemicVerdict: String, Sendable, Codable, Equatable, CaseIterable {
    case known
    case noEvidence
    case conflict
    case needsTool

    public static func project(
        state: BelnapState,
        needsToolScore: Double = 0,
        toolThreshold: Double = 0.5,
        hasTools: Bool = false
    ) -> EpistemicVerdict {
        let wantsTool = hasTools && EvidencePair.clamp(needsToolScore) >= EvidencePair.clamp(toolThreshold)
        switch state {
        case .both, .false:
            return .conflict
        case .true:
            return wantsTool ? .needsTool : .known
        case .neither:
            return wantsTool ? .needsTool : .noEvidence
        }
    }
}

public struct EpistemicGateSpec: Sendable, Codable, Equatable {
    public enum NoEvidencePolicy: String, Sendable, Codable, Equatable, CaseIterable {
        case abstain
        case escalate
    }

    public enum ConflictPolicy: String, Sendable, Codable, Equatable, CaseIterable {
        case report
        case escalate
    }

    public static let defaultAbstainMessage = "Insufficient evidence in the provided context to answer reliably."
    public static let defaultConflictMessage = "Conflicting or refuted evidence detected in the provided context."

    public var enabled: Bool
    public var supportThreshold: Double
    public var refutationThreshold: Double
    public var toolThreshold: Double
    public var onNoEvidence: NoEvidencePolicy
    public var onConflict: ConflictPolicy
    public var abstainMessage: String?

    public init(
        enabled: Bool = true,
        supportThreshold: Double = 0.5,
        refutationThreshold: Double = 0.5,
        toolThreshold: Double = 0.5,
        onNoEvidence: NoEvidencePolicy = .abstain,
        onConflict: ConflictPolicy = .report
    ) {
        self.enabled = enabled
        self.supportThreshold = supportThreshold
        self.refutationThreshold = refutationThreshold
        self.toolThreshold = toolThreshold
        self.onNoEvidence = onNoEvidence
        self.onConflict = onConflict
        self.abstainMessage = nil
    }

    public func withAbstainMessage(_ message: String?) -> EpistemicGateSpec {
        var copy = self
        copy.abstainMessage = message
        return copy
    }

    enum CodingKeys: String, CodingKey {
        case enabled
        case supportThreshold = "support_threshold"
        case refutationThreshold = "refutation_threshold"
        case toolThreshold = "tool_threshold"
        case onNoEvidence = "on_no_evidence"
        case onConflict = "on_conflict"
        case abstainMessage = "abstain_message"
    }
}

public enum EpistemicGateAction: String, Sendable, Codable, Equatable, CaseIterable {
    case proceed
    case abstain
    case reportConflict
    case escalate
    case routeTool
}

public struct EpistemicGateDecision: Sendable, Codable, Equatable {
    public var evidence: EvidencePair
    public var state: BelnapState
    public var verdict: EpistemicVerdict
    public var needsToolScore: Double
    public var action: EpistemicGateAction
    public var message: String?

    public init(
        evidence: EvidencePair,
        state: BelnapState,
        verdict: EpistemicVerdict,
        needsToolScore: Double,
        action: EpistemicGateAction,
        message: String? = nil
    ) {
        self.evidence = evidence
        self.state = state
        self.verdict = verdict
        self.needsToolScore = EvidencePair.clamp(needsToolScore)
        self.action = action
        self.message = message
    }

    public var shortCircuitsAnswer: Bool {
        action == .abstain || action == .reportConflict
    }

    enum CodingKeys: String, CodingKey {
        case evidence, state, verdict, action, message
        case needsToolScore = "needs_tool_score"
    }
}

public enum EpistemicGateCore {
    public static let supportID = "epistemic_support"
    public static let refutationID = "epistemic_refutation"
    public static let needsToolID = "epistemic_needs_tool"
    public static let questionIDs = [supportID, refutationID, needsToolID]

    public static func probes(includeTools: Bool = true) -> [String: SystemOneQuestion] {
        var result: [String: SystemOneQuestion] = [
            supportID: SystemOneQuestion(
                type: .noul,
                instructions: "The provided state contains sufficient direct evidence to answer the request accurately without guessing."
            ),
            refutationID: SystemOneQuestion(
                type: .noul,
                instructions: "The provided state contains contradictory facts or directly refutes a premise of the request."
            ),
        ]
        if includeTools {
            result[needsToolID] = SystemOneQuestion(
                type: .noul,
                instructions: "Answering the request accurately requires invoking an external tool or live retrieval beyond the provided state."
            )
        }
        return result
    }

    public static func mergedQuestions(
        _ userQuestions: [String: SystemOneQuestion],
        spec: EpistemicGateSpec?,
        includeTools: Bool = true
    ) -> [String: SystemOneQuestion] {
        guard let spec, spec.enabled else { return userQuestions }
        var merged = userQuestions
        for (key, value) in probes(includeTools: includeTools) { merged[key] = value }
        return merged
    }

    public static func userResponse(_ response: SystemOneResponse?, keeping userKeys: Set<String>) -> SystemOneResponse? {
        guard let response else { return nil }
        let filtered = response.answers.filter { userKeys.contains($0.key) }
        return SystemOneResponse(model: response.model, answers: filtered, usage: response.usage, calibrated: response.calibrated)
    }

    public static func reduce(
        _ response: SystemOneResponse,
        spec: EpistemicGateSpec,
        hasTools: Bool = false
    ) -> EpistemicGateDecision {
        let support = response.answers[supportID]?.noul ?? 0
        let refutation = response.answers[refutationID]?.noul ?? 0
        let toolScore = EvidencePair.clamp(response.answers[needsToolID]?.noul ?? 0)
        let evidence = EvidencePair(support: support, refutation: refutation)
        let state = BelnapState.project(
            evidence,
            supportThreshold: spec.supportThreshold,
            refutationThreshold: spec.refutationThreshold
        )
        let verdict = EpistemicVerdict.project(
            state: state,
            needsToolScore: toolScore,
            toolThreshold: spec.toolThreshold,
            hasTools: hasTools
        )
        let (action, message) = resolvePolicy(verdict: verdict, evidence: evidence, spec: spec)
        return EpistemicGateDecision(
            evidence: evidence,
            state: state,
            verdict: verdict,
            needsToolScore: toolScore,
            action: action,
            message: message
        )
    }

    static func resolvePolicy(
        verdict: EpistemicVerdict,
        evidence: EvidencePair,
        spec: EpistemicGateSpec
    ) -> (EpistemicGateAction, String?) {
        switch verdict {
        case .known:
            return (.proceed, nil)
        case .needsTool:
            return (.routeTool, nil)
        case .noEvidence:
            switch spec.onNoEvidence {
            case .abstain:
                let text = spec.abstainMessage?.trimmingCharacters(in: .whitespacesAndNewlines)
                return (.abstain, (text?.isEmpty == false ? text : nil) ?? EpistemicGateSpec.defaultAbstainMessage)
            case .escalate:
                return (.escalate, nil)
            }
        case .conflict:
            switch spec.onConflict {
            case .report:
                let summary = String(
                    format: "%@ (support=%.2f, refutation=%.2f)",
                    EpistemicGateSpec.defaultConflictMessage,
                    evidence.support,
                    evidence.refutation
                )
                return (.reportConflict, summary)
            case .escalate:
                return (.escalate, nil)
            }
        }
    }
}
