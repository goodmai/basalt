import Foundation

public struct SemanticTriplet: Sendable, Codable, Equatable, Hashable {
    public var subject: String
    public var predicate: String
    public var object: String
    public var confidence: Double
    public var source: String

    public init(subject: String, predicate: String, object: String, confidence: Double = 1.0, source: String = "verified") {
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = confidence
        self.source = source
    }

    public var normalizedKey: String {
        "\(subject.lowercased())|\(predicate.lowercased())|\(object.lowercased())"
    }
}

public struct KnowledgeGap: Sendable, Codable, Equatable {
    public var subject: String
    public var missingPredicates: [String]

    public init(subject: String, missingPredicates: [String]) {
        self.subject = subject
        self.missingPredicates = missingPredicates
    }
}

public struct SymbolicConflict: Sendable, Codable, Equatable {
    public var existing: SemanticTriplet
    public var incoming: SemanticTriplet
    public var reason: String

    public init(existing: SemanticTriplet, incoming: SemanticTriplet, reason: String) {
        self.existing = existing
        self.incoming = incoming
        self.reason = reason
    }
}

public struct SymbolicCommitReport: Sendable, Codable, Equatable {
    public var committed: [SemanticTriplet]
    public var skippedDuplicates: Int
    public var skippedLowConfidence: Int
    public var conflicts: [SymbolicConflict]

    public init(
        committed: [SemanticTriplet] = [],
        skippedDuplicates: Int = 0,
        skippedLowConfidence: Int = 0,
        conflicts: [SymbolicConflict] = []
    ) {
        self.committed = committed
        self.skippedDuplicates = skippedDuplicates
        self.skippedLowConfidence = skippedLowConfidence
        self.conflicts = conflicts
    }
}

public struct SymbolicKnowledgeGraph: Sendable, Equatable {
    public var functionalPredicates: Set<String>
    private var disjointPairs: Set<String>
    public private(set) var triplets: [SemanticTriplet]
    private var knownKeys: Set<String>

    public init(
        functionalPredicates: Set<String> = ["has_capital", "has_status", "has_diagnosis", "has_owner"],
        disjointPredicatePairs: [(String, String)] = [("is_allowed", "is_forbidden"), ("is_safe", "is_hazardous")]
    ) {
        self.functionalPredicates = Set(functionalPredicates.map { $0.lowercased() })
        self.disjointPairs = Set(disjointPredicatePairs.flatMap { a, b in
            let la = a.lowercased()
            let lb = b.lowercased()
            return ["\(la)|\(lb)", "\(lb)|\(la)"]
        })
        self.triplets = []
        self.knownKeys = []
    }

    public func query(subject: String, predicate: String? = nil) -> [SemanticTriplet] {
        let targetSubj = subject.lowercased()
        let targetPred = predicate?.lowercased()
        return triplets.filter { triplet in
            guard triplet.subject.lowercased() == targetSubj else { return false }
            if let targetPred {
                return triplet.predicate.lowercased() == targetPred
            }
            return true
        }
    }

    public func detectGap(subject: String, requiredPredicates: [String]) -> KnowledgeGap? {
        let existing = Set(query(subject: subject).map { $0.predicate.lowercased() })
        let missing = requiredPredicates.filter { !existing.contains($0.lowercased()) }
        return missing.isEmpty ? nil : KnowledgeGap(subject: subject, missingPredicates: missing)
    }

    public func buildInternalRAGPrompt(for gap: KnowledgeGap) -> String {
        let predList = gap.missingPredicates.joined(separator: ", ")
        return "Provide factual semantic triplets for subject '\(gap.subject)' covering predicates [\(predList)]. Output one triplet per line as: subject | predicate | object"
    }

    public static func parseTriplets(
        from text: String,
        source: String = "internal_rag",
        defaultConfidence: Double = 0.9
    ) -> [SemanticTriplet] {
        var result: [SemanticTriplet] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let cleaned = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "()-•* "))
            let sep: Character = cleaned.contains("|") ? "|" : ","
            let parts = cleaned.split(separator: sep).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard parts.count >= 3, !parts[0].isEmpty, !parts[1].isEmpty, !parts[2].isEmpty else { continue }
            let conf = parts.count >= 4 ? (Double(parts[3]) ?? defaultConfidence) : defaultConfidence
            result.append(SemanticTriplet(subject: parts[0], predicate: parts[1], object: parts[2], confidence: conf, source: source))
        }
        return result
    }

    public mutating func verifyAndCommit(
        _ candidates: [SemanticTriplet],
        minConfidence: Double = 0.7
    ) -> SymbolicCommitReport {
        var report = SymbolicCommitReport()
        for candidate in candidates {
            guard candidate.confidence >= minConfidence else {
                report.skippedLowConfidence += 1
                continue
            }
            let key = candidate.normalizedKey
            if knownKeys.contains(key) {
                report.skippedDuplicates += 1
                continue
            }
            if let conflict = findConflict(for: candidate) {
                report.conflicts.append(conflict)
                continue
            }
            triplets.append(candidate)
            knownKeys.insert(key)
            report.committed.append(candidate)
        }
        return report
    }

    private func findConflict(for candidate: SemanticTriplet) -> SymbolicConflict? {
        let subj = candidate.subject.lowercased()
        let pred = candidate.predicate.lowercased()
        let obj = candidate.object.lowercased()
        for existing in triplets where existing.subject.lowercased() == subj {
            let ePred = existing.predicate.lowercased()
            let eObj = existing.object.lowercased()
            if functionalPredicates.contains(pred), ePred == pred, eObj != obj {
                return SymbolicConflict(existing: existing, incoming: candidate, reason: "functional_predicate_violation")
            }
            if eObj == obj, disjointPairs.contains("\(ePred)|\(pred)") {
                return SymbolicConflict(existing: existing, incoming: candidate, reason: "disjoint_predicate_violation")
            }
        }
        return nil
    }
}
