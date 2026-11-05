import Foundation
import BasaltCore

public enum HarnessComponentVerdict: String, Codable, Sendable {
    case helped
    case hurt
    case noObservedEffect = "no observed effect"
    case confounded
}

public struct HarnessComponentAttribution: Codable, Sendable, Equatable {
    public let candidateId: String
    public let component: HarnessComponentKind
    public let paths: [String]
    public let verdict: HarnessComponentVerdict
}

enum HarnessDiff {
    static func changedComponents(base: HarnessTree, candidate: HarnessTree, spec: HarnessSurfaceSpec) -> [HarnessComponentKind: [String]] {
        var grouped: [HarnessComponentKind: [String]] = [:]
        for path in Set(base.files.keys).union(candidate.files.keys).sorted() where base.files[path] != candidate.files[path] {
            grouped[spec.classify(path: path), default: []].append(path)
        }
        return grouped
    }

    static func attribute(
        candidateId: String, base: HarnessTree, candidate: HarnessTree, comparedToBase: Int, spec: HarnessSurfaceSpec
    ) -> [HarnessComponentAttribution] {
        let changed = changedComponents(base: base, candidate: candidate, spec: spec)
        let verdict: HarnessComponentVerdict = changed.count > 1 ? .confounded
            : comparedToBase > 0 ? .helped : comparedToBase < 0 ? .hurt : .noObservedEffect
        return changed.keys.sorted { $0.rawValue < $1.rawValue }.map {
            HarnessComponentAttribution(candidateId: candidateId, component: $0, paths: changed[$0] ?? [], verdict: verdict)
        }
    }
}

public struct HarnessRunView: Codable, Sendable, Equatable {
    public let candidateId: String
    public let deliverable: String
    public let delivered: Bool
    public let conforms: Bool
    public let loopFree: Bool
    public let promptTokens: Int
    public let cachedPromptTokens: Int

    init(candidateId: String, response: GenerationResponse, task: HarnessTask) {
        let text = HarnessText.withoutThinking(response.generatedText).trimmingCharacters(in: .whitespacesAndNewlines)
        self.candidateId = candidateId
        self.deliverable = text
        self.delivered = !text.isEmpty && response.finishReason == .stop
        let lower = text.lowercased()
        self.conforms = task.requiredTerms.allSatisfy { lower.contains($0.lowercased()) }
            && (!task.expectsJSON || HarnessText.jsonObject(in: text) != nil)
        self.loopFree = !HarnessText.repeatsLine(text, times: 3)
        self.promptTokens = response.promptTokens
        self.cachedPromptTokens = response.cachedPromptTokens
    }

    var measured: [Bool] { [delivered, conforms, loopFree] }

    func compareMeasured(_ other: HarnessRunView) -> Int {
        for (mine, theirs) in zip(measured, other.measured) where mine != theirs {
            return mine ? 1 : -1
        }
        return 0
    }
}

public struct HarnessSelection: Codable, Sendable, Equatable {
    public let winnerId: String
    public let finalists: [String]
    public let judged: Bool
    public let judgeFallback: Bool
}

enum ContrastiveSelector {
    static let labels = (UnicodeScalar("A").value...UnicodeScalar("Z").value).compactMap { UnicodeScalar($0).map(String.init) }

    static func finalists(_ runs: [HarnessRunView]) -> [HarnessRunView] {
        guard let best = runs.max(by: { $0.compareMeasured($1) < 0 }) else { return [] }
        return runs.filter { $0.compareMeasured(best) == 0 }
    }

    static func anonymised(_ runs: [HarnessRunView], seed: UInt64) -> [(label: String, run: HarnessRunView)] {
        var generator = SplitMix64(seed: seed)
        return Array(zip(labels, runs.shuffled(using: &generator))).map { ($0, $1) }
    }

    static func judgePrompt(task: HarnessTask, candidates: [(label: String, run: HarnessRunView)]) -> String {
        var parts = [
            "Several anonymous deliverables answer the same task. Judge only their text, in this order: fidelity to the task, internal consistency, completeness and edge cases. The labels are random.",
            "<task>\n\(HarnessText.inert(task.prompt))\n</task>"
        ]
        parts += candidates.map { "<deliverable label=\"\($0.label)\">\n\(HarnessText.inert($0.run.deliverable))\n</deliverable>" }
        parts.append("Reply with exactly one line: \(replyPrefix)<label>")
        return parts.joined(separator: "\n\n")
    }

    static let replyPrefix = "BEST: "

    static func parseLabel(_ text: String, labels: [String]) -> String? {
        let cleaned = HarnessText.withoutThinking(text)
        guard let match = cleaned.range(of: #"BEST:\s*([A-Z])\b"#, options: [.regularExpression, .backwards]) else { return nil }
        guard let last = cleaned[match].last else { return nil }
        let label = String(last)
        return labels.contains(label) ? label : nil
    }

    static func seed(for slug: String) -> UInt64 {
        slug.utf8.reduce(14_695_981_039_346_656_037) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
    }

    static func selection(
        finalists: [HarnessRunView], candidates: [(label: String, run: HarnessRunView)], reply: String
    ) -> HarnessSelection {
        let ids = finalists.map(\.candidateId)
        if let label = parseLabel(reply, labels: candidates.map(\.label)),
           let chosen = candidates.first(where: { $0.label == label }) {
            return HarnessSelection(winnerId: chosen.run.candidateId, finalists: ids, judged: true, judgeFallback: false)
        }
        let fallback = finalists.first { $0.candidateId.hasPrefix("base_") } ?? finalists[0]
        return HarnessSelection(winnerId: fallback.candidateId, finalists: ids, judged: true, judgeFallback: true)
    }
}

struct HarnessCandidate: Sendable, Equatable {
    let id: String
    let tree: HarnessTree
    let delta: [String: String]

    var rolloutId: String { id.hasPrefix("base_") ? id : "probe_\(id)" }
}

struct HarnessVerdict: Sendable, Equatable {
    let selection: HarnessSelection
    let winner: HarnessRunView
    let delta: [String: String]
    let attributions: [HarnessComponentAttribution]
    let confidence: String

    var lesson: String {
        let changed = delta.keys.sorted().joined(separator: ", ")
        let how = selection.judged ? (selection.judgeFallback ? "tie, judge reply unusable" : "judged among \(selection.finalists.count) finalists") : "only finalist on measured stages"
        return String("""
        Winner: \(winner.candidateId) (\(how)).
        Changed files: \(changed.isEmpty ? "none" : changed).
        Measured: delivered=\(winner.delivered) conforms=\(winner.conforms) loopFree=\(winner.loopFree).

        Deliverable:
        \(winner.deliverable)
        """.prefix(8000))
    }
}

struct HarnessSearch: Sendable {
    let task: HarnessTask
    let base: HarnessTree
    let spec: HarnessSurfaceSpec
    let view: HarnessProposal.HarnessView
    let baseReport: HarnessValidationReport
    let skipped: [HarnessSkippedFile]
    let maxDeltaBytes: Int
    var ledger: HarnessLedger
    var warnings: [String]
    private(set) var runs: [HarnessRunView] = []
    private(set) var trees: [String: HarnessTree] = [:]
    private(set) var deltas: [String: [String: String]] = [:]
    private(set) var rejected: [HarnessRejection] = []
    private(set) var accepted: [HarnessTree] = []

    init(
        task: HarnessTask, base: HarnessTree, spec: HarnessSurfaceSpec, k: Int, skipped: [HarnessSkippedFile], maxDeltaBytes: Int
    ) throws {
        self.ledger = try HarnessLedger(k: k)
        self.task = task
        self.base = base
        self.spec = spec
        self.view = HarnessProposal.view(base, spec: spec)
        self.baseReport = HarnessValidator.validate(base, spec: spec)
        self.skipped = skipped
        self.maxDeltaBytes = maxDeltaBytes
        self.warnings = skipped.map { "skipped \($0)" }
    }

    mutating func reject(_ id: String, reasons: [String]) {
        rejected.append(HarnessRejection(candidateId: id, reasons: reasons))
    }

    mutating func admit(_ id: String, delta: [String: String]?) -> HarnessCandidate? {
        guard let delta else {
            reject(id, reasons: ["proposer reply is not a JSON object with string \"files\""])
            return nil
        }
        let admission = HarnessProposal.admit(
            delta, base: base, baseReport: baseReport, siblings: accepted, spec: spec,
            bounds: HarnessProposal.Bounds(truncated: view.truncated, skipped: skipped, maxDeltaBytes: maxDeltaBytes)
        )
        switch admission {
        case .failure(let rejection):
            reject(id, reasons: rejection.reasons)
            return nil
        case .success(let admitted):
            accepted.append(admitted.tree)
            warnings += admitted.warnings.map { "\(id): \($0)" }
            return HarnessCandidate(id: id, tree: admitted.tree, delta: delta)
        }
    }

    mutating func record(_ candidate: HarnessCandidate, response: GenerationResponse) {
        runs.append(HarnessRunView(candidateId: candidate.id, response: response, task: task))
        trees[candidate.id] = candidate.tree
        deltas[candidate.id] = candidate.delta
    }

    var bestBase: HarnessRunView {
        runs.filter { $0.candidateId.hasPrefix("base_") }.max { $0.compareMeasured($1) < 0 }!
    }

    var composeSources: [[String: String]] {
        let best = bestBase
        return runs.filter { !$0.candidateId.hasPrefix("base_") && $0.compareMeasured(best) >= 0 }
            .sorted { $0.compareMeasured($1) > 0 }
            .compactMap { deltas[$0.candidateId] }
    }

    static func union(_ sources: [[String: String]]) -> [String: String]? {
        sources.isEmpty ? nil : sources.reduce(into: [:]) { $0.merge($1) { first, _ in first } }
    }

    func verdict(selection: HarnessSelection) -> HarnessVerdict {
        let best = bestBase
        let winner = runs.first { $0.candidateId == selection.winnerId }!
        let attributions = runs.filter { !$0.candidateId.hasPrefix("base_") }.flatMap { run in
            HarnessDiff.attribute(
                candidateId: run.candidateId, base: base, candidate: trees[run.candidateId]!,
                comparedToBase: run.compareMeasured(best), spec: spec
            )
        }
        let confidence = winner.candidateId.hasPrefix("base_") ? "low" : winner.compareMeasured(best) > 0 ? "high" : "medium"
        return HarnessVerdict(
            selection: selection, winner: winner, delta: deltas[winner.candidateId] ?? [:],
            attributions: attributions, confidence: confidence
        )
    }

    func result(verdict: HarnessVerdict, librarySlug: String?) -> HarnessOptimizationResult {
        HarnessOptimizationResult(
            taskSlug: task.slug,
            winningCandidateId: verdict.winner.candidateId,
            winningDeliverable: verdict.winner.deliverable,
            winningDelta: verdict.delta,
            selection: verdict.selection,
            runs: runs,
            rejected: rejected,
            attributions: verdict.attributions,
            ledger: ledger,
            promptTokens: runs.reduce(0) { $0 + $1.promptTokens },
            cachedPromptTokens: runs.reduce(0) { $0 + $1.cachedPromptTokens },
            librarySlug: librarySlug,
            warnings: warnings
        )
    }
}
