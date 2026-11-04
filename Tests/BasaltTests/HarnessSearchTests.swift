import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltHarness

@Suite("Harness search value core")
struct HarnessSearchTests {
    static let base = HarnessTree(files: ["CLAUDE.md": "Answer precisely."])
    static let task = HarnessTask(slug: "audit-repo", prompt: "Audit the package.", domain: "swift", requiredTerms: ["SUMMARY"])

    static func response(_ text: String) -> GenerationResponse {
        GenerationResponse(
            generatedText: text, promptTokens: 10, completionTokens: 1, tokensPerSecond: 1, generationTime: 1,
            timeToFirstToken: 1, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop
        )
    }

    static func candidate(_ id: String, _ delta: [String: String]) -> HarnessCandidate {
        HarnessCandidate(id: id, tree: base.applying(delta: delta), delta: delta)
    }

    static func searched() throws -> HarnessSearch {
        var search = try HarnessSearch(task: task, base: base, spec: .claude, k: 1, skipped: [], maxDeltaBytes: 64 * 1024)
        search.record(candidate("base_r0", [:]), response: response("Targets checked."))
        search.record(candidate("inv_0", ["CLAUDE.md": "inv"]), response: response("Targets checked.\nSUMMARY: ok"))
        search.record(candidate("dom_0", ["CLAUDE.md": "dom"]), response: response(""))
        return search
    }

    @Test("Compose sources are the candidates at least as good as the best base run, best first")
    func composeSources() throws {
        let search = try Self.searched()
        #expect(search.bestBase.candidateId == "base_r0")
        #expect(search.composeSources == [["CLAUDE.md": "inv"]])
        #expect(search.result(verdict: search.verdict(selection: HarnessSelection(
            winnerId: "inv_0", finalists: ["inv_0"], judged: false, judgeFallback: false
        )), librarySlug: nil).promptTokens == 30)
    }

    @Test("The compose fallback is the union of the sources, the earlier source winning a conflict")
    func union() {
        #expect(HarnessSearch.union([]) == nil)
        #expect(HarnessSearch.union([["a": "1", "b": "1"], ["b": "2", "c": "2"]]) == ["a": "1", "b": "1", "c": "2"])
    }

    @Test("A winner that beats the best base run is stored with high confidence; attribution covers every candidate")
    func verdictOfCandidate() throws {
        let verdict = try Self.searched().verdict(selection: HarnessSelection(
            winnerId: "inv_0", finalists: ["inv_0"], judged: false, judgeFallback: false
        ))
        #expect(verdict.confidence == "high")
        #expect(verdict.delta == ["CLAUDE.md": "inv"])
        #expect(verdict.attributions.map(\.candidateId) == ["inv_0", "dom_0"])
        #expect(verdict.attributions.map(\.verdict) == [.helped, .hurt])
        #expect(verdict.lesson.hasPrefix("Winner: inv_0 (only finalist on measured stages).\nChanged files: CLAUDE.md."))
    }

    @Test("A base winner is stored with low confidence and an empty delta")
    func verdictOfBase() throws {
        let verdict = try Self.searched().verdict(selection: HarnessSelection(
            winnerId: "base_r0", finalists: ["base_r0", "inv_0"], judged: true, judgeFallback: true
        ))
        #expect(verdict.confidence == "low")
        #expect(verdict.delta.isEmpty)
        #expect(verdict.lesson.contains("(tie, judge reply unusable)") && verdict.lesson.contains("Changed files: none."))
    }

    @Test("Admission rejects an unparsable reply and a no-op edit, and names probes apart from base rollouts")
    func admission() throws {
        var search = try HarnessSearch(task: Self.task, base: Self.base, spec: .claude, k: 1, skipped: [], maxDeltaBytes: 64 * 1024)
        #expect(search.admit("inv_0", delta: nil) == nil)
        #expect(search.admit("dom_0", delta: ["CLAUDE.md": "Answer precisely."]) == nil)
        #expect(search.rejected == [
            HarnessRejection(candidateId: "inv_0", reasons: ["proposer reply is not a JSON object with string \"files\""]),
            HarnessRejection(candidateId: "dom_0", reasons: ["does not change the harness"])
        ])
        #expect(search.accepted.isEmpty)
        #expect(Self.candidate("base_r0", [:]).rolloutId == "base_r0")
        #expect(Self.candidate("inv_0", [:]).rolloutId == "probe_inv_0")
    }

    @Test("The judge's label picks the winner; an unusable reply falls back to a base finalist, else the first")
    func selection() throws {
        let search = try Self.searched()
        let finalists = search.runs.filter { $0.candidateId != "dom_0" }
        let candidates = [(label: "A", run: finalists[1]), (label: "B", run: finalists[0])]
        let judged = ContrastiveSelector.selection(finalists: finalists, candidates: candidates, reply: "BEST: A")
        #expect(judged == HarnessSelection(winnerId: "inv_0", finalists: ["base_r0", "inv_0"], judged: true, judgeFallback: false))
        let fallback = ContrastiveSelector.selection(finalists: finalists.reversed(), candidates: candidates, reply: "")
        #expect(fallback.winnerId == "base_r0" && fallback.judgeFallback)
        let noBase = ContrastiveSelector.selection(finalists: [finalists[1]], candidates: candidates, reply: "BEST: Z")
        #expect(noBase.winnerId == "inv_0" && noBase.judgeFallback)
    }

    @Test("A grader token in any harness file stops the search before any call, naming the first file")
    func leakInHarness() {
        let harness = HarnessTree(files: ["a.md": "fine", "c.md": "ground_truth_answer", "b.md": "see grader_verdict"])
        #expect(throws: HarnessBudgetError.leakInHarnessFile(path: "b.md", token: "grader_verdict")) {
            try HarnessLeakGuard.assertClean(harness: harness)
        }
        #expect(throws: Never.self) { try HarnessLeakGuard.assertClean(harness: Self.base) }
    }

    @Test("Library limits reject each invalid field with its own message, in a fixed order")
    func libraryLimits() throws {
        let limits = HarnessLibraryLimits()
        try limits.validate(topic: "t", cue: "c", domain: "general", content: "x", confidence: "low", delta: ["a": "b"])
        #expect(throws: HarnessLibraryError.invalidArgument("topic must be 1...200 characters")) {
            try limits.validate(topic: "", cue: "c", domain: "general", content: "", confidence: "certain", delta: nil)
        }
        #expect(throws: HarnessLibraryError.invalidArgument("confidence must be one of low, medium, high")) {
            try limits.validate(topic: "t", cue: "c", domain: "general", content: "x", confidence: "certain", delta: nil)
        }
        #expect(throws: HarnessLibraryError.invalidArgument("topic and cue reach the harness proposer and must not contain '\"score\":'")) {
            try limits.validate(topic: "t", cue: #"{"score": 1}"#, domain: "general", content: "x", confidence: "low", delta: nil)
        }
        #expect(throws: HarnessLibraryError.invalidArgument("harness delta must be at most 65536 bytes")) {
            try limits.validate(topic: "t", cue: "c", domain: "general", content: "x", confidence: "low", delta: ["a": String(repeating: "b", count: 70_000)])
        }
    }
}
