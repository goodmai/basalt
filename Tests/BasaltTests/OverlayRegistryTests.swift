import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference

@Suite("Research overlays: ids, a 30-minute TTL, undo and an append-only journal (#328)")
struct OverlayRegistryTests {
    static let start = Date(timeIntervalSince1970: 1_800_000_000)

    static func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("experiments-\(UUID().uuidString)")
    }

    @Test("Each overlay gets the next id and expires 30 minutes after it was set")
    func addsWithDefaultLifetime() {
        var registry = OverlayRegistry(journal: nil)
        let first = registry.add(.ablateHead(layer: 0, head: 1), description: "a", at: Self.start)
        let second = registry.add(nil, description: "b", at: Self.start)
        #expect(first.id == 1)
        #expect(second.id == 2)
        #expect(first.expiresAt == Self.start.addingTimeInterval(30 * 60))
        #expect(registry.overlays.map(\.info.description) == ["a", "b"])
    }

    @Test("An overlay lives until its expiry and is dropped at it")
    func expiresAtDeadline() {
        var registry = OverlayRegistry(journal: nil)
        let short = registry.add(nil, description: "short", at: Self.start, lifetime: 60)
        _ = registry.add(nil, description: "long", at: Self.start)
        #expect(registry.expire(at: Self.start.addingTimeInterval(59)).isEmpty)
        #expect(registry.expire(at: Self.start.addingTimeInterval(60)).map(\.info.id) == [short.id])
        #expect(registry.overlays.map(\.info.description) == ["long"])
    }

    @Test("Undo removes exactly the named overlay, and ids are not reused after a removal")
    func removesById() {
        var registry = OverlayRegistry(journal: nil)
        let info = registry.add(.ablateExpert(layer: 1, expert: 2), description: "x", at: Self.start)
        #expect(registry.remove(OverlayID(99)) == nil)
        #expect(registry.remove(info.id)?.info.id == info.id)
        #expect(registry.overlays.isEmpty)
        registry.removeAll()
        #expect(registry.add(nil, description: "y", at: Self.start).id == 2)
    }

    @Test("A journal line is one sorted JSON object with the event, id, description and ISO 8601 times")
    func journalLineFormat() throws {
        var registry = OverlayRegistry(journal: nil)
        let info = registry.add(nil, description: "delta on q_proj", at: Self.start)
        let line = try #require(OverlayRegistry.journalLine("add", info, at: Self.start))
        #expect(line.last == 0x0A)
        let entry = try JSONDecoder().decode(OverlayRegistry.JournalEntry.self, from: line)
        #expect(entry == OverlayRegistry.JournalEntry(
            event: "add", id: 1, description: "delta on q_proj",
            at: "2027-01-15T08:00:00Z", expiresAt: "2027-01-15T08:30:00Z"))
        #expect(String(decoding: line, as: UTF8.self).hasPrefix("{\"at\":"))
    }

    @Test("The journal appends to <fingerprint>.jsonl in a 0700 directory with a 0600 file")
    func journalAppendsPrivately() throws {
        let directory = Self.tempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var registry = OverlayRegistry(journal: directory)
        let info = registry.add(nil, description: "d", at: Self.start)
        try registry.record("add", info, at: Self.start, fingerprint: "fp")
        try registry.record("undo", info, at: Self.start, fingerprint: "fp")
        let file = directory.appendingPathComponent("fp.jsonl")
        let events = try String(contentsOf: file, encoding: .utf8).split(separator: "\n")
            .map { try JSONDecoder().decode(OverlayRegistry.JournalEntry.self, from: Data($0.utf8)).event }
        #expect(events == ["add", "undo"])
        let attributes = FileManager.default
        #expect((try attributes.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int) == 0o700)
        #expect((try attributes.attributesOfItem(atPath: file.path)[.posixPermissions] as? Int) == 0o600)
    }
}

extension MLXInferenceEngine {
    func journalExperiments(to directory: URL) {
        experiments.registry.journal = directory
    }
}

@Suite("Interventions and compare on tiny random models (#328)", .serialized)
struct InterventionCompareTests {
    enum Kind: String, CaseIterable, Sendable {
        case ablateHead, steer, ablateExpert

        var family: TinyModel.Family { self == .ablateExpert ? .qwen3Moe : .qwen3 }

        func intervention(hiddenSize: Int) -> Intervention {
            switch self {
            case .ablateHead: .ablateHead(layer: 0, head: 0)
            case .steer: .steer(layer: 1, vector: (0 ..< hiddenSize).map { Float($0 % 7) - 3 }, scale: 4)
            case .ablateExpert: .ablateExpert(layer: 0, expert: 0)
            }
        }
    }

    static let prompts = ["The capital of France is", "17 + 25 ="]

    static func engine(_ family: TinyModel.Family, journal: URL) async throws -> MLXInferenceEngine {
        let engine = try await TinyModel.engine(family)
        await engine.journalExperiments(to: journal)
        return engine
    }

    @Test("Without overlays the comparison is the baseline twice: KL 0, the same text and the same perplexity")
    func noOverlaysIsIdentity() async throws {
        let journal = OverlayRegistryTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: journal) }
        let engine = try await Self.engine(.qwen3, journal: journal)
        let report = try await engine.compare(CompareRequest(prompts: Self.prompts, overlays: [], maxTokens: 8))
        #expect(report.rows.map(\.prompt) == Self.prompts)
        for row in report.rows {
            #expect(row.klDivergence == 0)
            #expect(row.baselineText == row.overlayText)
            #expect(row.baselinePerplexity == row.overlayPerplexity)
            #expect(row.baselinePerplexity.isFinite)
        }
    }

    @Test("Each intervention moves the next-token distribution, is journaled, and undo removes it", arguments: Kind.allCases)
    func interventionMovesDistribution(_ kind: Kind) async throws {
        let journal = OverlayRegistryTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: journal) }
        let engine = try await Self.engine(kind.family, journal: journal)
        let hiddenSize = try await engine.anatomy().hiddenSize
        let overlay = try await engine.intervene(kind.intervention(hiddenSize: hiddenSize))
        #expect(await engine.overlays().map(\.id) == [overlay.id])

        let report = try await engine.compare(CompareRequest(prompts: Self.prompts, overlays: [overlay.id], maxTokens: 8))
        #expect(report.rows.allSatisfy { $0.klDivergence > 0 && $0.klDivergence.isFinite })

        try await engine.undo(overlay.id)
        #expect(await engine.overlays().isEmpty)
        await #expect(throws: BasaltError.self) { try await engine.undo(overlay.id) }
        let journaled = try FileManager.default.contentsOfDirectory(atPath: journal.path)
        #expect(journaled.count == 1)
    }

    @Test("Indices outside the model and a steering vector of the wrong size are refused")
    func refusesInvalidInterventions() async throws {
        let journal = OverlayRegistryTests.tempDirectory()
        defer { try? FileManager.default.removeItem(at: journal) }
        let engine = try await Self.engine(.qwen3, journal: journal)
        let anatomy = try await engine.anatomy()
        let invalid: [Intervention] = [
            .ablateHead(layer: anatomy.layers.count, head: 0),
            .ablateHead(layer: 0, head: anatomy.layers[0].heads ?? 0),
            .ablateExpert(layer: 0, expert: 0),
            .steer(layer: 0, vector: [1, 2, 3], scale: 1),
            .steer(layer: 0, vector: Array(repeating: .nan, count: anatomy.hiddenSize), scale: 1),
        ]
        for intervention in invalid {
            await #expect(throws: BasaltError.self) { try await engine.intervene(intervention) }
        }
        #expect(await engine.overlays().isEmpty)
        await #expect(throws: BasaltError.self) {
            try await engine.compare(CompareRequest(prompts: Self.prompts, overlays: [OverlayID(7)]))
        }
    }
}
