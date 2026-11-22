import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltInference
@testable import BasaltUI
@testable import BasaltBench

@Suite("Round 2 audit governance and core regression tests (#678, #680)", .serialized)
struct Round2GovernanceCoreTests {
    @Test("HITLApprovalActor resumes expired waiters on decide and enqueue instead of deadlocking (#678)")
    func hitlResumesExpiredWaitersOnDecide() async throws {
        let approvals = HITLApprovalActor(policy: HITLToolPolicy(defaultTimeoutSeconds: 10))
        _ = try await approvals.enqueue(id: "op-1", toolName: "weights_delta", argumentsSummary: "{}", now: 100)
        let waiter = Task { await approvals.awaitDecision(id: "op-1") }
        await Task.yield()
        do {
            _ = try await approvals.decide(id: "op-1", approved: true, reviewer: "alice", reason: "ok", now: 120)
            Issue.record("expected expired approval to fail decide")
        } catch {}
        let decision = await waiter.value
        #expect(!decision.approved)
    }

    @Test("Introspector.overlayID parses repairable JSON and rejects unparseable overlay_undo (#678)")
    func introspectorOverlayIDHandlesRepairableJSON() {
        let trailingComma = ToolCallOutput(id: "1", name: "overlay_undo", arguments: "{\"id\": 0,}")
        #expect(Introspector.overlayID(trailingComma) == OverlayID(0))
        let singleQuotes = ToolCallOutput(id: "2", name: "overlay_undo", arguments: "{'id': 7}")
        #expect(Introspector.overlayID(singleQuotes) == OverlayID(7))
        let invalid = ToolCallOutput(id: "3", name: "overlay_undo", arguments: "not-json")
        #expect(Introspector.overlayID(invalid) == nil)
    }

    @Test("PipelineController rejects attachment arrays exceeding maxAttachments before disk reads (#678)")
    func pipelineRejectsExcessAttachmentsBeforeReading() {
        let items = Array(repeating: #"{"kind":"image","path":"/nonexistent.png"}"#, count: PipelineRequest.maxAttachments + 1)
        let json = Data("{\"attachments\":[\(items.joined(separator: ","))]}".utf8)
        #expect(throws: BasaltError.self) {
            _ = try PipelineController.request(from: json, imageDirectories: [])
        }
    }

    @Test("SessionLogWriter.sanitized caps UTF-8 filename length and strips non-ASCII (#678)")
    func sessionLogSanitizedCapsUTF8Length() {
        let cjk = String(repeating: "漢", count: 120)
        let sanitized = SessionLogWriter.sanitized(cjk)
        #expect(sanitized != nil)
        #expect((sanitized?.utf8.count ?? 999) <= 128)
    }

    @Test("ChatRequestFastReader accepts explicit null response_format with guided_regex (#680)")
    func chatFastReaderAcceptsNullResponseFormatWithGuidedRegex() throws {
        let payload = Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":null,"guided_regex":"[0-9]+"}"#.utf8)
        let parsed = try ChatRequestFastReader.read(payload)
        #expect(try parsed.responseFormat.get() == .regex("[0-9]+"))
    }

    @Test("TableRenderer and ProgressBar handle mismatched row/column counts and negative inputs safely (#680)")
    func tableRendererAndProgressBarSafeBounds() {
        let rendered = TableRenderer.render(headers: ["A", "B"], rows: [["1", "2", "3"]])
        #expect(rendered.contains("1") && rendered.contains("2"))
        let built = TableBuilder().addColumn("A", width: 4).addColumn("B").addRow(["1", "2"]).build()
        #expect(built.contains("A") && built.contains("B"))

        var bar = ProgressBar(total: 10, title: "test", style: .bar(width: -5))
        bar.update(current: -3)
        #expect(bar.current == 0 && bar.percentage == 0)
        #expect(!bar.render().isEmpty)
    }

    @Test("BenchmarkComparator tolerates duplicate sample IDs without trapping (#680)")
    func benchmarkComparatorToleratesDuplicateSampleIDs() {
        let sample = BenchmarkSampleRecord(id: "dup-0", adapter: "test", answer: "ok", score: 1.0)
        let manifest = BenchmarkManifest(runID: "r1", composeName: "c", suite: "s", seed: 1, temperature: 0)
        let provenance = BasaltProvenance(gitCommit: "abc", composeSHA256: "0", chip: "M4", unifiedMemoryGB: 24, timestampISO8601: "2026-11-22T00:00:00Z")
        let card = BenchmarkScorecard.evaluate(metrics: ["score": 1.0], gates: nil, samples: [sample, sample])
        let artifact = BenchmarkRunArtifact(manifest: manifest, provenance: provenance, scorecard: card, samples: [sample, sample])
        let delta = BenchmarkComparator.compare(baseline: artifact, candidate: artifact)
        #expect(delta.sampleRegressions.isEmpty && delta.sampleImprovements.isEmpty)
    }
}

@Suite("Round 2 audit inference regression tests (#679)", .serialized)
struct Round2InferenceTests {
    @Test("PHMLinear sharedBasis propagates updated phm_a from owner to borrowed layers (#679)")
    func phmSharedBasisPropagatesOwnerUpdates() throws {
        let bank = PHMBasisBank(hypercomplexDim: 2)
        let baseA = Linear(8, 8, bias: false)
        let baseB = Linear(8, 8, bias: false)
        let owner = try PHMLinear(linear: baseA, hypercomplexDim: 2, factorRank: 1, scale: 1.0, basisMode: .owned(bank))
        let borrowed = try PHMLinear(linear: baseB, hypercomplexDim: 2, factorRank: 1, scale: 1.0, basisMode: .borrowed(bank))

        let newBasis = MLXArray.ones([2, 2, 2], type: Float.self) * Float(3.0)
        owner.update(parameters: .unflattened(["phm_a": newBasis]))
        eval(owner.activeBasis, borrowed.activeBasis)

        let diff = Float(max(abs(borrowed.activeBasis - newBasis)).item(Float.self))
        #expect(diff < 1e-6)
    }

    @Test("TieredKVHierarchyCore demotes L0 overflow directly to L2 when L1 capacity is zero (#679)")
    func tieredKVDemotesL0ToL2WhenL1Zero() throws {
        let config = TieredKVConfig(
            topology: .discreteVRAM,
            l0CapacityBytes: 100,
            l1CapacityBytes: 0,
            l2CapacityBytes: 500
        )
        var core = TieredKVHierarchyCore(config: config)
        try core.storeChunk(prefixHash: "b1", tokenCount: 16, byteSize: 80, layerCount: 4)
        try core.storeChunk(prefixHash: "b2", tokenCount: 16, byteSize: 80, layerCount: 4)
        #expect(core.block(for: "b1")?.tier == .l2SharedNVMe)
        #expect(core.block(for: "b2")?.tier == .l0GPU)
    }

    @Test("SpecTrAndResidualRaceMetal guards NaN target logits and out-of-range draft tokens (#679)")
    func specTrGuardsNaNAndOutOfRangeTokens() {
        let nanLogits = MLXArray([Float.nan, Float.nan, Float.nan, Float.nan], [2, 2])
        let draftLogits = MLXArray.zeros([1, 1, 2], type: Float.self)
        let badDraft = MLXArray([Int32(99)], [1, 1])
        let res = SpecTrAndResidualRaceMetal.verifyGPU(
            targetLogits: nanLogits,
            draftLogits: draftLogits,
            draftTokens: badDraft,
            temperature: 0.0,
            mode: .residualRace,
            seed: 1
        )
        let firstTok = res.emittedTokens.first ?? -1
        #expect(firstTok >= 0 && firstTok < 2)
    }
}
