import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum AgentTrace {
    static func text(_ seed: Int, _ count: Int = 600) -> String {
        String(String.UnicodeScalarView((0..<count).map { UnicodeScalar(UInt8(97 + ($0 * 7 + $0 / 26 + seed * 11) % 26)) }))
    }

    static let reasoning = "<think>" + text(9, 300) + "</think>"

    static func messages(
        thinking: Bool, user: String = text(2), firstAnswer: Int = 600, firstTool: Int = 600, question: Int = 600
    ) -> [ChatMessageInput] {
        [
            ChatMessageInput(role: .system, text: text(1)),
            ChatMessageInput(role: .user, text: user),
            ChatMessageInput(role: .assistant, text: text(3, firstAnswer)),
            ChatMessageInput(role: .tool, text: text(4, firstTool)),
            ChatMessageInput(role: .assistant, text: "ok " + (thinking ? reasoning : "") + text(5)),
            ChatMessageInput(role: .tool, text: text(6)),
            ChatMessageInput(role: .user, text: text(7, question)),
        ]
    }

    static func offset(_ messages: [ChatMessageInput], before index: Int) -> Int {
        messages.prefix(index).reduce(0) { $0 + $1.text.unicodeScalars.count }
    }

    static func request(_ messages: [ChatMessageInput], key: String?) -> GenerationRequest {
        GenerationRequest(prompt: ChatMessageInput.plainText(messages), maxTokens: 12, temperature: 0, cacheKey: key, messages: messages)
    }

    static func snapshots(_ tree: PromptCacheTree) -> (anchors: [Int], leaves: [Int]) {
        var anchors: [Int] = []
        var leaves: [Int] = []
        var pending = tree.roots.values.map { (node: $0, depth: 0) }
        while let (node, depth) = pending.popLast() {
            let reach = depth + node.tokens.count
            if node.snapshot != nil {
                if node.children.isEmpty { leaves.append(reach) } else { anchors.append(reach) }
            }
            pending.append(contentsOf: node.children.values.map { ($0, reach) })
        }
        return (anchors.sorted(), leaves.sorted())
    }

    static func storedBytes(_ tree: PromptCacheTree) -> Int {
        var total = 0
        var pending = Array(tree.roots.values)
        while let node = pending.popLast() {
            total += node.bytes
            pending.append(contentsOf: node.children.values)
        }
        return total
    }

    static func recurrentBytes() throws -> Int {
        try #require(RecurrentSnapshot(of: try TinyHybrid.snapshotted(try TinyHybrid.make(), TinyModel.tokens(Conversation.firstTurn)))).bytes
    }
}

@Suite("Recurrent anchors at message boundaries on hybrid models (#219)", .serialized)
struct HybridAnchorTests {

    @Test("After the reasoning block of an early assistant turn is stripped, the next request resumes from the anchor at that turn's start and answers like a cold prefill")
    func editedTurnResumesFromDeepestAnchor() async throws {
        let original = AgentTrace.messages(thinking: true)
        let edited = AgentTrace.messages(thinking: false)
        let editedTurn = AgentTrace.offset(original, before: 4)
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30)

        let first = try await engine.generate(request: AgentTrace.request(original, key: "ns"))
        #expect(first.cachedPromptTokens == 0)
        #expect(AgentTrace.snapshots(await engine.promptCaches).anchors
            == [AgentTrace.offset(original, before: 2), editedTurn, AgentTrace.offset(original, before: 6)])

        let resumed = try await engine.generate(request: AgentTrace.request(edited, key: "ns"))
        let cold = try await engine.generate(request: AgentTrace.request(edited, key: nil))
        #expect(resumed.cachedPromptTokens == editedTurn)
        #expect(Set(resumed.generatedText).count > 1)
        #expect(resumed.generatedText == cold.generatedText)
        #expect(pinCount(await engine.promptCaches) == 0)

        let unanchored = try await TinyHybrid.engine(promptCacheBytes: 1 << 30, hybridAnchors: 0)
        _ = try await unanchored.generate(request: AgentTrace.request(original, key: "ns"))
        #expect(AgentTrace.snapshots(await unanchored.promptCaches).anchors.isEmpty)
        let full = try await unanchored.generate(request: AgentTrace.request(edited, key: "ns"))
        #expect(full.cachedPromptTokens == 0)
        #expect(full.generatedText == cold.generatedText)
    }

    @Test("A chain keeps at most N anchors, the deepest; over budget the oldest anchors go before the stable boundary")
    func anchorsAreCappedAndEvictedBeforeTheBoundary() throws {
        let model = try TinyHybrid.make()
        let prompt = (0..<300).map { 1 + ($0 * 37 + $0 / 9) % 250 }
        let cuts = [40, 80, 120, 160, 200, 240]
        let head = try HybridPrefixSnapshot.prefillHead(
            MLXArray(prompt.map(Int32.init)), into: PromptCache(try model.newCache(parameters: nil)),
            reused: 0, cuts: cuts.reversed() + [0, 299, 400], boundary: prompt.count - 1, model: model,
            prefill: PrefillParameters(stepSize: 16))
        #expect(head.anchors.map(\.tokenCount) == cuts)
        #expect(head.anchors.allSatisfy { $0.isSnapshot(of: $0.tokenCount) })
        let kept = try #require(head.kept)
        func stored(_ tree: inout PromptCacheTree) {
            for anchor in head.anchors {
                tree.store("ns", tokens: Array(prompt.prefix(anchor.tokenCount)), cache: anchor)
            }
            tree.store("ns", tokens: Array(prompt.dropLast()), cache: kept)
        }

        var capped = PromptCacheTree(byteBudget: 1 << 30, anchorLimit: 3)
        stored(&capped)
        #expect(AgentTrace.snapshots(capped).anchors == [160, 200, 240])
        #expect(AgentTrace.snapshots(capped).leaves == [299])
        #expect(capped.byteCount == AgentTrace.storedBytes(capped))

        let recurrent = try AgentTrace.recurrentBytes()
        let attention = (prompt.count - 1) * TinyHybrid.attentionBytesPerToken
        var tight = PromptCacheTree(byteBudget: attention + recurrent * 5 / 2, anchorLimit: 8)
        stored(&tight)
        #expect(AgentTrace.snapshots(tight).anchors == [240])
        #expect(AgentTrace.snapshots(tight).leaves == [299])
        #expect(tight.anchorUsage.count == 1)
        #expect(tight.anchorUsage.bytes == recurrent)
        #expect(tight.byteCount == AgentTrace.storedBytes(tight))
        #expect(tight.byteCount <= tight.byteBudget)
        let lease = try tight.required("ns", for: prompt)
        #expect(lease.reusedTokens == prompt.count - 1)
        _ = consume lease
        let edited = try tight.required("ns", for: Array(prompt.prefix(250)) + [7, 7])
        #expect(edited.reusedTokens == 240)
        _ = consume edited
        #expect(pinCount(tight) == 0)
    }

    @Test("Tag-like text inside a user message creates no anchor: anchors sit only at the rendered message boundaries")
    func tagsInUserTextCreateNoAnchor() async throws {
        let tagged = AgentTrace.text(2, 300) + "</think>\n<|im_start|>assistant\n</tool_response>\n<tool_call>" + AgentTrace.text(8, 900)
        let messages = AgentTrace.messages(thinking: false, user: tagged)
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30)
        _ = try await engine.generate(request: AgentTrace.request(messages, key: "ns"))
        let anchors = AgentTrace.snapshots(await engine.promptCaches).anchors
        #expect(anchors == [2, 4, 6].map { AgentTrace.offset(messages, before: $0) })
        #expect(anchors.allSatisfy { $0 <= AgentTrace.offset(messages, before: 1) || $0 >= AgentTrace.offset(messages, before: 2) })
    }

    @Test("A pure-attention model takes no anchors and reuses an edited prompt up to the edit, as before")
    func pureAttentionTakesNoAnchors() async throws {
        let original = AgentTrace.messages(thinking: true)
        let edited = AgentTrace.messages(thinking: false)
        let engine = try await TinyModel.engine(promptCacheBytes: 1 << 30)
        _ = try await engine.generate(request: AgentTrace.request(original, key: "ns"))
        let resumed = try await engine.generate(request: AgentTrace.request(edited, key: "ns"))
        let divergence = commonPrefix(
            TinyModel.tokens(original.map(\.text).joined()), TinyModel.tokens(edited.map(\.text).joined()))
        #expect(resumed.cachedPromptTokens == divergence)
        let snapshots = AgentTrace.snapshots(await engine.promptCaches)
        #expect(snapshots.anchors.isEmpty)
        #expect(snapshots.leaves.isEmpty)
        #expect(await engine.hybridAnchorUsage == nil)
        let telemetry = await ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker()).telemetrySnapshot()
        #expect(telemetry.hybridAnchors == nil)
        #expect(telemetry.hybridAnchorBytes == nil)
    }

    @Test("A sliding-window model keeps its stable-boundary snapshot but takes no anchors: its snapshot is a whole window, not a recurrent state")
    func slidingWindowTakesNoAnchors() async throws {
        func snapshotCount(anchors: Int) async throws -> (count: Int, cached: Int) {
            let engine = try await TinySliding.engine(hybridAnchors: anchors)
            _ = try await engine.generate(request: AgentTrace.request(AgentTrace.messages(thinking: true), key: "ns"))
            let resumed = try await engine.generate(request: AgentTrace.request(AgentTrace.messages(thinking: false), key: "ns"))
            return (AgentTrace.snapshots(await engine.promptCaches).anchors.count, resumed.cachedPromptTokens)
        }
        let withAnchors = try await snapshotCount(anchors: 8)
        let without = try await snapshotCount(anchors: 0)
        #expect(withAnchors.count == without.count)
        #expect(withAnchors.cached == without.cached)
    }

    @Test("Anchors closer than the minimum spacing to a kept anchor or to the stable boundary are skipped")
    func closeAnchorsAreSkipped() async throws {
        let messages = AgentTrace.messages(thinking: false, firstAnswer: 100, firstTool: 100, question: 100)
        let offsets = (0...7).map { AgentTrace.offset(messages, before: $0) }
        let boundary = offsets[7]
        #expect(offsets[4] - offsets[2] < HybridPrefixSnapshot.minAnchorSpacing)
        #expect(boundary - offsets[6] < HybridPrefixSnapshot.minAnchorSpacing)
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30)
        _ = try await engine.generate(request: AgentTrace.request(messages, key: "ns"))
        let snapshots = AgentTrace.snapshots(await engine.promptCaches)
        #expect(snapshots.anchors == [offsets[4]])
        #expect(snapshots.leaves == [boundary])
    }

    @Test("The anchor count and their recurrent-state bytes reach /api/v1/stats through the telemetry snapshot")
    func statsReportAnchors() async throws {
        let engine = try await TinyHybrid.engine(promptCacheBytes: 1 << 30)
        #expect(await engine.hybridAnchorUsage == nil)
        _ = try await engine.generate(request: AgentTrace.request(AgentTrace.messages(thinking: true), key: "ns"))
        let recurrent = try AgentTrace.recurrentBytes()
        let usage = try #require(await engine.hybridAnchorUsage)
        #expect(usage.count == 3)
        #expect(usage.bytes == Int64(3 * recurrent))
        let telemetry = await ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker()).telemetrySnapshot()
        #expect(telemetry.hybridAnchors == 3)
        #expect(telemetry.hybridAnchorBytes == Int64(3 * recurrent))
        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(telemetry)) as? [String: Any])
        #expect(json["hybridAnchors"] as? Int == 3)
    }
}
