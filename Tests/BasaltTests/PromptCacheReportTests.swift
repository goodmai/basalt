import Foundation
import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Prompt cache report: per-namespace shape, pins, snapshots and live launches (#328)", .serialized)
struct PromptCacheReportTests {
    let fixture = PromptCacheTreeTests()

    func bytes(_ tokens: Int) -> ByteCount {
        ByteCount(tokens * PromptCacheTreeTests.bytesPerToken)
    }

    @Test("Per namespace the report counts nodes, leaves, stored tokens, bytes, pinned nodes and the latest touch; a filter keeps one namespace and the totals")
    func countsEachNamespace() throws {
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        let a = Array(0..<10), b = Array(0..<5) + [100, 101, 102], c = fixture.branch(7, 20)
        tree.checkedStore("chat", tokens: a, cache: fixture.cache(a))
        tree.checkedStore("chat", tokens: b, cache: fixture.cache(b))
        tree.checkedStore("tools", tokens: c, cache: fixture.cache(c))
        let held = try tree.required("chat", for: b + [0])

        let report = tree.report(namespace: nil)
        #expect(report.namespaces.map(\.name) == ["chat", "tools"])
        let chat = try #require(report.namespaces.first), tools = try #require(report.namespaces.last)
        #expect([chat.nodes, chat.leaves, chat.tokens, chat.snapshots, chat.pinned] == [3, 2, 13, 0, 1])
        #expect([tools.nodes, tools.leaves, tools.tokens, tools.snapshots, tools.pinned] == [1, 1, 20, 0, 0])
        #expect(chat.bytes == bytes(13))
        #expect(tools.bytes == bytes(20))
        #expect(chat.lastUse > tools.lastUse)
        #expect(report.bytes == tree.byteCount)
        #expect(report.budget == ByteCount(1 << 30))
        #expect(report.evictions == 0)
        #expect(tree.report(namespace: "tools").namespaces.map(\.name) == ["tools"])
        #expect(tree.report(namespace: "missing").namespaces.isEmpty)
        #expect(tree.report(namespace: "missing").bytes == tree.byteCount)
        _ = consume held

        tree.checkedRebudget(bytes(13))
        let shrunk = tree.report(namespace: nil)
        #expect(shrunk.budget == bytes(13))
        #expect(shrunk.bytes == bytes(13))
        #expect(shrunk.evictions == tree.evictions)
        #expect(shrunk.evictions > 0)
        #expect(shrunk.namespaces.allSatisfy { $0.pinned == 0 })
        #expect(shrunk.namespaces.reduce(ByteCount.zero) { $0 + $1.bytes } == shrunk.bytes)
    }

    @Test("Recurrent snapshots are counted per namespace, the retained anchor and the leaf alike, with their bytes in the namespace total")
    func countsSnapshots() throws {
        let model = try TinyModel.make(.qwen35)
        let opening = TinyModel.tokens(Conversation.firstTurn)
        let turns = (0..<4).map { turn in opening + (0..<(turn * 20)).map { 100 + $0 % 50 } }
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.checkedStore("key:a", tokens: Array(opening.dropLast()), cache: try TinyHybrid.snapshotted(model, opening))
        for prompt in turns {
            tree.checkedStore("key:b", tokens: Array(prompt.dropLast()), cache: try TinyHybrid.snapshotted(model, prompt))
        }
        let lease = try tree.required("key:a", for: opening + [9, 9])

        let report = tree.report(namespace: nil)
        #expect(report.namespaces.map(\.name) == ["key:a", "key:b"])
        let a = try #require(report.namespaces.first), b = try #require(report.namespaces.last)
        #expect([a.nodes, a.leaves, a.tokens, a.snapshots, a.pinned] == [1, 1, opening.count - 1, 1, 1])
        #expect([b.nodes, b.leaves, b.tokens, b.snapshots, b.pinned] == [2, 1, turns[3].count - 1, 2, 0])
        #expect(a.bytes + b.bytes == tree.byteCount)
        #expect(a.bytes > ByteCount((opening.count - 1) * TinyHybrid.attentionBytesPerToken))
        _ = consume lease
        #expect(tree.checkInvariants(leases: 0) == [])
    }

    @Test("A live launch reports its reused tokens and its cache layers' class names; an uncached launch reports no layers")
    func launchReport() {
        let stream = AsyncStream<Generation> { $0.finish() }
        let cached = MLXInferenceEngine.GenerationLaunch(
            stream: stream, task: nil, cache: PromptCache([KVCacheSimple(), MambaCache()]), reusedTokens: 42)
        #expect(cached.report.reusedTokens == 42)
        #expect(cached.report.layers == ["KVCacheSimple", "MambaCache"])
        let uncached = MLXInferenceEngine.GenerationLaunch(stream: stream, task: nil, cache: nil, reusedTokens: 0)
        #expect(uncached.report.layers.isEmpty)
    }

    @Test("On the engine after two cached turns the report shows the split conversation, no pins and no live leases or launches")
    func engineReportAfterTwoTurns() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let first = TinyModel.tokens(Conversation.firstTurn), next = TinyModel.tokens(Conversation.afterToolCall)
        let shared = commonPrefix(first, next)
        _ = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let second = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(second.cachedPromptTokens == shared)

        let report = await engine.promptCacheReport(namespace: nil)
        let (budget, stored) = await engine.readPromptCaches { ($0.byteBudget, $0.byteCount) }
        #expect(report.namespaces.map(\.name) == ["ns"])
        let ns = try #require(report.namespaces.first)
        #expect([ns.nodes, ns.leaves, ns.tokens, ns.snapshots, ns.pinned] == [3, 2, first.count + next.count - shared, 0, 0])
        #expect(ns.bytes == stored)
        #expect(ns.bytes > 0)
        #expect(report.bytes == stored)
        #expect(report.budget == budget)
        #expect(report.evictions == 0)
        #expect(report.leases == 0)
        #expect(report.launches.isEmpty)
        #expect(await engine.promptCacheReport(namespace: "other").namespaces.isEmpty)
        #expect(await engine.checkInvariants(idle: true) == [])
    }
}
