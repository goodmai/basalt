import Foundation
import MLXLMCommon
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Before a long prefill the engine drops the unpinned prompt cache, keeps the leased prefix and restores the budget (#741 P1)")
struct LongPrefillCacheTests {
    let fixture = PromptCacheTreeTests()

    @Test("makeRoom evicts every unpinned entry, keeps the leased prefix, reports the bytes it dropped and leaves the budget as it was")
    func makeRoomKeepsTheLease() throws {
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        let chat = Array(0 ..< 10), tools = fixture.branch(7, 20)
        tree.checkedStore("chat", tokens: chat, cache: fixture.cache(chat))
        tree.checkedStore("tools", tokens: tools, cache: fixture.cache(tools))
        let held = try tree.required("chat", for: chat + [0])
        let dropped = tree.makeRoom(tree.byteBudget)
        #expect(dropped == ByteCount(20 * PromptCacheTreeTests.bytesPerToken))
        #expect(tree.storedPrefix(of: tools, in: "tools") == 0)
        #expect(tree.storedPrefix(of: chat, in: "chat") == 10)
        #expect(tree.byteBudget == ByteCount(1 << 30))
        #expect(tree.checkInvariants(leases: 1).isEmpty)
        _ = consume held
        tree.checkedStore("tools", tokens: tools, cache: fixture.cache(tools))
        #expect(tree.storedPrefix(of: tools, in: "tools") == 20)
    }

    @Test("Only a prefill of 8,192 new tokens or more clears the cache first")
    func threshold() {
        #expect(!MLXInferenceEngine.clearsBeforePrefill(newTokens: MemoryDirective.longContextTokens - 1))
        #expect(MLXInferenceEngine.clearsBeforePrefill(newTokens: MemoryDirective.longContextTokens))
    }
}
