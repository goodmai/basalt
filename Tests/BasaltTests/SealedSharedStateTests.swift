import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("State shared across requests is a sealed value or sits behind a Mutex, not behind @unchecked Sendable (#300)")
struct SealedSharedStateTests {
    @Test("Concurrent requests on one CacheBlendStore all see the chunks of a single caller")
    func cacheBlendStoreServesConcurrentTasks() async {
        let store = CacheBlendStore(ratio: 0, byteBudget: 0)
        let prompt = Array(Files.reordered.dropLast())
        let tokenizer = ParagraphTokenizer()
        let expected = CacheBlendStore.chunks(prompt.map { tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false) })
        let seen = await withTaskGroup(of: [Range<Int>].self) { group in
            for _ in 0 ..< 8 { group.addTask { store.chunks(prompt, tokenizer: tokenizer) } }
            return await group.reduce(into: []) { $0.append($1) }
        }
        #expect(seen.count == 8)
        #expect(seen.allSatisfy { $0 == expected })
    }

    @Test("An MTP prefix hands out drafter caches equal to the stored one, and a write to one reaches neither the prefix nor the next")
    func mtpPrefixHandsOutIndependentCaches() throws {
        let source = KVCacheSimple()
        let keys = MLXArray((0 ..< 24).map(Float.init), [1, 2, 3, 4])
        _ = source.update(keys: keys, values: -keys)
        let prefix = try #require(MTPPrefix(MTPDrafterState(cache: [source], nextPosition: 3, carryHidden: MLXArray.ones([1, 1, 4]))))
        _ = source.update(keys: keys, values: keys)

        let first = prefix.state
        _ = first.cache[0].update(keys: keys, values: keys)
        let second = prefix.state

        #expect(prefix.nextPosition == 3)
        #expect(first.cache[0].offset == 6)
        #expect(second.cache[0].offset == 3)
        #expect(second.nextPosition == 3)
        #expect(arrayEqual(second.cache[0].state[0], keys).item(Bool.self))
        #expect(arrayEqual(second.cache[0].state[1], -keys).item(Bool.self))
        #expect(arrayEqual(try #require(second.carryHidden), MLXArray.ones([1, 1, 4])).item(Bool.self))
    }

    @Test("Only a drafter state with a carried hidden row and plain KV caches becomes a prefix")
    func mtpPrefixNeedsCarryAndSimpleCaches() {
        #expect(MTPPrefix(MTPDrafterState(cache: [KVCacheSimple()])) == nil)
        #expect(MTPPrefix(MTPDrafterState(cache: [RotatingKVCache(maxSize: 8)], carryHidden: MLXArray.ones([1, 1, 4]))) == nil)
        #expect(MTPPrefix(MTPDrafterState(cache: [KVCacheSimple()], carryHidden: MLXArray.ones([1, 1, 4])))?.layers.map(\.count) == [0])
    }
}
