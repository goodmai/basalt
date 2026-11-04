import Testing
import BasaltCore
@testable import BasaltInference

@Suite("A request without a prompt cache gets its KV cache from one pure plan: head split, Quest, zones, or the stock cache")
struct GenerationLaunchPlanTests {
    static let mask = KVHeadMask(retrieval: [[0, 1]])

    @Test("Head split wins over Quest and zones, and both custom caches step aside when --kv-bits quantizes the cache")
    func priority() {
        #expect(FreshCache.plan(kvBitsSet: false, headMask: Self.mask, questTopK: 16, zones: .standard, precision: .k8v4) == .headSplit(Self.mask))
        #expect(FreshCache.plan(kvBitsSet: false, headMask: nil, questTopK: 16, zones: .standard, precision: .k8v4) == .quest(topK: 16))
        #expect(FreshCache.plan(kvBitsSet: true, headMask: Self.mask, questTopK: 16, zones: .standard, precision: .k8v4) == .zoned(.standard, .k8v4))
        #expect(FreshCache.plan(kvBitsSet: true, headMask: Self.mask, questTopK: 16, zones: nil, precision: nil) == .stock)
    }

    @Test("Zones need both a layout and a precision; otherwise the stock cache serves")
    func zonesNeedPrecision() {
        #expect(FreshCache.plan(kvBitsSet: false, headMask: nil, questTopK: nil, zones: .standard, precision: nil) == .stock)
        #expect(FreshCache.plan(kvBitsSet: false, headMask: nil, questTopK: nil, zones: nil, precision: .q8) == .stock)
        #expect(FreshCache.plan(kvBitsSet: false, headMask: nil, questTopK: nil, zones: .standard, precision: .q8) == .zoned(.standard, .q8))
    }
}
