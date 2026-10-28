import Testing
import Foundation
@testable import BasaltCore

@Suite("TokenBudgetCalculator")
struct TokenBudgetCalculatorTests {

    static let gib: Int64 = 1 << 30
    static let qwen32B = KVCacheGeometryTests.dense(layers: 64, kvHeads: 8)
    static var qwen3_4B: KVCacheGeometry { KVCacheGeometry(config: KVCacheGeometryTests.qwen3_4B)! }

    static func availableRAM(usable bytes: Int64, weightsMB: Int = 0) -> Int64 {
        Int64((Double(bytes + Int64(weightsMB) << 20) / TokenBudgetCalculator.safetyMargin).rounded(.up))
    }

    @Test("4 GiB left after the weights with a 32B fp16 geometry holds 16,384 tokens, not 128k")
    func denseBudget() {
        let tokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: Self.availableRAM(usable: 4 * Self.gib, weightsMB: 18_000),
            modelSizeMB: 18_000, kv: Self.qwen32B
        )
        #expect(tokens == 16_384)
    }

    @Test("The prompt is subtracted from the budget")
    func promptSubtracted() {
        let ram = Self.availableRAM(usable: 4 * Self.gib)
        let tokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: ram, modelSizeMB: 0, kv: Self.qwen32B, promptTokens: 10_000
        )
        #expect(tokens == 6_384)
    }

    @Test("--kv-bits 4 stretches the same memory by 16/4.5")
    func kvBits() {
        let ram = Self.availableRAM(usable: 4 * Self.gib)
        let tokens = TokenBudgetCalculator.calculateMaxTokens(availableRAM: ram, modelSizeMB: 0, kv: Self.qwen32B, kvPrecision: .q4)
        #expect(tokens == 16_384 * 16 * 2 / 9)
    }

    @Test("Qwen3-4B on 16 GB free: the cap falls from the 2-bytes-per-token 128k to what 144 KiB per token fits")
    func qwen3_4BCap() {
        let tokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: 16 * Self.gib, modelSizeMB: 2_200, kv: Self.qwen3_4B
        )
        let expected = (Int64(Double(16 * Self.gib) * 0.8) - 2_200 << 20) / 147_456
        #expect(tokens == Int(expected))
        #expect(tokens < 80_000)
    }

    @Test("The hybrid Ornith geometry keeps the 128k cap where a dense one of its size would not")
    func hybridCap() {
        let ornith = KVCacheGeometry(config: KVCacheGeometryTests.ornith9B)!
        let ram = 12 * Self.gib
        #expect(TokenBudgetCalculator.calculateMaxTokens(availableRAM: ram, modelSizeMB: 5_000, kv: ornith) == 128_000)
        #expect(TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: ram, modelSizeMB: 5_000, kv: KVCacheGeometry(parameterProxy: 9)) < 128_000)
    }

    @Test("Huge RAM is capped at 128k", arguments: [64, 128] as [Int64])
    func cap(gb: Int64) {
        #expect(TokenBudgetCalculator.calculateMaxTokens(availableRAM: gb * Self.gib, modelSizeMB: 4000, kv: Self.qwen3_4B) == 128_000)
    }

    @Test("Weights at or over the usable RAM, or no RAM at all, leave the 512-token floor",
          arguments: [(8, 7500), (8, 10_000), (4, 3500), (0, 0)] as [(Int64, Int)])
    func floor(gb: Int64, modelMB: Int) {
        #expect(TokenBudgetCalculator.calculateMaxTokens(availableRAM: gb * Self.gib, modelSizeMB: modelMB, kv: Self.qwen32B) == 512)
    }

    @Test("A prompt larger than the budget leaves the 512-token floor")
    func promptOverBudget() {
        let ram = Self.availableRAM(usable: 4 * Self.gib)
        #expect(TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: ram, modelSizeMB: 0, kv: Self.qwen32B, promptTokens: 100_000) == 512)
    }
}
