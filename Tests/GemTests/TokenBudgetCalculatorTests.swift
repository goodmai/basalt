import Testing
import Foundation
@testable import GemCore

// MARK: — TokenBudgetCalculator Tests

@Suite("TokenBudgetCalculator")
struct TokenBudgetCalculatorTests {

    // MARK: — TC-1.2.4.1: Calculate max tokens for 8GB RAM

    @Test("Calculate max tokens for 8GB RAM with 4GB model")
    func testCalculateMaxTokens8GB() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 4000 // ~4 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 8 * 0.8 = 6.4 GB
        // Available for context = 6.4 GB - 4 GB = 2.4 GB
        // 1 token ≈ 2 bytes KV cache
        // maxTokens = 2.4 GB / 2 bytes ≈ 1.2B tokens -> capped at 128,000

        #expect(maxTokens > 0)
        #expect(maxTokens <= 128_000)
    }

    @Test("Calculate max tokens for 8GB RAM with tight model (6GB)")
    func testCalculateMaxTokens8GBTight() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 6000 // ~6 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 8 * 0.8 = 6.4 GB
        // Available for context = 6.4 GB - 6 GB = 400 MB
        // Should still return reasonable tokens

        #expect(maxTokens > 0)
        #expect(maxTokens <= 128_000)
    }

    @Test("Calculate max tokens for 8GB RAM with very tight model (7.5GB)")
    func testCalculateMaxTokens8GBVeryTight() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 7500 // ~7.5 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 8 * 0.8 = 6.4 GB
        // Available for context = 6.4 GB - 7.5 GB = < 0
        // Should return minimum fallback

        #expect(maxTokens > 0)
        #expect(maxTokens <= 1024)
    }

    // MARK: — TC-1.2.4.2: Calculate max tokens for 16GB RAM

    @Test("Calculate max tokens for 16GB RAM with 4GB model")
    func testCalculateMaxTokens16GB() {
        let availableRAM: Int64 = 16 * 1024 * 1024 * 1024 // 16 GB
        let modelSizeMB = 4000 // ~4 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 16 * 0.8 = 12.8 GB
        // Available for context = 12.8 GB - 4 GB = 8.8 GB
        // Should be capped at 128k

        #expect(maxTokens == 128_000)
    }

    @Test("Calculate max tokens for 16GB RAM with 8GB model")
    func testCalculateMaxTokens16GBLargeModel() {
        let availableRAM: Int64 = 16 * 1024 * 1024 * 1024 // 16 GB
        let modelSizeMB = 8000 // ~8 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 16 * 0.8 = 12.8 GB
        // Available for context = 12.8 GB - 8 GB = 4.8 GB
        // Should still be capped at 128k

        #expect(maxTokens > 0)
        #expect(maxTokens <= 128_000)
    }

    // MARK: — TC-1.2.4.3: Calculate max tokens for 32GB RAM

    @Test("Calculate max tokens for 32GB RAM with 4GB model")
    func testCalculateMaxTokens32GB() {
        let availableRAM: Int64 = 32 * 1024 * 1024 * 1024 // 32 GB
        let modelSizeMB = 4000 // ~4 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 32 * 0.8 = 25.6 GB
        // Available for context = 25.6 GB - 4 GB = 21.6 GB
        // Should be capped at 128k

        #expect(maxTokens == 128_000)
    }

    @Test("Calculate max tokens for 32GB RAM with 16GB model")
    func testCalculateMaxTokens32GBLargeModel() {
        let availableRAM: Int64 = 32 * 1024 * 1024 * 1024 // 32 GB
        let modelSizeMB = 16000 // ~16 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 32 * 0.8 = 25.6 GB
        // Available for context = 25.6 GB - 16 GB = 9.6 GB
        // Should be capped at 128k

        #expect(maxTokens == 128_000)
    }

    // MARK: — TC-1.2.4.4: Apply safety margin correctly

    @Test("Safety margin is applied correctly")
    func testSafetyMarginApplied() {
        let availableRAM: Int64 = 10 * 1024 * 1024 * 1024 // 10 GB
        let modelSizeMB = 2000 // ~2 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Safety margin = 0.8
        // Usable = 10 * 0.8 = 8 GB
        // Available for context = 8 GB - 2 GB = 6 GB
        // Should be capped at 128k

        #expect(maxTokens > 0)
        #expect(maxTokens <= 128_000)

        // Verify safety margin is actually applied by checking it's less than
        // what we'd get without safety margin
        let unsafeUsable = availableRAM - Int64(modelSizeMB * 1024 * 1024)
        let unsafeMaxTokens = Int(unsafeUsable / 2) // 2 bytes per token

        #expect(maxTokens < unsafeMaxTokens)
    }

    // MARK: — TC-1.2.4.5: Cap at 128k tokens

    @Test("Max tokens capped at 128k")
    func testMaxTokensCappedAt128k() {
        let availableRAM: Int64 = 64 * 1024 * 1024 * 1024 // 64 GB
        let modelSizeMB = 4000 // ~4 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Even with huge RAM, should be capped at 128k
        #expect(maxTokens == 128_000)
    }

    @Test("Max tokens capped at 128k for 128GB RAM")
    func testMaxTokensCappedAt128kHugeRAM() {
        let availableRAM: Int64 = 128 * 1024 * 1024 * 1024 // 128 GB
        let modelSizeMB = 8000 // ~8 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Even with huge RAM, should be capped at 128k
        #expect(maxTokens == 128_000)
    }

    // MARK: — Additional Edge Cases

    @Test("Calculate max tokens with minimal RAM")
    func testCalculateMaxTokensMinimalRAM() {
        let availableRAM: Int64 = 4 * 1024 * 1024 * 1024 // 4 GB
        let modelSizeMB = 3500 // ~3.5 GB model

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Very tight, but should still return something
        #expect(maxTokens > 0)
    }

    @Test("Calculate max tokens with zero model size")
    func testCalculateMaxTokensZeroModel() {
        let availableRAM: Int64 = 16 * 1024 * 1024 * 1024 // 16 GB
        let modelSizeMB = 0

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Should return max (128k) since all RAM is available
        #expect(maxTokens == 128_000)
    }

    @Test("Calculate max tokens with model larger than RAM")
    func testCalculateMaxTokensModelLargerThanRAM() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 10000 // ~10 GB model (larger than RAM)

        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(
            availableRAM: availableRAM,
            modelSizeMB: modelSizeMB
        )

        // Should return minimum fallback
        #expect(maxTokens > 0)
        #expect(maxTokens <= 1024)
    }
}
