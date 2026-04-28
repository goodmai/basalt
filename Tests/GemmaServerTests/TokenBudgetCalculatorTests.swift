import XCTest
@testable import GemmaServerCore

final class TokenBudgetCalculatorTests: XCTestCase {
    
    func testCalculateMaxTokens_NormalCase() {
        let availableRAM: Int64 = 16 * 1024 * 1024 * 1024 // 16 GB
        let modelSizeMB = 4000 // ~4 GB model
        
        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(availableRAM: availableRAM, modelSizeMB: modelSizeMB)
        
        // Safety margin = 0.8
        // Usable = 16 * 0.8 = 12.8 GB
        // Available for context = 12.8 GB - 4 GB = 8.8 GB
        // 1 token = 2 bytes KV cache
        // maxTokens = 8.8 GB / 2 bytes = 4.4 GB tokens -> capped at 128,000
        
        XCTAssertEqual(maxTokens, 128_000)
    }
    
    func testCalculateMaxTokens_TightRAM() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 6000 // ~6 GB model
        
        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(availableRAM: availableRAM, modelSizeMB: modelSizeMB)
        
        // Safety margin = 0.8
        // Usable = 8 * 0.8 = 6.4 GB
        // Available for context = 6.4 GB - 6 GB = 400 MB
        // 1 token = 2 bytes KV cache
        // maxTokens = 400 MB / 2 bytes = 200 MB tokens -> approx 200,000,000 -> capped at 128,000
        
        XCTAssertEqual(maxTokens, 128_000)
    }

    func testCalculateMaxTokens_VeryTightRAM() {
        let availableRAM: Int64 = 8 * 1024 * 1024 * 1024 // 8 GB
        let modelSizeMB = 7500 // ~7.5 GB model
        
        let maxTokens = TokenBudgetCalculator.calculateMaxTokens(availableRAM: availableRAM, modelSizeMB: modelSizeMB)
        
        // Safety margin = 0.8
        // Usable = 8 * 0.8 = 6.4 GB
        // Available for context = 6.4 GB - 7.5 GB = < 0
        // maxTokens = fallback or very small
        
        XCTAssertLessThanOrEqual(maxTokens, 1024)
        XCTAssertGreaterThan(maxTokens, 0)
    }
}