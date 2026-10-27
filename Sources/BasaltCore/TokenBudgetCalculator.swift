import Foundation
import os

public struct TokenBudgetCalculator: Sendable {
    
    public static let upperCapTokens: Int = 128_000
    public static let safetyMargin = 0.80
    
    public static func calculateMaxTokens(
        availableRAM: Int64, modelSizeMB: Int, kv: KVCacheGeometry, kvBits: Int? = nil, promptTokens: Int = 0
    ) -> Int {
        guard availableRAM > 0 else { return 512 }
        let usableRAM = Int64(Double(availableRAM) * safetyMargin)
        let availableForContext = usableRAM - Int64(modelSizeMB) * 1024 * 1024
        guard availableForContext > 0 else { return 512 }
        let feasibleTokens = kv.maxTokens(fitting: availableForContext, kvBits: kvBits) - max(0, promptTokens)
        return min(max(feasibleTokens, 512), upperCapTokens)
    }
}
