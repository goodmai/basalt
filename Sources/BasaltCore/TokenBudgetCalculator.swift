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

    public static func calculateMaxTokensForSystem(
        modelSizeMB: Int, kv: KVCacheGeometry, kvBits: Int? = nil, promptTokens: Int = 0
    ) -> Int {
        calculateMaxTokens(
            availableRAM: getAvailableSystemMemory(), modelSizeMB: modelSizeMB,
            kv: kv, kvBits: kvBits, promptTokens: promptTokens
        )
    }

    private static func getAvailableSystemMemory() -> Int64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let pageSize = UInt64(getpagesize())
            let availablePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
            return Int64(availablePages * pageSize)
        } else {
            return Int64(ProcessInfo.processInfo.physicalMemory / 2)
        }
    }
}
