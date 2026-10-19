import Foundation
import os

public struct TokenBudgetCalculator: Sendable {
    
    /// Upper limit for context window / token budget
    public static let upperCapTokens: Int = 128_000
    
    /// Calculates the maximum number of tokens that can be generated based on available RAM and model size.
    /// Dynamic RAM check ensures launch feasibility without exceeding memory limits.
    ///
    /// - Parameters:
    ///   - availableRAM: Free + Inactive RAM in bytes
    ///   - modelSizeMB: Size of the model in MB
    /// - Returns: Maximum feasible token count (dynamically measured, upper cap 128,000)
    public static func calculateMaxTokens(availableRAM: Int64, modelSizeMB: Int) -> Int {
        guard availableRAM > 0 else { return 512 }
        
        let safetyMargin = 0.80
        let usableRAM = Int64(Double(availableRAM) * safetyMargin)
        let modelSizeBytes = Int64(modelSizeMB) * 1024 * 1024
        let availableForContext = usableRAM - modelSizeBytes
        
        if availableForContext <= 0 {
            // Memory is tight, reserve minimum 512 tokens
            return 512
        }
        
        let bytesPerToken: Int64 = 2
        let feasibleTokens = Int(availableForContext / bytesPerToken)
        let calculated = max(feasibleTokens, 512)
        return min(calculated, upperCapTokens)
    }
    
    /// Calculates the maximum number of tokens using the system's current available memory.
    public static func calculateMaxTokensForSystem(modelSizeMB: Int) -> Int {
        let availableRAM = getAvailableSystemMemory()
        return calculateMaxTokens(availableRAM: availableRAM, modelSizeMB: modelSizeMB)
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
