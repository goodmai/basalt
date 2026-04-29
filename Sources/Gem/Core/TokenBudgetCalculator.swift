import Foundation
import os

public struct TokenBudgetCalculator: Sendable {
    
    /// Calculates the maximum number of tokens that can be generated based on available RAM and model size.
    ///
    /// - Parameters:
    ///   - availableRAM: Free RAM in bytes
    ///   - modelSizeMB: Size of the model in MB
    /// - Returns: Maximum token count (capped at 128_000, fallback to 1024 if tight)
    public static func calculateMaxTokens(availableRAM: Int64, modelSizeMB: Int) -> Int {
        let safetyMargin = 0.8 // Reserve 20% for OS and other apps
        let usableRAM = Int64(Double(availableRAM) * safetyMargin)
        let modelSizeBytes = Int64(modelSizeMB) * 1024 * 1024
        
        let availableForContext = usableRAM - modelSizeBytes
        
        if availableForContext <= 0 {
            // Not enough RAM for the model + 20% OS buffer.
            // Fallback to a very small context window to prevent immediate OOM
            return 1024
        }
        
        // Rough estimate: 1 token ≈ 2 bytes in KV cache (FP16)
        let bytesPerToken: Int64 = 2
        let maxTokens = Int(availableForContext / bytesPerToken)
        
        return min(maxTokens, 128_000) // Cap at 128k
    }
    
    /// Calculates the maximum number of tokens using the system's current available memory.
    public static func calculateMaxTokensForSystem(modelSizeMB: Int) -> Int {
        // Since os_proc_available_memory is not available on macOS,
        // we use host_statistics64 to calculate free + inactive + speculative memory
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
            // Free + Inactive represents available memory for new allocations
            let pageSize = UInt64(getpagesize())
            let availablePages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
            return Int64(availablePages * pageSize)
        } else {
            // Fallback to total physical memory / 2 if statistics fail
            return Int64(ProcessInfo.processInfo.physicalMemory / 2)
        }
    }
}
