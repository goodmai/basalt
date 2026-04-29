// MODULE: SystemProfiler
// OWNER: AI Assistant
// STATUS: IN_PROGRESS
// LAST_MODIFIED: 2025-04-28
// TODO: Add Metal GPU detection
// TODO: Add chip model detection (M1/M2/M3/M4/M5)

import Foundation
import Metal

/// System resource profiler for hardware detection and model recommendations
/// Used during onboarding to recommend optimal models based on available resources
public actor SystemProfiler {
    
    /// System resources detected from hardware
    public struct SystemResources: Codable, Sendable {
        public let totalRAM: Int64           // bytes
        public let availableRAM: Int64       // bytes  
        public let cpuCores: Int
        public let gpuName: String
        public let gpuMemory: Int64          // bytes
        public let diskSpace: Int64          // bytes available
        public let osVersion: String
        public let chipModel: String         // M1, M2, M3, M4, M5
        
        /// Human-readable RAM in GB
        public var totalRAMGB: Int {
            Int(totalRAM / 1_073_741_824)
        }
        
        /// Human-readable GPU memory in GB
        public var gpuMemoryGB: Int {
            Int(gpuMemory / 1_073_741_824)
        }
        
        /// Human-readable disk space in GB
        public var diskSpaceGB: Int {
            Int(diskSpace / 1_073_741_824)
        }
    }
    
    /// Model recommendation based on hardware specs
    public struct ModelRecommendation: Sendable {
        public let modelId: String
        public let modelName: String
        public let estimatedRAM: Int      // MB
        public let estimatedTPS: Int      // tokens per second
        public let reason: String
    }
    
    public init() {}
    
    /// Detect system resources
    public func detectResources() async -> SystemResources {
        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let cpuCores = ProcessInfo.processInfo.processorCount
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        
        // Metal GPU detection
        let gpuName: String
        let gpuMemory: Int64
        if let device = MTLCreateSystemDefaultDevice() {
            gpuName = device.name
            gpuMemory = Int64(device.recommendedMaxWorkingSetSize)
        } else {
            gpuName = "Unknown"
            gpuMemory = 0
        }
        
        // Disk space
        let diskSpace = getAvailableDiskSpace()
        
        // Available RAM
        let availableRAM = getAvailableRAM()
        
        // Chip model detection
        let chipModel = detectChipModel()
        
        return SystemResources(
            totalRAM: Int64(totalRAM),
            availableRAM: availableRAM,
            cpuCores: cpuCores,
            gpuName: gpuName,
            gpuMemory: gpuMemory,
            diskSpace: diskSpace,
            osVersion: osVersion,
            chipModel: chipModel
        )
    }
    
    /// Recommend optimal model based on system resources
    public func recommendModel(resources: SystemResources) -> ModelRecommendation {
        let ramGB = Int(resources.totalRAM / 1_073_741_824)
        
        switch ramGB {
        case 0..<8:
            return ModelRecommendation(
                modelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                modelName: "Qwen 2.5 0.5B (4-bit)",
                estimatedRAM: 512,
                estimatedTPS: 120,
                reason: "Optimized for systems with <8GB RAM"
            )
            
        case 8..<16:
            return ModelRecommendation(
                modelId: "mlx-community/Qwen2.5-3B-Instruct-4bit",
                modelName: "Qwen 2.5 3B (4-bit)",
                estimatedRAM: 2300,
                estimatedTPS: 80,
                reason: "Good balance for 8-16GB RAM systems"
            )
            
        case 16..<32:
            return ModelRecommendation(
                modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                modelName: "Qwen 2.5 7B (4-bit)",
                estimatedRAM: 5800,
                estimatedTPS: 45,
                reason: "High quality for 16-32GB RAM systems"
            )
            
        case 32..<64:
            return ModelRecommendation(
                modelId: "mlx-community/Qwen2.5-14B-Instruct-4bit",
                modelName: "Qwen 2.5 14B (4-bit)",
                estimatedRAM: 9500,
                estimatedTPS: 25,
                reason: "Best quality for 32-64GB RAM systems"
            )
            
        default:
            return ModelRecommendation(
                modelId: "mlx-community/Qwen2.5-32B-Instruct-4bit",
                modelName: "Qwen 2.5 32B (4-bit)",
                estimatedRAM: 19000,
                estimatedTPS: 15,
                reason: "Maximum quality for high-end systems (64GB+ RAM)"
            )
        }
    }
    
    // MARK: - Private Helpers
    
    /// Get available RAM using macOS APIs
    private func getAvailableRAM() -> Int64 {
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(
                    mach_host_self(),
                    HOST_VM_INFO64,
                    $0,
                    &count
                )
            }
        }
        
        guard result == KERN_SUCCESS else {
            // Fallback: return 50% of total RAM as estimate
            return Int64(ProcessInfo.processInfo.physicalMemory) / 2
        }
        
        // Use constant page size value instead of global variable
        let pageSize: Int64 = 16384  // 16KB on Apple Silicon
        let freePages = Int64(vmStats.free_count)
        let inactivePages = Int64(vmStats.inactive_count)
        
        return (freePages + inactivePages) * pageSize
    }
    
    /// Get available disk space
    private func getAvailableDiskSpace() -> Int64 {
        let fileURL = URL(fileURLWithPath: NSHomeDirectory())
        
        do {
            let values = try fileURL.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            return Int64(values.volumeAvailableCapacity ?? 0)
        } catch {
            return 0
        }
    }
    
    /// Detect Apple Silicon chip model (M1/M2/M3/M4/M5)
    private func detectChipModel() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        guard let machineString = machine else {
            return "Unknown"
        }
        
        // Parse machine identifier (e.g., "arm64" or specific model)
        // Try to get more specific chip info via sysctlbyname
        var size: size_t = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        
        var cpuBrand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &cpuBrand, &size, nil, 0)
        
        // Convert to String, handling null termination
        let brandString = cpuBrand.withUnsafeBufferPointer { buffer in
            String(validatingUTF8: buffer.baseAddress!) ?? "Unknown"
        }
        
        // Detect M-series chips (including future M5)
        if brandString.contains("Apple M1") {
            return "M1"
        } else if brandString.contains("Apple M2") {
            return "M2"
        } else if brandString.contains("Apple M3") {
            return "M3"
        } else if brandString.contains("Apple M4") {
            return "M4"
        } else if brandString.contains("Apple M5") {
            return "M5"
        } else if brandString.contains("Apple M") {
            // Future M-series (M6, M7, etc.)
            return "Apple Silicon (M-series)"
        } else if machineString.contains("arm64") {
            return "Apple Silicon"
        } else {
            return "Intel"
        }
    }
}
