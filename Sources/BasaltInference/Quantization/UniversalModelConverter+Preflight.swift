import BasaltCore
import Foundation

extension UniversalModelConverter {
    static func workingCopyBytes(_ detected: DetectedSourceCheckpoint) -> Int64 {
        guard detected.requiresCanonicalDecode else { return detected.totalBytes }
        return Int64((Double(detected.totalBytes) * 16 / max(1, detected.effectiveBPW)).rounded())
    }

    static func stagingProblem(workingBytes: Int64, freeDiskBytes: Int64, totalRAMBytes: Int64) -> String? {
        let need = ByteCountFormatter.string(fromByteCount: workingBytes, countStyle: .file)
        if workingBytes > freeDiskBytes {
            let free = ByteCountFormatter.string(fromByteCount: freeDiskBytes, countStyle: .file)
            return "convert: the source decodes to a \(need) dense working copy, but the temporary volume has \(free) free"
        }
        if Double(workingBytes) > Double(totalRAMBytes) * 0.88 {
            let ram = ByteCountFormatter.string(fromByteCount: totalRAMBytes, countStyle: .memory)
            return "convert: the \(need) dense working copy must stay resident while it is quantized, and this Mac has \(ram) of memory; convert on a Mac with more memory"
        }
        return nil
    }

    static func checkStaging(_ detected: DetectedSourceCheckpoint) throws {
        let tmp = FileManager.default.temporaryDirectory
        let free = (try? tmp.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? .max
        if let problem = stagingProblem(
            workingBytes: workingCopyBytes(detected), freeDiskBytes: free,
            totalRAMBytes: Int64(ProcessInfo.processInfo.physicalMemory)) {
            throw BasaltError.invalidRequestStructure(details: problem)
        }
    }
}
