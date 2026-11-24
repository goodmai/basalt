import Foundation
import Metal

public struct ExpertArenaHost: Sendable, Equatable, Codable {
    public var totalRAMBytes: ByteCount
    public var availableMemoryBytes: ByteCount
    public var metalWorkingSetBytes: ByteCount

    public init(
        totalRAMBytes: ByteCount,
        availableMemoryBytes: ByteCount,
        metalWorkingSetBytes: ByteCount
    ) {
        self.totalRAMBytes = totalRAMBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.metalWorkingSetBytes = metalWorkingSetBytes
    }

    public init(resources: SystemProfiler.SystemResources) {
        self.init(
            totalRAMBytes: ByteCount(resources.totalRAM),
            availableMemoryBytes: ByteCount(resources.availableRAM),
            metalWorkingSetBytes: ByteCount(resources.gpuMemory)
        )
    }

    public static func current() -> ExpertArenaHost {
        let total = Int64(clamping: ProcessInfo.processInfo.physicalMemory)
        let available = availableMemoryIncludingPurgeable(totalRAM: total)
        let metal = MTLCreateSystemDefaultDevice().map { Int64(clamping: $0.recommendedMaxWorkingSetSize) } ?? 0
        return ExpertArenaHost(
            totalRAMBytes: ByteCount(total),
            availableMemoryBytes: ByteCount(available),
            metalWorkingSetBytes: ByteCount(metal)
        )
    }

    public static func availableMemoryIncludingPurgeable(totalRAM: Int64) -> Int64 {
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard status == KERN_SUCCESS else {
            return max(0, totalRAM / 2)
        }
        let pages = Int64(vmStats.free_count) + Int64(vmStats.inactive_count) + Int64(vmStats.purgeable_count)
        let bytes = pages.multipliedReportingOverflow(by: Int64(getpagesize()))
        let raw = bytes.overflow ? totalRAM : bytes.partialValue
        return min(max(0, totalRAM), max(0, raw))
    }
}

public struct ExpertArenaWorkload: Sendable, Equatable, Codable {
    public var denseWeightsBytes: ByteCount
    public var kvCacheBytes: ByteCount
    public var scratchOverheadBytes: ByteCount
    public var expertBytesTotal: ByteCount
    public var ngramRowCacheBytes: ByteCount

    public init(
        denseWeightsBytes: ByteCount,
        kvCacheBytes: ByteCount = .zero,
        scratchOverheadBytes: ByteCount = .zero,
        expertBytesTotal: ByteCount,
        ngramRowCacheBytes: ByteCount = .zero
    ) {
        self.denseWeightsBytes = denseWeightsBytes
        self.kvCacheBytes = kvCacheBytes
        self.scratchOverheadBytes = scratchOverheadBytes
        self.expertBytesTotal = expertBytesTotal
        self.ngramRowCacheBytes = ngramRowCacheBytes
    }
}

public struct ExpertArenaInput: Sendable, Equatable, Codable {
    public var host: ExpertArenaHost
    public var workload: ExpertArenaWorkload
    public var explicitOverrideGB: Double?

    public init(
        host: ExpertArenaHost,
        workload: ExpertArenaWorkload,
        explicitOverrideGB: Double? = nil
    ) {
        self.host = host
        self.workload = workload
        self.explicitOverrideGB = explicitOverrideGB
    }

    public var totalRAMBytes: ByteCount { host.totalRAMBytes }
    public var availableMemoryBytes: ByteCount { host.availableMemoryBytes }
    public var metalWorkingSetBytes: ByteCount { host.metalWorkingSetBytes }
    public var denseWeightsBytes: ByteCount { workload.denseWeightsBytes }
    public var kvCacheBytes: ByteCount { workload.kvCacheBytes }
    public var scratchOverheadBytes: ByteCount { workload.scratchOverheadBytes }
    public var expertBytesTotal: ByteCount { workload.expertBytesTotal }
    public var ngramRowCacheBytes: ByteCount { workload.ngramRowCacheBytes }
}

public struct ExpertArenaPlan: Sendable, Equatable, Codable {
    public enum Mode: String, Sendable, Equatable, Codable {
        case resident
        case streaming
    }

    public let mode: Mode
    public let arenaBytes: ByteCount
    public let workingSetCapBytes: ByteCount
    public let macOSReserveBytes: ByteCount
    public let fixedFootprintBytes: ByteCount
    public let projectedFootprintBytes: ByteCount
    public let recommendedPromptChunkTokens: Int
    public let usedExplicitOverride: Bool
    public let overrideWasClamped: Bool
    public let fitsBudget: Bool
}

public enum ExpertArenaPlanner {
    public static let gib: Int64 = 1_073_741_824
    public static let stepBytes = ByteCount(3 * (1 << 29))
    public static let macOSMinimumReserve = ByteCount(7 * gib)
    public static let macOSReserveShare = 0.15
    public static let appsHeadroomBelow64GB = ByteCount(5 * gib)
    public static let cap24GB = ByteCount(14 * gib)
    public static let residentShareAt64GBPlus = 0.70
    public static let streamingShareAt64GBPlus = 0.60
    public static let tier24GB = ByteCount(24 * gib)
    public static let tier64GB = ByteCount(64 * gib)
    public static let tier96GB = ByteCount(96 * gib)
    public static let streamingChunkBelow96GB = 1024
    public static let streamingChunkAt96GBPlus = 4096
    public static let residentChunkTokens = 2048

    private struct PlanContext {
        let total: ByteCount
        let reserve: ByteCount
        let residentBudget: ByteCount
        let streamingBudget: ByteCount
        let fixed: ByteCount
        let experts: ByteCount
    }

    public static func plan(_ input: ExpertArenaInput) -> ExpertArenaPlan {
        let total = max(.zero, input.totalRAMBytes)
        let reserve = macOSReserve(for: total)
        let ramMinusOS = clampedSubtract(total, reserve)
        let residentTierCap = residentCap(total: total, ramMinusOS: ramMinusOS)
        let streamingTierCap = streamingCap(total: total, ramMinusOS: ramMinusOS, residentCap: residentTierCap)
        let metalCap = input.metalWorkingSetBytes > .zero ? input.metalWorkingSetBytes : ramMinusOS
        let availCap = min(total, max(.zero, input.availableMemoryBytes))
        let fixed = saturatingSum([
            max(.zero, input.denseWeightsBytes),
            max(.zero, input.kvCacheBytes),
            max(.zero, input.scratchOverheadBytes),
            max(.zero, input.ngramRowCacheBytes),
        ])
        let context = PlanContext(
            total: total,
            reserve: reserve,
            residentBudget: min(residentTierCap, min(metalCap, availCap)),
            streamingBudget: min(streamingTierCap, min(metalCap, availCap)),
            fixed: fixed,
            experts: max(.zero, input.expertBytesTotal)
        )
        if let overrideGB = input.explicitOverrideGB, overrideGB.isFinite, overrideGB > 0 {
            return planExplicitOverride(overrideGB: overrideGB, context: context)
        }
        return planAutomatic(context: context)
    }

    public static func planForCurrentHost(
        workload: ExpertArenaWorkload,
        explicitOverrideGB: Double? = nil
    ) -> ExpertArenaPlan {
        plan(ExpertArenaInput(host: .current(), workload: workload, explicitOverrideGB: explicitOverrideGB))
    }

    public static func macOSReserve(for totalRAM: ByteCount) -> ByteCount {
        let clampedTotal = max(.zero, totalRAM)
        let shareBytes = scaleBytes(clampedTotal, by: macOSReserveShare)
        return max(macOSMinimumReserve, shareBytes)
    }

    public static func recommendedPromptChunkTokens(mode: ExpertArenaPlan.Mode, totalRAM: ByteCount) -> Int {
        switch mode {
        case .resident:
            residentChunkTokens
        case .streaming:
            totalRAM >= tier96GB ? streamingChunkAt96GBPlus : streamingChunkBelow96GB
        }
    }

    private static func planExplicitOverride(overrideGB: Double, context: PlanContext) -> ExpertArenaPlan {
        let scaled = (overrideGB * Double(gib)).rounded(.towardZero)
        let requestedBytes = scaled < Double(Int64.max) ? ByteCount(Int64(scaled)) : ByteCount(Int64.max)
        let ceiling = min(context.total, context.experts)
        let arena = min(requestedBytes, ceiling)
        let clamped = requestedBytes > ceiling
        let mode: ExpertArenaPlan.Mode = (context.experts == .zero || arena >= context.experts) ? .resident : .streaming
        let cap = mode == .resident ? context.residentBudget : context.streamingBudget
        let projected = saturatingSum([context.fixed, arena])
        return ExpertArenaPlan(
            mode: mode,
            arenaBytes: arena,
            workingSetCapBytes: cap,
            macOSReserveBytes: context.reserve,
            fixedFootprintBytes: context.fixed,
            projectedFootprintBytes: projected,
            recommendedPromptChunkTokens: recommendedPromptChunkTokens(mode: mode, totalRAM: context.total),
            usedExplicitOverride: true,
            overrideWasClamped: clamped,
            fitsBudget: projected <= cap
        )
    }

    private static func planAutomatic(context: PlanContext) -> ExpertArenaPlan {
        let fitsResident = context.experts == .zero
            || (context.fixed <= context.residentBudget && context.experts <= clampedSubtract(context.residentBudget, context.fixed))
        if fitsResident {
            let projected = saturatingSum([context.fixed, context.experts])
            return ExpertArenaPlan(
                mode: .resident,
                arenaBytes: context.experts,
                workingSetCapBytes: context.residentBudget,
                macOSReserveBytes: context.reserve,
                fixedFootprintBytes: context.fixed,
                projectedFootprintBytes: projected,
                recommendedPromptChunkTokens: recommendedPromptChunkTokens(mode: .resident, totalRAM: context.total),
                usedExplicitOverride: false,
                overrideWasClamped: false,
                fitsBudget: projected <= context.residentBudget
            )
        }
        let headroom = clampedSubtract(context.streamingBudget, context.fixed)
        let stepped = ByteCount((headroom.rawValue / stepBytes.rawValue) * stepBytes.rawValue)
        let arena = min(stepped, context.experts)
        let projected = saturatingSum([context.fixed, arena])
        return ExpertArenaPlan(
            mode: .streaming,
            arenaBytes: arena,
            workingSetCapBytes: context.streamingBudget,
            macOSReserveBytes: context.reserve,
            fixedFootprintBytes: context.fixed,
            projectedFootprintBytes: projected,
            recommendedPromptChunkTokens: recommendedPromptChunkTokens(mode: .streaming, totalRAM: context.total),
            usedExplicitOverride: false,
            overrideWasClamped: false,
            fitsBudget: projected <= context.streamingBudget
        )
    }

    private static func residentCap(total: ByteCount, ramMinusOS: ByteCount) -> ByteCount {
        if total <= tier24GB {
            return min(cap24GB, ramMinusOS)
        }
        return min(scaleBytes(total, by: residentShareAt64GBPlus), ramMinusOS)
    }

    private static func streamingCap(
        total: ByteCount,
        ramMinusOS: ByteCount,
        residentCap: ByteCount
    ) -> ByteCount {
        if total <= tier24GB {
            return min(cap24GB, ramMinusOS)
        }
        if total < tier64GB {
            return min(residentCap, clampedSubtract(ramMinusOS, appsHeadroomBelow64GB))
        }
        return min(scaleBytes(total, by: streamingShareAt64GBPlus), ramMinusOS)
    }

    private static func scaleBytes(_ value: ByteCount, by ratio: Double) -> ByteCount {
        guard value.rawValue > 0, ratio.isFinite, ratio > 0 else { return .zero }
        let scaled = (Double(value.rawValue) * ratio).rounded(.towardZero)
        guard scaled < Double(Int64.max) else { return ByteCount(Int64.max) }
        return ByteCount(Int64(scaled))
    }

    private static func clampedSubtract(_ lhs: ByteCount, _ rhs: ByteCount) -> ByteCount {
        lhs.rawValue > rhs.rawValue ? ByteCount(lhs.rawValue - rhs.rawValue) : .zero
    }

    private static func saturatingSum(_ values: [ByteCount]) -> ByteCount {
        var total: Int64 = 0
        for value in values where value.rawValue > 0 {
            let (next, overflow) = total.addingReportingOverflow(value.rawValue)
            total = overflow ? Int64.max : next
        }
        return ByteCount(total)
    }
}
