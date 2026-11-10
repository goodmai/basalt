import Foundation

public enum ComputeUnit: String, Sendable, Equatable, Codable {
    case gpu, ane, cpu
}

public enum WorkloadClass: String, Sendable, Equatable, Codable {
    case decode
    case encoder
    case bandwidthStreaming = "bandwidth_streaming"
    case training
}

public enum KeepPolicy: String, Sendable, Equatable, Codable, CaseIterable {
    case resident, sequential
}

public enum PipelineMemory: String, Sendable, Equatable, Codable, CaseIterable {
    case resident, sequential, auto
}

public struct Workload: Sendable, Equatable, Codable {
    public static let perceiveRole = "perceive"

    public var name: String
    public var role: String
    public var unit: ComputeUnit
    public var kind: WorkloadClass
    public var residentBytes: Int64
    public var keep: KeepPolicy

    public init(name: String, role: String, unit: ComputeUnit, kind: WorkloadClass, residentBytes: Int64, keep: KeepPolicy = .resident) {
        (self.name, self.role, self.unit, self.kind, self.residentBytes, self.keep) = (name, role, unit, kind, residentBytes, keep)
    }

    var swappable: Bool { role == Self.perceiveRole || kind == .decode }

    func kept(_ policy: KeepPolicy) -> Workload {
        var copy = self
        copy.keep = policy
        return copy
    }
}

public struct PowerState: Sendable, Equatable, Codable {
    public var onAC: Bool
    public var charging: Bool
    public var batteryPercent: Int?
    public var adapterWatts: Int?

    public init(onAC: Bool, charging: Bool, batteryPercent: Int?, adapterWatts: Int?) {
        (self.onAC, self.charging, self.batteryPercent, self.adapterWatts) = (onAC, charging, batteryPercent, adapterWatts)
    }

    public static let desktop = PowerState(onAC: true, charging: false, batteryPercent: nil, adapterWatts: nil)

    public var degraded: Bool {
        guard onAC else { return true }
        guard let watts = adapterWatts, let percent = batteryPercent else { return false }
        return watts < ComputePlan.minimumAdapterWatts && !charging && percent < 50
    }
}

public struct Placement: Sendable, Equatable, Codable {
    public var workload: Workload
    public var concurrentWithDecode: Bool
    public var maxDuty: Double?
    public var expectedDecodeDrop: ClosedRange<Double>?
    public var note: String
}

public struct PlanVerdict: Sendable, Equatable, Codable {
    public var placements: [Placement]
    public var residentBytes: Int64
    public var budgetBytes: Int64
    public var degraded: Bool
    public var refusal: String?
    public var memory: PipelineMemory = .resident
    public var swapBytesPerRequest: Int64 = 0
    public var expectedSwapSeconds: Double = 0

    public func runsBesideDecode(_ role: String) -> Bool {
        placements.first { $0.workload.role == role }?.concurrentWithDecode ?? false
    }

    public func keep(of role: String) -> KeepPolicy {
        placements.first { $0.workload.role == role }?.workload.keep ?? .resident
    }

    public func keep(ofKind kind: WorkloadClass) -> KeepPolicy {
        placements.first { $0.workload.kind == kind }?.workload.keep ?? .resident
    }
}

public enum ComputePlan {
    public static let minimumAdapterWatts = 60

    public static let swapReserveBytes: Int64 = 2_000_000_000
    public static let loadBytesPerSecond = 2.0e9

    public static func admit(_ workloads: [Workload], budgetBytes: Int64, power: PowerState = .desktop, memory: PipelineMemory = .resident) -> PlanVerdict {
        let kept = keep(workloads, budgetBytes: budgetBytes, memory: memory)
        let sequential = kept.filter { $0.keep == .sequential }
        let resident = residentBytes(kept)
        let needed = resident + (sequential.isEmpty ? 0 : swapReserveBytes)
        let degraded = power.degraded
        var refusal: String?
        if budgetBytes > 0, needed > budgetBytes {
            let list = kept.map { "\($0.name) \(gigabytes($0.residentBytes))" }.joined(separator: ", ")
            refusal = sequential.isEmpty
                ? "the models need \(gigabytes(resident)) resident against a \(gigabytes(budgetBytes)) working set: \(list)"
                : "even one stage at a time the models need \(gigabytes(resident)) resident plus \(gigabytes(swapReserveBytes)) for KV and activations against a \(gigabytes(budgetBytes)) working set: \(list)"
        }
        let swapBytes = total(sequential)
        return PlanVerdict(
            placements: kept.map { place($0, degraded: degraded) }, residentBytes: resident, budgetBytes: budgetBytes, degraded: degraded, refusal: refusal,
            memory: sequential.isEmpty ? .resident : .sequential, swapBytesPerRequest: swapBytes, expectedSwapSeconds: Double(swapBytes) / loadBytesPerSecond)
    }

    static func keep(_ workloads: [Workload], budgetBytes: Int64, memory: PipelineMemory) -> [Workload] {
        let fitsResident = budgetBytes <= 0 || residentBytes(workloads) <= budgetBytes
        guard workloads.contains(where: { $0.role == Workload.perceiveRole }), memory == .sequential || (memory == .auto && !fitsResident) else {
            return workloads
        }
        let perceivers = workloads.map { $0.role == Workload.perceiveRole ? $0.kept(.sequential) : $0 }
        guard budgetBytes > 0, residentBytes(perceivers) + swapReserveBytes > budgetBytes else { return perceivers }
        return workloads.map { $0.swappable ? $0.kept(.sequential) : $0 }
    }

    static func total(_ workloads: [Workload]) -> Int64 { workloads.reduce(0) { $0 + $1.residentBytes } }

    static func residentBytes(_ workloads: [Workload]) -> Int64 {
        total(workloads.filter { $0.keep == .resident }) + (workloads.filter { $0.keep == .sequential }.map(\.residentBytes).max() ?? 0)
    }

    static func place(_ workload: Workload, degraded: Bool) -> Placement {
        var placement = placeKind(workload, degraded: degraded)
        if workload.keep == .sequential {
            placement.note = workload.kind == .decode ? "unloaded while a perceiver runs, reloaded after it" : "loaded for its stage and unloaded after it, in the gate's exclusive section"
        }
        return placement
    }

    static func placeKind(_ workload: Workload, degraded: Bool) -> Placement {
        switch (workload.kind, workload.unit) {
        case (.decode, _):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: nil, expectedDecodeDrop: nil, note: "decoder")
        case (.encoder, .ane) where !degraded:
            return Placement(workload: workload, concurrentWithDecode: true, maxDuty: 1, expectedDecodeDrop: 0.0...0.07, note: "compute-bound on the Neural Engine: runs beside decode (#273, #399: -2..5% at 25% duty, -4..7% continuous)")
        case (.encoder, .ane):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: nil, expectedDecodeDrop: nil, note: "power-degraded: queued behind decode")
        case (.encoder, _):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: nil, expectedDecodeDrop: 0.27...0.32, note: "encoder on the GPU: gated between generations (beside decode it would cost -27..32%)")
        case (.bandwidthStreaming, _):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: degraded ? 0 : 0.1, expectedDecodeDrop: 0.06...0.08, note: "bandwidth-bound: queued, or paced to at most 10% duty")
        case (.training, _):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: nil, expectedDecodeDrop: nil, note: "distillation is time-sliced on the shared base weights, one optimizer step per hold, so requests wait at most one step; train and JEPA still hold the model for the whole run")
        }
    }

    public static func gigabytes(_ bytes: Int64) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }

    public static func describe(_ verdict: PlanVerdict) -> [String] {
        let sequential = verdict.memory == .sequential
            ? ", sequential: about \(String(format: "%.0f", verdict.expectedSwapSeconds)) s of model loading per request (\(gigabytes(verdict.swapBytesPerRequest)) at \(gigabytes(Int64(loadBytesPerSecond)))/s from a cold disk; about twice as fast from a warm page cache)"
            : ", resident"
        let header = "compute plan: \(gigabytes(verdict.residentBytes)) resident of \(gigabytes(verdict.budgetBytes))" + sequential + (verdict.degraded ? ", power-degraded (no concurrent sidecars)" : "")
        return [header] + verdict.placements.map { "  \($0.workload.role) \($0.workload.name) on \($0.workload.unit.rawValue): \($0.note)" }
    }
}
