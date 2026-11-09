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

public struct Workload: Sendable, Equatable, Codable {
    public var name: String
    public var role: String
    public var unit: ComputeUnit
    public var kind: WorkloadClass
    public var residentBytes: Int64

    public init(name: String, role: String, unit: ComputeUnit, kind: WorkloadClass, residentBytes: Int64) {
        (self.name, self.role, self.unit, self.kind, self.residentBytes) = (name, role, unit, kind, residentBytes)
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

public struct PlanVerdict: Sendable, Equatable {
    public var placements: [Placement]
    public var residentBytes: Int64
    public var budgetBytes: Int64
    public var degraded: Bool
    public var refusal: String?
}

public enum ComputePlan {
    public static let minimumAdapterWatts = 60

    public static func admit(_ workloads: [Workload], budgetBytes: Int64, power: PowerState = .desktop) -> PlanVerdict {
        let resident = workloads.reduce(Int64(0)) { $0 + $1.residentBytes }
        let degraded = power.degraded
        let placements = workloads.map { place($0, degraded: degraded) }
        var refusal: String?
        if budgetBytes > 0, resident > budgetBytes {
            refusal = "the models need \(gigabytes(resident)) resident against a \(gigabytes(budgetBytes)) working set: " + workloads.map { "\($0.name) \(gigabytes($0.residentBytes))" }.joined(separator: ", ")
        }
        return PlanVerdict(placements: placements, residentBytes: resident, budgetBytes: budgetBytes, degraded: degraded, refusal: refusal)
    }

    static func place(_ workload: Workload, degraded: Bool) -> Placement {
        switch (workload.kind, workload.unit) {
        case (.decode, _):
            return Placement(workload: workload, concurrentWithDecode: false, maxDuty: nil, expectedDecodeDrop: nil, note: "decoder")
        case (.encoder, .ane) where !degraded:
            return Placement(workload: workload, concurrentWithDecode: true, maxDuty: 1, expectedDecodeDrop: 0.0...0.07, note: "compute-bound on the Neural Engine: runs beside decode (#273: -0..3% at 25% duty, -3..7% continuous)")
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
        let header = "compute plan: \(gigabytes(verdict.residentBytes)) resident of \(gigabytes(verdict.budgetBytes))" + (verdict.degraded ? ", power-degraded (no concurrent sidecars)" : "")
        return [header] + verdict.placements.map { "  \($0.workload.role) \($0.workload.name) on \($0.workload.unit.rawValue): \($0.note)" }
    }
}
