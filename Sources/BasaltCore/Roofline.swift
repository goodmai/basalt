import Foundation
import IOKit

public struct AppleSiliconChip: Sendable, Equatable {
    public enum Tier: Sendable, Equatable { case base, pro, max, ultra }

    public let generation: Int
    public let tier: Tier

    public init?(_ name: String) {
        let upper = name.uppercased()
        guard let match = upper.range(of: #"\bM\d+\b"#, options: .regularExpression),
              let generation = Int(upper[match].dropFirst()) else { return nil }
        self.generation = generation
        tier = upper.contains("ULTRA") ? .ultra : upper.contains("MAX") ? .max : upper.contains("PRO") ? .pro : .base
    }

    static let publishedBandwidthGBs: [Int: [Tier: Double]] = [
        1: [.base: 68, .pro: 200, .max: 400, .ultra: 800],
        2: [.base: 100, .pro: 200, .max: 400, .ultra: 800],
        3: [.base: 100, .pro: 150, .max: 400, .ultra: 819],
        4: [.base: 120, .pro: 273, .max: 546, .ultra: 1092],
        5: [.base: 153, .pro: 307, .max: 614, .ultra: 1200],
        6: [.base: 170],
    ]
    static let binnedMaxBandwidthGBs: [Int: Double] = [3: 300, 4: 410, 5: 460]
    static let fullMaxGPUCores = 40

    public func memoryBandwidthGBs(gpuCores: Int? = nil) -> Double? {
        guard let row = Self.publishedBandwidthGBs[generation] else { return nil }
        if tier == .max, let gpuCores, gpuCores < Self.fullMaxGPUCores, let binned = Self.binnedMaxBandwidthGBs[generation] {
            return binned
        }
        return row[tier] ?? row[.base]
    }

    public var hasNeuralAccelerators: Bool { generation >= 5 }

    public static func neuralAcceleratorsActive(architecture: String, os: OperatingSystemVersion) -> Bool {
        guard let match = architecture.range(of: #"(?<=applegpu_g)\d+"#, options: .regularExpression),
              let gpuGeneration = Int(architecture[match]) else { return false }
        return gpuGeneration >= 17 && (os.majorVersion, os.minorVersion) >= (26, 2)
    }

    static func gpuCoreCount() -> Int? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }
        return IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? Int
    }

    public static func gpuSummary(architecture: String?, cores: Int?, neuralAccelerators: Bool?) -> String {
        let accelerators = neuralAccelerators.map { $0 ? "on" : "off" } ?? "unknown"
        return "\(architecture ?? "unknown architecture") | \(cores.map { "\($0) cores" } ?? "cores unknown") | neural accelerators: \(accelerators)"
    }
}

extension FitScorer {
    public static let neuralAcceleratorPrefillFactor = 3.5

    public static func prefillSpeed(activeParameters: Double, computeTFLOPS: Double, neuralAccelerators: Bool) -> Double {
        computeTFLOPS * 1e3 * (neuralAccelerators ? neuralAcceleratorPrefillFactor : 1) / (2 * max(0.1, activeParameters))
    }

    public static func prefillSpeed(
        model: ModelDefinition,
        resources: SystemProfiler.SystemResources,
        profile: HardwareBenchmarkProfile
    ) -> (tps: Double, measured: Bool)? {
        if let own = profile.measurement(for: model.id, currentChip: resources.chipModel)?.prefillTokensPerSecond {
            return (own, true)
        }
        let accelerated = resources.neuralAccelerators ?? false
        let calibrations = profile.matching(chip: resources.chipModel).values.compactMap { m -> Double? in
            guard let rate = m.prefillTokensPerSecond else { return nil }
            let total = ModelDefinition.inferTotalParameters(id: m.modelId, name: "", ramMB: 0)
            let active = ModelDefinition.inferActiveParameters(id: m.modelId, name: "", total: total)
            return rate * 2 * active / 1e3 / (accelerated ? neuralAcceleratorPrefillFactor : 1)
        }.sorted()
        guard !calibrations.isEmpty else { return nil }
        let tflops = calibrations[calibrations.count / 2]
        return (prefillSpeed(activeParameters: model.activeParameters, computeTFLOPS: tflops, neuralAccelerators: accelerated), false)
    }
}
