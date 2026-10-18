import Foundation

// MARK: — Hardware Benchmark Profile & Calibration (Epic 16.7 / Issue #31)

/// Record of an empirical benchmark measurement on this hardware
public struct BenchmarkMeasurement: Codable, Sendable, Equatable {
    public let modelId: String
    public let chipModel: String
    public let measuredTPS: Double
    public let measuredTTFT: Double
    public let timestamp: Date
    public let quant: String?

    public init(
        modelId: String,
        chipModel: String,
        measuredTPS: Double,
        measuredTTFT: Double,
        timestamp: Date = Date(),
        quant: String? = nil
    ) {
        self.modelId = modelId
        self.chipModel = chipModel
        self.measuredTPS = measuredTPS
        self.measuredTTFT = measuredTTFT
        self.timestamp = timestamp
        self.quant = quant
    }
}

/// Profile persisting real hardware performance measurements for self-calibration
public struct HardwareBenchmarkProfile: Codable, Sendable {
    public var chipModel: String
    public var measurements: [String: BenchmarkMeasurement] // key: modelId or repo name
    public var lastUpdated: Date

    public init(
        chipModel: String = "Apple Silicon",
        measurements: [String: BenchmarkMeasurement] = [:],
        lastUpdated: Date = Date()
    ) {
        self.chipModel = chipModel
        self.measurements = measurements
        self.lastUpdated = lastUpdated
    }

    /// α_calib = 1.0 until the first benchmark, then TPS_measured / TPS_theoretical
    public func calibrationFactor(for modelId: String, theoreticalTPS: Double, currentChip: String? = nil) -> Double {
        guard theoreticalTPS > 0, let m = measurement(for: modelId, currentChip: currentChip) else {
            return 1.0
        }
        return m.measuredTPS / theoreticalTPS
    }

    /// Retrieve matching measurement by exact ID or clean repo basename match, validating chip compatibility
    public func measurement(for modelId: String, currentChip: String? = nil) -> BenchmarkMeasurement? {
        func chipMatches(_ m: BenchmarkMeasurement) -> Bool {
            guard let cur = currentChip?.trimmingCharacters(in: .whitespacesAndNewlines), !cur.isEmpty else {
                return true
            }
            let curLower = cur.lowercased()
            let mLower = m.chipModel.lowercased()
            if curLower == mLower { return true }
            for series in ["m1", "m2", "m3", "m4", "m5"] {
                if curLower.contains(series) && mLower.contains(series) {
                    for tier in ["ultra", "max", "pro"] {
                        if curLower.contains(tier) != mLower.contains(tier) {
                            return false
                        }
                    }
                    return true
                }
            }
            return false
        }

        if let direct = measurements[modelId], chipMatches(direct) {
            return direct
        }

        let target = modelId.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let targetBase = target.split(separator: "/").last.map(String.init) ?? target

        for (key, m) in measurements {
            guard chipMatches(m) else { continue }
            let keyLower = key.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            let keyBase = keyLower.split(separator: "/").last.map(String.init) ?? keyLower

            if keyLower == target {
                return m
            }
            if !targetBase.isEmpty && keyBase == targetBase {
                return m
            }
            // Slash boundary suffix match only (e.g. "org/repo" vs "repo")
            if target.hasSuffix("/" + keyLower) || keyLower.hasSuffix("/" + target) {
                return m
            }
        }
        return nil
    }
}

public enum HardwareBenchmarkProfileError: Error, LocalizedError, Equatable {
    case implausibleMeasurement(tps: Double)

    public var errorDescription: String? {
        switch self {
        case .implausibleMeasurement(let tps):
            return String(format: "measured %.2f tok/s — the run produced (almost) no tokens; not recorded", tps)
        }
    }
}

/// Thread-safe store for reading and writing the hardware benchmark profile
public actor HardwareBenchmarkProfileStore {
    public static let shared = HardwareBenchmarkProfileStore()

    public static var defaultProfilePath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".gem/hardware_profile.json").path
    }

    private let profilePath: String
    private var cachedProfile: HardwareBenchmarkProfile?

    // One date strategy for both directions: a default decoder (deferredToDate) cannot read
    // the ISO-8601 the encoder writes, and loadSync's try? would turn that into an empty profile.
    static var decoder: JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    static var encoder: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }

    /// Below this a run produced (almost) nothing — e.g. an immediate EOS — and calibrating
    /// from it would show ~1 tok/s as "measured ✓".
    public static let minimumMeasurableTPS = 1.0

    public init(customPath: String? = nil) {
        if let customPath = customPath {
            self.profilePath = (customPath as NSString).expandingTildeInPath
        } else {
            self.profilePath = Self.defaultProfilePath
        }
    }

    /// Read profile synchronously without awaiting actor (useful during fast scoring)
    public static func loadSync(path: String? = nil) -> HardwareBenchmarkProfile {
        let actualPath = path ?? defaultProfilePath
        let url = URL(fileURLWithPath: actualPath)
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let profile = try? Self.decoder.decode(HardwareBenchmarkProfile.self, from: data) else {
            return HardwareBenchmarkProfile()
        }
        return profile
    }

    public func load() -> HardwareBenchmarkProfile {
        if let cached = cachedProfile {
            return cached
        }
        let profile = Self.loadSync(path: profilePath)
        self.cachedProfile = profile
        return profile
    }

    public func record(_ measurement: BenchmarkMeasurement) throws {
        guard measurement.measuredTPS >= Self.minimumMeasurableTPS else {
            throw HardwareBenchmarkProfileError.implausibleMeasurement(tps: measurement.measuredTPS)
        }
        var profile = load()
        profile.chipModel = measurement.chipModel
        profile.measurements[measurement.modelId] = measurement
        profile.lastUpdated = Date()
        self.cachedProfile = profile

        let url = URL(fileURLWithPath: profilePath)
        let directory = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        try Self.encoder.encode(profile).write(to: url, options: .atomic)
    }

    public func reset() {
        self.cachedProfile = nil
        let url = URL(fileURLWithPath: profilePath)
        try? FileManager.default.removeItem(at: url)
    }
}
