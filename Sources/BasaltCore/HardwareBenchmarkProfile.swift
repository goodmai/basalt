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

        let normalize = { (id: String) in id.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) }
        let target = normalize(modelId)
        let candidates = measurements.filter { chipMatches($0.value) }

        // An exact id wins. Otherwise fall back to the repo basename ("org/repo" vs "repo",
        // or the same repo under another org), but only when exactly one entry matches:
        // with two, dictionary order would pick one at random — possibly another model's.
        if let exact = candidates.first(where: { normalize($0.key) == target }) {
            return exact.value
        }
        let base = { (id: String) in id.split(separator: "/").last.map(String.init) ?? id }
        let targetBase = base(target)
        guard !targetBase.isEmpty else { return nil }
        let byBase = candidates.filter { base(normalize($0.key)) == targetBase }
        return byBase.count == 1 ? byBase.first?.value : nil
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
        let basaltPath = home.appendingPathComponent(".basalt/hardware_profile.json").path
        if FileManager.default.fileExists(atPath: basaltPath) {
            return basaltPath
        }
        let legacyPath = home.appendingPathComponent(".gem/hardware_profile.json").path
        if FileManager.default.fileExists(atPath: legacyPath) {
            return legacyPath
        }
        return basaltPath
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

    /// The stored profile; an empty one when no file exists yet. Throws when a file exists but
    /// cannot be read — the caller must not mistake that for "no measurements".
    public static func loadChecked(path: String? = nil) throws -> HardwareBenchmarkProfile {
        let url = URL(fileURLWithPath: path ?? defaultProfilePath)
        guard FileManager.default.fileExists(atPath: url.path) else { return HardwareBenchmarkProfile() }
        return try decoder.decode(HardwareBenchmarkProfile.self, from: Data(contentsOf: url))
    }

    /// Read profile synchronously without awaiting actor (useful during fast scoring).
    /// Scoring degrades to the analytical model when the file is unreadable.
    public static func loadSync(path: String? = nil) -> HardwareBenchmarkProfile {
        (try? loadChecked(path: path)) ?? HardwareBenchmarkProfile()
    }

    public func load() -> HardwareBenchmarkProfile {
        if let cached = cachedProfile {
            return cached
        }
        // Cache only what was actually read: caching the empty fallback of an unreadable file
        // would let the next record() write over it.
        guard let profile = try? Self.loadChecked(path: profilePath) else { return HardwareBenchmarkProfile() }
        self.cachedProfile = profile
        return profile
    }

    public func record(_ measurement: BenchmarkMeasurement) throws {
        guard measurement.measuredTPS >= Self.minimumMeasurableTPS else {
            throw HardwareBenchmarkProfileError.implausibleMeasurement(tps: measurement.measuredTPS)
        }
        // Not load(): a profile that exists but does not decode must stop the write, otherwise
        // recording one measurement would silently replace every earlier one.
        var profile = try cachedProfile ?? Self.loadChecked(path: profilePath)
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
