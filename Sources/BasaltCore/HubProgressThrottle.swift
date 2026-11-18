import Foundation

public struct HubTransferTelemetry: Codable, Sendable, Equatable {
    public var done: Int64
    public var total: Int64
    public var bytesPerSecond: Int64
    public var etaSeconds: Int?

    public init(done: Int64 = 0, total: Int64 = 0, bytesPerSecond: Int64 = 0, etaSeconds: Int? = nil) {
        self.done = done
        self.total = total
        self.bytesPerSecond = bytesPerSecond
        self.etaSeconds = etaSeconds
    }
}

public struct HubPublishRequest: Codable, Sendable, Equatable {
    public var sourceId: String
    public var repoId: String
    public var isPrivate: Bool
    public var commitMessage: String

    public init(
        sourceId: String,
        repoId: String,
        isPrivate: Bool = false,
        commitMessage: String = "Upload MLX checkpoint via Basalt"
    ) {
        self.sourceId = sourceId
        self.repoId = repoId
        self.isPrivate = isPrivate
        self.commitMessage = commitMessage
    }
}

public struct HubPublishResult: Codable, Sendable, Equatable {
    public var repoId: String
    public var url: String
    public var uploadedFiles: Int
    public var totalBytes: Int64

    public init(repoId: String, url: String, uploadedFiles: Int, totalBytes: Int64) {
        self.repoId = repoId
        self.url = url
        self.uploadedFiles = uploadedFiles
        self.totalBytes = totalBytes
    }
}

package struct HubProgressThrottle: Sendable {
    package let minIntervalNanoseconds: UInt64
    package let ewmaAlpha: Double
    private(set) var files: [String: (done: Int64, total: Int64)] = [:]
    private(set) var lastEmitNs: UInt64?
    private(set) var lastSampleNs: UInt64?
    private(set) var lastSampleDone: Int64 = 0
    private(set) var smoothedBytesPerSec: Double = 0
    private(set) var hasEmittedStart = false

    package init(minIntervalNanoseconds: UInt64 = 100_000_000, ewmaAlpha: Double = 0.30) {
        self.minIntervalNanoseconds = minIntervalNanoseconds
        self.ewmaAlpha = ewmaAlpha
    }

    package mutating func seedTotals(_ knownTotals: [String: Int64]) {
        for (name, size) in knownTotals where size > 0 {
            let existingDone = files[name]?.done ?? 0
            files[name] = (existingDone, max(size, existingDone))
        }
    }

    package mutating func record(
        file: String,
        done: Int64,
        total: Int64,
        at nowNs: UInt64
    ) -> HubTransferTelemetry? {
        let prevTotal = files[file]?.total ?? 0
        let mergedTotal = max(prevTotal, max(total, done))
        files[file] = (max(0, done), mergedTotal)

        let agg = aggregate()
        updateRate(done: agg.done, at: nowNs)

        let isComplete = agg.total > 0 && agg.done >= agg.total
        let shouldEmit: Bool
        if !hasEmittedStart {
            shouldEmit = true
        } else if isComplete {
            shouldEmit = true
        } else if let last = lastEmitNs {
            shouldEmit = nowNs >= last && (nowNs - last) >= minIntervalNanoseconds
        } else {
            shouldEmit = true
        }

        guard shouldEmit else { return nil }
        hasEmittedStart = true
        lastEmitNs = nowNs
        return snapshot(done: agg.done, total: agg.total)
    }

    package func currentSnapshot() -> HubTransferTelemetry {
        let agg = aggregate()
        return snapshot(done: agg.done, total: agg.total)
    }

    private func aggregate() -> (done: Int64, total: Int64) {
        files.values.reduce((Int64(0), Int64(0))) { partial, entry in
            (partial.0 + entry.done, partial.1 + entry.total)
        }
    }

    private mutating func updateRate(done: Int64, at nowNs: UInt64) {
        guard let prevNs = lastSampleNs else {
            lastSampleNs = nowNs
            lastSampleDone = done
            return
        }
        guard nowNs > prevNs else { return }
        let dtSec = Double(nowNs - prevNs) / 1_000_000_000.0
        guard dtSec >= 0.020 else { return }
        let deltaBytes = max(0, done - lastSampleDone)
        let instRate = Double(deltaBytes) / dtSec
        if smoothedBytesPerSec <= 0 {
            smoothedBytesPerSec = instRate
        } else {
            smoothedBytesPerSec = ewmaAlpha * instRate + (1.0 - ewmaAlpha) * smoothedBytesPerSec
        }
        lastSampleNs = nowNs
        lastSampleDone = done
    }

    private func snapshot(done: Int64, total: Int64) -> HubTransferTelemetry {
        let bps = Int64(max(0, smoothedBytesPerSec.rounded()))
        let remaining = max(0, total - done)
        let eta: Int?
        if remaining == 0 {
            eta = 0
        } else if bps > 0 {
            eta = Int((Double(remaining) / Double(bps)).rounded(.up))
        } else {
            eta = nil
        }
        return HubTransferTelemetry(done: done, total: total, bytesPerSecond: bps, etaSeconds: eta)
    }
}
