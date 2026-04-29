import Foundation
import SQLite

/// Stores benchmark results in a local SQLite database for historical tracking
public actor BenchmarkResultStore {
    private let db: Connection

    // Tables
    private static let benchmarks = Table("benchmarks")
    private static let id = Expression<Int64>("id")
    private static let timestamp = Expression<Date>("timestamp")
    private static let modelPath = Expression<String>("model_path")
    private static let modelId = Expression<String?>("model_id")
    private static let benchmarkType = Expression<String>("benchmark_type") // "synthetic", "swe", "context_degradation"
    private static let resultJSON = Expression<String>("result_json")
    private static let gitCommit = Expression<String?>("git_commit")
    private static let systemInfo = Expression<String?>("system_info")

    public init(dbPath: String = ".benchmark_results.db") throws {
        let connection = try Connection(dbPath)

        // Create table
        try connection.run(Self.benchmarks.create(ifNotExists: true) { t in
            t.column(Self.id, primaryKey: .autoincrement)
            t.column(Self.timestamp)
            t.column(Self.modelPath)
            t.column(Self.modelId)
            t.column(Self.benchmarkType)
            t.column(Self.resultJSON)
            t.column(Self.gitCommit)
            t.column(Self.systemInfo)
        })

        // Create indexes for common queries
        try connection.run(Self.benchmarks.createIndex(Self.modelPath, Self.benchmarkType, ifNotExists: true))
        try connection.run(Self.benchmarks.createIndex(Self.timestamp, ifNotExists: true))

        self.db = connection
    }

    // MARK: — Store Results

    public func storeSyntheticBenchmark(
        modelPath: String,
        modelId: String?,
        result: SyntheticBenchmarkResult
    ) throws {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(result)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        try db.run(Self.benchmarks.insert(
            Self.timestamp <- Date(),
            Self.modelPath <- modelPath,
            Self.modelId <- modelId,
            Self.benchmarkType <- "synthetic",
            Self.resultJSON <- jsonString,
            Self.gitCommit <- getCurrentGitCommit(),
            Self.systemInfo <- getSystemInfo()
        ))
    }

    public func storeContextDegradationBenchmark(
        modelPath: String,
        modelId: String?,
        report: ContextDegradationReport
    ) throws {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(report)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        try db.run(Self.benchmarks.insert(
            Self.timestamp <- Date(),
            Self.modelPath <- modelPath,
            Self.modelId <- modelId,
            Self.benchmarkType <- "context_degradation",
            Self.resultJSON <- jsonString,
            Self.gitCommit <- getCurrentGitCommit(),
            Self.systemInfo <- getSystemInfo()
        ))
    }

    public func storeSWEBenchmark(
        modelPath: String,
        modelId: String?,
        result: SWEBenchmarkResult
    ) throws {
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(result)
        let jsonString = String(data: jsonData, encoding: .utf8)!

        try db.run(Self.benchmarks.insert(
            Self.timestamp <- Date(),
            Self.modelPath <- modelPath,
            Self.modelId <- modelId,
            Self.benchmarkType <- "swe",
            Self.resultJSON <- jsonString,
            Self.gitCommit <- getCurrentGitCommit(),
            Self.systemInfo <- getSystemInfo()
        ))
    }

    // MARK: — Query Results

    public func getRecentBenchmarks(
        modelPath: String? = nil,
        benchmarkType: String? = nil,
        limit: Int = 10
    ) throws -> [BenchmarkRecord] {
        var query = Self.benchmarks.order(Self.timestamp.desc).limit(limit)

        if let modelPath = modelPath {
            query = query.filter(Self.modelPath == modelPath)
        }

        if let benchmarkType = benchmarkType {
            query = query.filter(Self.benchmarkType == benchmarkType)
        }

        var records: [BenchmarkRecord] = []
        for row in try db.prepare(query) {
            records.append(BenchmarkRecord(
                id: try row.get(Self.id),
                timestamp: try row.get(Self.timestamp),
                modelPath: try row.get(Self.modelPath),
                modelId: try row.get(Self.modelId),
                benchmarkType: try row.get(Self.benchmarkType),
                resultJSON: try row.get(Self.resultJSON),
                gitCommit: try row.get(Self.gitCommit),
                systemInfo: try row.get(Self.systemInfo)
            ))
        }

        return records
    }

    public func getBenchmarkHistory(
        modelPath: String,
        benchmarkType: String,
        since: Date? = nil
    ) throws -> [BenchmarkRecord] {
        var query = Self.benchmarks
            .filter(Self.modelPath == modelPath && Self.benchmarkType == benchmarkType)
            .order(Self.timestamp.asc)

        if let since = since {
            query = query.filter(Self.timestamp >= since)
        }

        var records: [BenchmarkRecord] = []
        for row in try db.prepare(query) {
            records.append(BenchmarkRecord(
                id: try row.get(Self.id),
                timestamp: try row.get(Self.timestamp),
                modelPath: try row.get(Self.modelPath),
                modelId: try row.get(Self.modelId),
                benchmarkType: try row.get(Self.benchmarkType),
                resultJSON: try row.get(Self.resultJSON),
                gitCommit: try row.get(Self.gitCommit),
                systemInfo: try row.get(Self.systemInfo)
            ))
        }

        return records
    }

    public func detectRegression(
        modelPath: String,
        benchmarkType: String,
        currentResult: Double,
        metric: String,
        threshold: Double = 0.10 // 10% degradation
    ) throws -> RegressionDetection? {
        let history = try getBenchmarkHistory(modelPath: modelPath, benchmarkType: benchmarkType)

        guard history.count >= 2 else {
            return nil // Not enough data
        }

        // Get baseline (average of last 5 runs, excluding current)
        let recentRuns = history.suffix(6).dropLast()
        guard !recentRuns.isEmpty else { return nil }

        // This is simplified - in real implementation, parse JSON and extract specific metric
        // For now, just demonstrate the concept

        let baseline = currentResult * 1.1 // Placeholder
        let degradation = (baseline - currentResult) / baseline

        if degradation > threshold {
            return RegressionDetection(
                metric: metric,
                baseline: baseline,
                current: currentResult,
                degradationPercent: degradation * 100,
                threshold: threshold * 100
            )
        }

        return nil
    }

    // MARK: — Helpers

    private func getCurrentGitCommit() -> String? {
        let task = Process()
        task.launchPath = "/usr/bin/env"
        task.arguments = ["git", "rev-parse", "--short", "HEAD"]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()

        do {
            try task.run()
            task.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    private func getSystemInfo() -> String? {
        let task = Process()
        task.launchPath = "/usr/bin/env"
        task.arguments = ["uname", "-a"]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()

        do {
            try task.run()
            task.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }
}

// MARK: — Data Types

public struct BenchmarkRecord: Codable, Sendable {
    public let id: Int64
    public let timestamp: Date
    public let modelPath: String
    public let modelId: String?
    public let benchmarkType: String
    public let resultJSON: String
    public let gitCommit: String?
    public let systemInfo: String?
}

public struct SyntheticBenchmarkResult: Codable {
    public let avgTPS: Double
    public let avgTTFT: Double
    public let minTPS: Double
    public let maxTPS: Double
    public let stdDevTPS: Double
    public let memoryPeakMB: Int
    public let iterations: Int
}

public struct SWEBenchmarkResult: Codable {
    public let tasks: [SWETaskResult]
    public let overallScore: Double
    public let totalTime: Double
}

public struct SWETaskResult: Codable {
    public let taskName: String
    public let correctness: Double
    public let completeness: Double
    public let codeQuality: Double
    public let timeEfficiency: Double
    public let timeTaken: Double
    public let timeBudget: Double
}

public struct RegressionDetection {
    public let metric: String
    public let baseline: Double
    public let current: Double
    public let degradationPercent: Double
    public let threshold: Double
}
