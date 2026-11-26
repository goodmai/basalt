import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension EMeansPrecision: ExpressibleByArgument {}

public enum EMeansFinalizerOption: String, ExpressibleByArgument, Sendable, CaseIterable {
    case none
    case lastEpochMean = "last-epoch-mean"
    case slowEMA = "slow-ema"
}

public struct EMeansCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "emeans",
        abstract: "Cluster dense vectors with Meta FAIR e-means scaling-law EMA clustering"
    )

    @Option(name: .long, help: "Input .safetensors or .jsonl file of vectors")
    public var input: String

    @Option(name: [.short, .long], help: "Number of clusters K")
    public var clusters: Int

    @Option(name: .long, help: "Training epochs (default 2.5)")
    public var epochs: Double = 2.5

    @Option(name: [.short, .long], help: "Output .safetensors path for centroids and counts")
    public var output: String

    @Option(name: .long, help: "Tensor key inside .safetensors input (default: first 2-D float tensor)")
    public var key: String?

    @Option(name: .long, help: "Override mini-batch size (default: scaling-law knee)")
    public var batchSize: Int?

    @Option(name: .long, help: "Override peak EMA step size in (0, 1]")
    public var peakStep: Float?

    @Option(name: .long, help: "Finalizer: last-epoch-mean, slow-ema, or none")
    public var finalizer: EMeansFinalizerOption = .lastEpochMean

    @Option(name: .long, help: "Slow-EMA speed multiplier when --finalizer slow-ema")
    public var slowSpeed: Float = 0.25

    @Option(name: .long, help: "Assignment GEMM dtype: float32, bfloat16, or float16")
    public var dtype: EMeansPrecision = .float32

    @Option(name: .long, help: "Deterministic SplitMix64 seed")
    public var seed: UInt64 = 0

    public init() {}

    public mutating func validate() throws {
        guard clusters > 0 else {
            throw ValidationError("--clusters must be greater than 0")
        }
        guard epochs.isFinite && epochs > 0 else {
            throw ValidationError("--epochs must be finite and greater than 0")
        }
        guard output.lowercased().hasSuffix(".safetensors") else {
            throw ValidationError("--output must end with .safetensors")
        }
        try validateOptionalOverrides()
    }

    public func makeConfig() -> EMeansConfig {
        let resolvedFinalizer: EMeansFinalizer = switch finalizer {
        case .none: .none
        case .lastEpochMean: .lastEpochMean
        case .slowEMA: .slowEMA(speedMultiplier: slowSpeed)
        }
        var config = EMeansConfig(
            clusters: clusters,
            epochs: epochs,
            batchSize: batchSize,
            finalizer: resolvedFinalizer,
            seed: seed,
            assignmentDType: dtype
        )
        config.peakStep = peakStep
        return config
    }

    public mutating func run() async throws {
        let report = try EMeans.fitFile(
            inputURL: URL(fileURLWithPath: input),
            outputURL: URL(fileURLWithPath: output),
            tensorKey: key,
            config: makeConfig()
        )
        print("centroids: [\(report.clusters), \(report.dimensions)] -> \(report.outputPath)")
        print("mse: \(String(format: "%.6f", Double(report.metrics.trainMSE)))")
        print("dead_clusters: \(report.metrics.deadClusters)")
        print("revivals: \(report.metrics.revivals)")
        print("batch_size: \(report.metrics.batchSize)")
        print("peak_step: \(String(format: "%.6f", Double(report.metrics.peakStep)))")
        print("steps: \(report.metrics.steps)")
        print("throughput: \(String(format: "%.1f", report.vectorsPerSecond)) vectors/s")
    }

    private func validateOptionalOverrides() throws {
        if let batchSize, batchSize <= 0 {
            throw ValidationError("--batch-size must be greater than 0")
        }
        if let peakStep, !(peakStep.isFinite && peakStep > 0 && peakStep <= 1) {
            throw ValidationError("--peak-step must be in (0, 1]")
        }
        guard slowSpeed.isFinite && slowSpeed > 0 else {
            throw ValidationError("--slow-speed must be finite and greater than 0")
        }
    }
}
