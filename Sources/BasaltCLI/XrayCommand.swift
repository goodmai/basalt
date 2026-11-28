import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

struct XrayInspectSnapshot: Sendable, Codable {
    var anatomy: ModelAnatomy
    var tensors: [TensorInfo]
}

struct XrayCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "xray",
        abstract: "Inspect model anatomy, activation depth, outliers, and causal attribution",
        subcommands: [
            Inspect.self,
            Depth.self,
            Outliers.self,
            Causal.self,
        ],
        defaultSubcommand: Inspect.self
    )

    static func resolveModelPath(_ model: String) throws -> String {
        if FileManager.default.fileExists(atPath: model) { return model }
        if let cached = ModelCache.resolve(repoId: model)?.path { return cached }
        throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
    }

    static func withLoadedEngine<T: Sendable>(
        model: String,
        body: (MLXInferenceEngine) async throws -> T
    ) async throws -> T {
        let path = try resolveModelPath(model)
        let engine = MLXInferenceEngine(.standard)
        defer {
            await withTaskCancellationShield {
                await engine.unload()
            }
        }
        try await engine.load(modelPath: path)
        return try await body(engine)
    }

    struct Inspect: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "inspect", abstract: "Print architecture, quantization, MoE/hybrid geometry, and tensor summary")

        @Argument(help: "Local model directory or cached Hugging Face repo ID")
        var model: String

        @Flag(name: .long, help: "Emit JSON instead of a human-readable summary")
        var json: Bool = false

        mutating func run() async throws {
            let output: String = try await XrayCommand.withLoadedEngine(model: model) { engine in
                let anatomy = try await engine.anatomy()
                let tensors = try await engine.tensors(matching: nil)
                if json {
                    return try XrayFormatter.json(XrayInspectSnapshot(anatomy: anatomy, tensors: tensors))
                }
                return XrayFormatter.formatInspect(anatomy, tensors: tensors)
            }
            print(output)
        }
    }

    struct Depth: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "depth", abstract: "Run activation depth diagnostics (S_max, softcap saturation, residual growth, effective rank)")

        @Argument(help: "Local model directory or cached Hugging Face repo ID")
        var model: String

        @Option(name: .long, help: "Prompt to analyze")
        var prompt: String

        @Option(name: .customLong("max-seq-len"), help: "Maximum sequence length cap (default: 512)")
        var maxSeqLen: Int = 512

        @Option(name: .long, help: "Comma-separated layer indices to inspect")
        var layers: String?

        @Flag(name: .long, help: "Emit JSON instead of a human-readable table")
        var json: Bool = false

        mutating func run() async throws {
            let request = DepthDiagnosticsRequest(
                prompt: prompt,
                maxSequenceLength: maxSeqLen,
                layers: try XrayFormatter.parseLayers(layers)
            )
            try request.validate()
            let output: String = try await XrayCommand.withLoadedEngine(model: model) { engine in
                let report = try await engine.runDepthDiagnostics(request)
                return json ? try XrayFormatter.json(report) : XrayFormatter.formatDepth(report)
            }
            print(output)
        }
    }

    struct Outliers: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "outliers", abstract: "Run activation outlier kurtosis, crest factor, dead-neuron, and attention sink profile")

        @Argument(help: "Local model directory or cached Hugging Face repo ID")
        var model: String

        @Option(name: .long, help: "Prompt to analyze")
        var prompt: String

        @Option(name: .customLong("sink-count"), help: "Number of prefix sink tokens (default: 4)")
        var sinkCount: Int = 4

        @Option(name: .long, help: "Comma-separated layer indices to inspect")
        var layers: String?

        @Flag(name: .long, help: "Emit JSON instead of a human-readable table")
        var json: Bool = false

        mutating func run() async throws {
            let request = OutlierProfileRequest(
                prompts: [prompt],
                sinkTokenCount: sinkCount,
                layers: try XrayFormatter.parseLayers(layers)
            )
            try request.validate()
            let output: String = try await XrayCommand.withLoadedEngine(model: model) { engine in
                let report = try await engine.runOutlierProfile(request)
                return json ? try XrayFormatter.json(report) : XrayFormatter.formatOutliers(report)
            }
            print(output)
        }
    }

    struct Causal: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "causal", abstract: "Run Direct Logit Attribution and causal residual patching sweep")

        @Argument(help: "Local model directory or cached Hugging Face repo ID")
        var model: String

        @Option(name: .long, help: "Clean prompt")
        var clean: String

        @Option(name: .long, help: "Corrupted prompt")
        var corrupted: String

        @Option(name: .long, help: "Target token text")
        var target: String?

        @Option(name: .long, help: "Foil token text")
        var foil: String?

        @Option(name: .customLong("target-id"), help: "Target token ID")
        var targetID: Int?

        @Option(name: .customLong("foil-id"), help: "Foil token ID")
        var foilID: Int?

        @Option(name: .long, help: "Comma-separated layer indices to inspect")
        var layers: String?

        @Flag(name: .customLong("no-patching"), help: "Skip causal residual patching and compute DLA only")
        var noPatching: Bool = false

        @Flag(name: .long, help: "Emit JSON instead of a human-readable table")
        var json: Bool = false

        mutating func run() async throws {
            var request = CausalSweepRequest(
                cleanPrompt: clean,
                corruptedPrompt: corrupted,
                targetToken: target,
                foilToken: foil,
                layers: try XrayFormatter.parseLayers(layers),
                includePatching: !noPatching
            )
            request.targetTokenID = targetID
            request.foilTokenID = foilID
            try request.validate()
            let output: String = try await XrayCommand.withLoadedEngine(model: model) { engine in
                let report = try await engine.runCausalSweep(request)
                return json ? try XrayFormatter.json(report) : XrayFormatter.formatCausal(report)
            }
            print(output)
        }
    }
}
