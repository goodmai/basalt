import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension WeightConverter.Format: ExpressibleByArgument {}

struct ConvertSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Write a converted copy of an MLX model: requantize, dequantize, merge a LoRA adapter or export HF safetensors",
        discussion: """
            Reads the source shard by shard and writes a new directory; the source is never written. \
            Tokenizer files and the chat template are copied, config.json gets the new quantization block \
            and the weights get a model.safetensors.index.json with shards of at most 5 GB. \
            GGUF is not supported.
            """
    )

    @Argument(help: "Local MLX model directory or cached repo ID")
    var source: String

    @Option(name: .long, help: "New output directory; must not exist and must not be inside the source")
    var out: String

    @Option(name: .long, help: "Requantize linear weights: affine:<bits>:<group>, e.g. affine:4:64, mixed:<low>:<high>:<group> with --budget-gb, or nf4:<group> (32, 64 or 128; experimental: every linear, MoE experts included, runs on the native nf4 kernels of basalt's mlx fork)")
    var quantize: String?

    @Option(name: .long, help: "Total weight size for --quantize mixed, in GB (10^9 bytes): layers with the largest Marchenko-Pastur noise variance get the high bits while the total fits")
    var budgetGb: Double?

    @Option(name: .long, help: "Dequantize to dense weights: bf16 or f16")
    var dequantize: String?

    @Option(name: .long, help: "Merge an mlx-lm LoRA adapter directory (adapter_config.json + adapters.safetensors)")
    var adapter: String?

    @Option(name: .long, help: "mlx (default) or hf: dense tensors, no .scales/.biases, config.json without quantization")
    var format: WeightConverter.Format = .mlx

    mutating func run() async throws {
        let precision: WeightConverter.Precision
        do {
            precision = try WeightConverter.precision(quantize: quantize, dequantize: dequantize, format: format, budgetGB: budgetGb)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let expanded = (source as NSString).expandingTildeInPath
        guard let sourceURL = FileManager.default.fileExists(atPath: expanded)
            ? URL(fileURLWithPath: expanded) : ModelCache.resolve(repoId: source)
        else { throw ValidationError("\(source) is neither a local directory nor a cached repo") }
        let converter = WeightConverter(
            precision: precision, format: format,
            adapter: adapter.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) })
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)

        let start = ContinuousClock.now
        let summary: WeightConverter.Summary
        do {
            summary = try converter.run(source: sourceURL, output: outURL)
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
        let elapsed = ContinuousClock.now - start
        printSuccess("Converted \(sourceURL.path) → \(outURL.path)")
        print("  Tensors:   \(summary.tensors) → \(summary.outputTensors) in \(summary.shards) shard(s)")
        print("  Bytes:     \(formatBytes(Int64(summary.bytesBefore))) → \(formatBytes(Int64(summary.bytesAfter)))")
        if let plan = summary.plan {
            print("  Plan:      \(plan.highLayers.count) of \(plan.layers) layers at the high bits, \(plan.unscored) unscored kept low")
        }
        print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
    }
}
