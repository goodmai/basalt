import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension WeightConverter.Format: ExpressibleByArgument {}
extension GGUFImporter.Grid: ExpressibleByArgument {}

struct ConvertSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Write a converted copy of an MLX model: requantize, dequantize, merge a LoRA adapter or export HF safetensors",
        discussion: """
            Reads the source shard by shard and writes a new directory; the source is never written. \
            Tokenizer files and the chat template are copied, config.json gets the new quantization block \
            and the weights get a model.safetensors.index.json with shards of at most 5 GB. \
            Experimental: a .gguf source (Qwen3.5/3.8 hybrid models, such as the GSQ-RCO releases) keeps its GGUF \
            blocks with --grid native (default), which basalt runs on its own i-quant and k-quant Metal kernels; \
            --grid affine dequantizes tensor by tensor into MLX affine weights at the bit-width nearest each GGUF type \
            (IQ1/IQ2/Q2_K → 2, IQ3 → 3, IQ4/Q4_K → 4, Q6_K → 6, Q8_0/BF16 → 8). --template names the MLX \
            checkpoint of the same model whose config.json and tokenizer are copied.
            """
    )

    @Argument(help: "Local MLX model directory, cached repo ID, or a .gguf file with --template")
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

    @Option(name: .long, help: "With a .gguf source: the MLX checkpoint (directory or cached repo) of the same model, for config.json and the tokenizer")
    var template: String?

    @Option(name: .long, help: "With a .gguf source: native (default) keeps the GGUF blocks for basalt's i-quant kernels, affine requantizes into MLX affine weights")
    var grid: GGUFImporter.Grid = .native

    @Option(name: .long, help: "With a .gguf source and --grid affine: the smallest MLX bit-width to write (2 to 8; default 2), to raise IQ2-class tensors to 3 bits")
    var minBits = 2

    @Option(name: .long, help: "With a .gguf source and --grid affine: the MLX group size (32, 64 or 128; default 64)")
    var groupSize = 64

    @Flag(
        name: .customLong("experimental-pv-tune"),
        help: "Experimental (#410): after writing quantized MLX weights (affine 2/3/4-bit or nf4), run PV-Tuning discrete code flips and continuous scale/norm refinement against top-64 teacher log-probabilities with monotone held-out KL rollback. Measured on M4 Pro: Gemma 4 26B-A4B 3-bit affine 16k PPL 130.97 → 119.62 (-8.7% in 185 s) and nf4 76.63 → 54.38 (-29.0% in 159 s); Qwen3.8-27B 3-bit affine 4.702 → 4.652 (-1.1%, behind GSQ-RCO IQ3_S at 4.169)"
    )
    var experimentalPvTune = false

    @Option(name: .customLong("pv-teacher"), help: "With --experimental-pv-tune: teacher MLX directory or cached repo ID (default: the conversion source)")
    var pvTeacher: String?

    @Option(name: .customLong("pv-calibration"), help: "With --experimental-pv-tune: UTF-8 text file for teacher log-probability calibration (default: built-in prose/code/JSON corpus)")
    var pvCalibration: String?

    func validate() throws {
        if (pvTeacher != nil || pvCalibration != nil), !experimentalPvTune {
            throw ValidationError("--pv-teacher and --pv-calibration require --experimental-pv-tune")
        }
        if experimentalPvTune, format != .mlx || dequantize != nil || source.lowercased().hasSuffix(".gguf") {
            throw ValidationError("--experimental-pv-tune applies to quantized MLX conversions (--format mlx without --dequantize)")
        }
    }

    static func resolveModelDirectory(_ spec: String) throws -> URL {
        let expanded = (spec as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) { return URL(fileURLWithPath: expanded) }
        if !spec.hasPrefix("/"), !spec.hasPrefix("."), !spec.hasPrefix("~"),
           let cached = ModelCache.resolve(repoId: spec) {
            return cached
        }
        throw ValidationError("\(spec) is neither a local directory nor a cached repo")
    }

    mutating func run() async throws {
        if source.lowercased().hasSuffix(".gguf") { return try importGGUF() }
        let precision: WeightConverter.Precision
        do {
            precision = try WeightConverter.precision(quantize: quantize, dequantize: dequantize, format: format, budgetGB: budgetGb)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let sourceURL = try Self.resolveModelDirectory(source)
        let teacherURL = try pvTeacher.map(Self.resolveModelDirectory) ?? sourceURL
        let calibrationText = try pvCalibration.map { path throws -> String in
            let expanded = (path as NSString).expandingTildeInPath
            guard let text = try? String(contentsOfFile: expanded, encoding: .utf8) else {
                throw ValidationError("cannot read --pv-calibration file at \(expanded)")
            }
            return text
        }
        let converter = WeightConverter(
            precision: precision, format: format,
            adapter: adapter.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) })
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)

        let start = ContinuousClock.now
        let summary: WeightConverter.Summary
        let pvSummary: PVTuneSummary?
        do {
            summary = try converter.run(source: sourceURL, output: outURL)
            pvSummary = experimentalPvTune
                ? try await PVTuner.tuneConvertedDirectory(output: outURL, teacher: teacherURL, calibrationText: calibrationText) { print("  [pv-tune] \($0)") }
                : nil
        } catch {
            if experimentalPvTune { try? FileManager.default.removeItem(at: outURL) }
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
        if let pv = pvSummary {
            print(String(
                format: "  PV-Tune:   valKL %.4f → %.4f, %d code(s) updated, %d tensor(s) rewritten",
                pv.valKLBefore, pv.valKLAfter, pv.totalUpdatedCodes, pv.rewrittenTensors))
        }
        print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
    }
}

extension ConvertSubcommand {
    func importGGUF() throws {
        guard let template, let templateURL = FileManager.default.fileExists(atPath: (template as NSString).expandingTildeInPath)
            ? URL(fileURLWithPath: (template as NSString).expandingTildeInPath) : ModelCache.resolve(repoId: template) else {
            throw ValidationError("a .gguf source needs --template <MLX checkpoint of the same model>")
        }
        guard (2 ... 8).contains(minBits), [32, 64, 128].contains(groupSize) else { throw ValidationError("--min-bits takes 2 to 8 and --group-size 32, 64 or 128") }
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let start = ContinuousClock.now
        do {
            let file = try GGUFFile(url: URL(fileURLWithPath: (source as NSString).expandingTildeInPath))
            let importer = GGUFImporter(file: file, layout: try GGUFImporter.layout(template: templateURL), grid: grid, groupSize: groupSize, minimumBits: minBits)
            let summary = try importer.run(output: outURL, template: templateURL)
            printSuccess("Imported \(file.url.lastPathComponent) → \(outURL.path)")
            print("  Tensors:   \(summary.tensors), \(formatBytes(Int64(summary.bytes)))")
            if !summary.types.isEmpty { print("  GGUF:      " + summary.types.sorted { $0.value > $1.value }.map { "\($0.key) × \($0.value)" }.joined(separator: ", ")) }
            if !summary.bits.isEmpty { print("  Bits:      " + summary.bits.sorted { $0.key < $1.key }.map { "\($0.key)-bit × \($0.value)" }.joined(separator: ", ")) }
            if !summary.skipped.isEmpty { print("  Skipped:   \(summary.skipped.count) tensors without an MLX counterpart (\(summary.skipped.prefix(4).joined(separator: ", ")))") }
            print("  Time:      \((ContinuousClock.now - start).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }
}
