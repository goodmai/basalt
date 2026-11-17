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

    @Option(name: .long, help: "Requantize linear weights: affine:<bits>:<group>, mixed:<low>:<high>:<group> with --budget-gb, nf4:<group>, or gsq-rco with --bits <bpw>")
    var quantize: String?

    @Option(name: .long, help: "Target average bits-per-weight (1.5625 to 8.5) for --quantize gsq-rco")
    var bits: Double?

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
        help: "Experimental (#410): after writing quantized MLX weights (affine 2/3/4-bit or nf4), run PV-Tuning discrete code flips and continuous scale/norm refinement against top-64 teacher log-probabilities with monotone held-out KL rollback"
    )
    var experimentalPvTune = false

    @Option(name: .customLong("pv-teacher"), help: "With --experimental-pv-tune: teacher MLX directory or cached repo ID (default: the conversion source)")
    var pvTeacher: String?

    @Option(name: .customLong("pv-calibration"), help: "With --experimental-pv-tune, --experimental-qtip, --experimental-qtip-budget, or --quantize gsq-rco: UTF-8 text file for calibration (default: built-in prose/code/JSON corpus)")
    var pvCalibration: String?

    @Option(
        name: .customLong("experimental-qtip"),
        help: "Experimental (#409): quantize dense linear weights into QTIP bitshift trellis blocks (2.0, 2.5 or 3.0 bpw; L=16, V=2, HYB Q=9) with Kronecker RHT and BlockLDLQ Hessian feedback"
    )
    var experimentalQtip: Double?

    @Option(
        name: .customLong("experimental-qtip-budget"),
        help: "Experimental (#409 Phase 2): mixed-rate QTIP + 4-bit allocation across 2.0/2.5/3.0/4.0 bpw under an average bits-per-weight budget (2.0 to 4.0), keeping ssm, attn.v and router projections protected"
    )
    var experimentalQtipBudget: Double?

    @Flag(
        name: .customLong("experimental-macaq-refine"),
        help: "Experimental (#535): refine native MLX quantized codes (affine 2/3/4-bit, nf4 or mxfp4) in place via MACaQ-R streaming shared-input Hessians and active-set greedy coordinate-descent flips with zero runtime overhead"
    )
    var experimentalMacaqRefine = false

    @Option(
        name: .customLong("experimental-macaq"),
        help: "Experimental (#535): quantize linear weights with MACaQ channel balance, block-Hadamard rotation and Hessian flip-refinement (affine:<2|3|4>:<32|64|128> or mxfp4)"
    )
    var experimentalMacaq: String?

    @Option(name: .customLong("macaq-block"), help: "With --experimental-macaq: Hadamard rotation block size (32 or 64; default 64)")
    var macaqBlock: Int?

    @Option(name: .customLong("macaq-alpha"), help: "With --experimental-macaq: channel-balance exponent alpha in [0, 1] (default 0.75)")
    var macaqAlpha: Float?

    func validate() throws {
        let gsqBPW = try gsqRCOBits()
        try validatePVOptions(isGSQRCO: gsqBPW != nil)
        try validateUniformQTIP()
        try validateBudgetQTIP(isGSQRCO: gsqBPW != nil)
        try validateGSQRCOOptions(gsqBPW)
        try validateMACaQOptions(isGSQRCO: gsqBPW != nil)
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
        if let spec = experimentalMacaq { return try await convertMACaQ(spec: spec) }
        if let bpw = try gsqRCOBits() { return try await convertGSQRCO(bpw: bpw) }
        if let bpw = experimentalQtipBudget { return try await convertQTIPBudget(bpw: bpw) }
        if let bpw = experimentalQtip { return try await convertQTIP(bpw: bpw) }
        try await runStandardConversion()
    }
}

extension ConvertSubcommand {
    var isGGUFSource: Bool { source.lowercased().hasSuffix(".gguf") }

    func gsqRCOBits() throws -> Double? {
        if quantize == "gsq-rco" {
            guard let bits, bits.isFinite, (1.5625 ... 8.5).contains(bits) else {
                throw ValidationError("--quantize gsq-rco requires --bits between 1.5625 and 8.5")
            }
            return bits
        }
        if let q = quantize, q.hasPrefix("gsq-rco:") {
            guard bits == nil, let val = Double(q.dropFirst("gsq-rco:".count)),
                  val.isFinite, (1.5625 ... 8.5).contains(val) else {
                throw ValidationError("--quantize gsq-rco:<bpw> takes 1.5625 to 8.5 without --bits")
            }
            return val
        }
        if bits != nil {
            throw ValidationError("--bits requires --quantize gsq-rco")
        }
        return nil
    }

    func validatePVOptions(isGSQRCO: Bool) throws {
        if pvTeacher != nil, !experimentalPvTune && !experimentalMacaqRefine {
            throw ValidationError("--pv-teacher requires --experimental-pv-tune or --experimental-macaq-refine")
        }
        let usesCalibration = experimentalPvTune || experimentalMacaqRefine || experimentalMacaq != nil
            || experimentalQtip != nil || experimentalQtipBudget != nil || isGSQRCO
        if pvCalibration != nil, !usesCalibration {
            throw ValidationError("--pv-calibration requires --experimental-pv-tune, --experimental-macaq-refine, --experimental-macaq, --experimental-qtip, --experimental-qtip-budget or --quantize gsq-rco")
        }
        if experimentalPvTune, format != .mlx || dequantize != nil || isGGUFSource {
            throw ValidationError("--experimental-pv-tune applies to quantized MLX conversions (--format mlx without --dequantize)")
        }
    }

    var hasDirectSourceConflict: Bool {
        [quantize != nil, dequantize != nil, adapter != nil, format != .mlx, isGGUFSource]
            .contains(true)
    }

    func validateUniformQTIP() throws {
        guard let bpw = experimentalQtip else { return }
        guard QTIPRate(bpw: bpw) != nil else {
            throw ValidationError("--experimental-qtip takes 2.0, 2.5 or 3.0, got \(bpw)")
        }
        if hasDirectSourceConflict || experimentalQtipBudget != nil {
            throw ValidationError("--experimental-qtip converts an MLX source directly without --quantize, --dequantize, --adapter, --experimental-qtip-budget or --format hf")
        }
    }

    func validateBudgetQTIP(isGSQRCO: Bool) throws {
        guard let budget = experimentalQtipBudget else { return }
        guard budget.isFinite, (2.0 ... 4.0).contains(budget) else {
            throw ValidationError("--experimental-qtip-budget takes 2.0 to 4.0 bpw, got \(budget)")
        }
        if hasDirectSourceConflict || budgetGb != nil || isGSQRCO {
            throw ValidationError("--experimental-qtip-budget converts an MLX source directly without --quantize, --dequantize, --budget-gb, --adapter or --format hf")
        }
    }

    func validateGSQRCOOptions(_ targetBPW: Double?) throws {
        guard targetBPW != nil else { return }
        let hasConflict = dequantize != nil || budgetGb != nil || adapter != nil || experimentalPvTune
        if hasConflict || format != .mlx || isGGUFSource {
            throw ValidationError("--quantize gsq-rco converts an MLX source directly without --dequantize, --budget-gb, --adapter, --experimental-pv-tune or --format hf")
        }
    }

    func readCalibrationText() throws -> String? {
        try pvCalibration.map { path throws -> String in
            let expanded = (path as NSString).expandingTildeInPath
            guard let text = try? String(contentsOfFile: expanded, encoding: .utf8) else {
                throw ValidationError("cannot read --pv-calibration file at \(expanded)")
            }
            return text
        }
    }
}

extension ConvertSubcommand {
    func runStandardConversion() async throws {
        let precision: WeightConverter.Precision
        do {
            precision = try WeightConverter.precision(quantize: quantize, dequantize: dequantize, format: format, budgetGB: budgetGb)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let sourceURL = try Self.resolveModelDirectory(source)
        let teacherURL = try pvTeacher.map(Self.resolveModelDirectory) ?? sourceURL
        let calibrationText = try readCalibrationText()
        let converter = WeightConverter(
            precision: precision, format: format,
            adapter: adapter.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) })
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)

        let start = ContinuousClock.now
        let summary: WeightConverter.Summary
        let pvSummary: PVTuneSummary?
        let macaqSummary: MACaQSummary?
        do {
            summary = try converter.run(source: sourceURL, output: outURL)
            macaqSummary = try await runOptionalMACaQRefine(output: outURL, teacher: teacherURL, calibrationText: calibrationText)
            pvSummary = experimentalPvTune
                ? try await PVTuner.tuneConvertedDirectory(output: outURL, teacher: teacherURL, calibrationText: calibrationText) { print("  [pv-tune] \($0)") }
                : nil
        } catch {
            if experimentalPvTune || experimentalMacaqRefine { try? FileManager.default.removeItem(at: outURL) }
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
        printMACaQRefineSummary(macaqSummary)
        if let pv = pvSummary {
            print(String(
                format: "  PV-Tune:   valKL %.4f → %.4f, %d code(s) updated, %d tensor(s) rewritten",
                pv.valKLBefore, pv.valKLAfter, pv.totalUpdatedCodes, pv.rewrittenTensors))
        }
        print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
    }

    func convertGSQRCO(bpw: Double) async throws {
        let sourceURL = try Self.resolveModelDirectory(source)
        let calibrationText = try readCalibrationText()
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let start = ContinuousClock.now
        do {
            let result = try await GSQRCOConverter.convertDirectory(
                source: sourceURL, output: outURL, targetBPW: bpw, calibrationText: calibrationText
            ) { print("  [gsq-rco] \($0)") }
            let elapsed = ContinuousClock.now - start
            let achieved = String(format: "%.3f", result.summary.plan.achievedBPW)
            printSuccess("Converted \(sourceURL.path) → \(outURL.path) (GSQ-RCO \(achieved) bpw)")
            print("  GSQ-RCO:   \(result.summary.quantizedLayers) layer(s) in \(result.shards) shard(s)")
            print("  Bytes:     \(formatBytes(Int64(result.bytesAfter)))")
            print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }

    func convertQTIPBudget(bpw: Double) async throws {
        let sourceURL = try Self.resolveModelDirectory(source)
        let teacherURL = try pvTeacher.map(Self.resolveModelDirectory) ?? sourceURL
        let calibrationText = try readCalibrationText()
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let start = ContinuousClock.now
        do {
            let result = try await GSQRCOConverter.convertQTIPBudgetDirectory(
                source: sourceURL, output: outURL, targetBPW: bpw, calibrationText: calibrationText
            ) { print("  [qtip-budget] \($0)") }
            let pvSummary = experimentalPvTune
                ? try await PVTuner.tuneConvertedDirectory(output: outURL, teacher: teacherURL, calibrationText: calibrationText) { print("  [pv-tune] \($0)") }
                : nil
            let elapsed = ContinuousClock.now - start
            let achieved = String(format: "%.3f", result.summary.plan.achievedBPW)
            printSuccess("Converted \(sourceURL.path) → \(outURL.path) (QTIP mixed \(achieved) bpw)")
            print("  QTIP:      \(result.summary.qtipLayers) QTIP + \(result.summary.affineProtectedLayers) 4-bit layer(s) in \(result.shards) shard(s)")
            print("  Bytes:     \(formatBytes(Int64(result.bytesAfter)))")
            if let pv = pvSummary {
                print(String(
                    format: "  PV-Tune:   valKL %.4f → %.4f, %d tile/code update(s), %d tensor(s) rewritten",
                    pv.valKLBefore, pv.valKLAfter, pv.totalUpdatedCodes, pv.rewrittenTensors))
            }
            print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            if experimentalPvTune { try? FileManager.default.removeItem(at: outURL) }
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }
}

extension ConvertSubcommand {
    func convertQTIP(bpw: Double) async throws {
        guard let rate = QTIPRate(bpw: bpw) else {
            throw ValidationError("--experimental-qtip takes 2.0, 2.5 or 3.0, got \(bpw)")
        }
        let sourceURL = try Self.resolveModelDirectory(source)
        let teacherURL = try pvTeacher.map(Self.resolveModelDirectory) ?? sourceURL
        let calibrationText = try readCalibrationText()
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let start = ContinuousClock.now
        do {
            let result = try await QTIPNativeLoader.convertDirectory(
                source: sourceURL, output: outURL, rate: rate, calibrationText: calibrationText
            ) { print("  [qtip] \($0)") }
            let pvSummary = experimentalPvTune
                ? try await PVTuner.tuneConvertedDirectory(output: outURL, teacher: teacherURL, calibrationText: calibrationText) { print("  [pv-tune] \($0)") }
                : nil
            let elapsed = ContinuousClock.now - start
            printSuccess("Converted \(sourceURL.path) → \(outURL.path) (QTIP \(rate.label) bpw)")
            print("  QTIP:      \(result.summary.quantizedLayers) layer(s) in \(result.shards) shard(s)")
            print("  Bytes:     \(formatBytes(Int64(result.bytesAfter)))")
            if let pv = pvSummary {
                print(String(
                    format: "  PV-Tune:   valKL %.4f → %.4f, %d tile/code update(s), %d tensor(s) rewritten",
                    pv.valKLBefore, pv.valKLAfter, pv.totalUpdatedCodes, pv.rewrittenTensors))
            }
            print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            if experimentalPvTune { try? FileManager.default.removeItem(at: outURL) }
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }

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
