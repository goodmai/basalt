import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

enum NF4Run {
    static let env = ProcessInfo.processInfo.environment
    static let formats: [PackFormat] = [.nf4(groupSize: 32), .nf4(groupSize: 64), .affine(bits: 4, groupSize: 32), .affine(bits: 4, groupSize: 64), .mxfp4]
    static let sampledExperts = 4

    static func models(_ variable: String) -> [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env[variable] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    static func path(_ entry: String) -> String { entry.hasPrefix("/") ? entry : ModelCache.cacheDir(for: entry).path }

    static func layerIndex(_ path: String) -> Int? {
        let parts = path.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else { return nil }
        return Int(parts[at + 1])
    }

    static func sample(_ container: ModelContainer) async -> (linears: [String], experts: [String]) {
        await container.perform { context in
            let modules = context.model.namedModules()
            let last = modules.compactMap { layerIndex($0.0) }.max() ?? 0
            let chosen: Set<Int> = [0, last / 2, last]
            let inChosen = { (path: String) in layerIndex(path).map(chosen.contains) ?? false }
            let plain = [ObjectIdentifier(Linear.self), ObjectIdentifier(QuantizedLinear.self)]
            let linears = modules.filter { plain.contains(ObjectIdentifier(type(of: $0.1))) && (inChosen($0.0) || $0.0.hasSuffix("lm_head")) }
            let experts = modules.filter { $0.1 is SwitchLinear && inChosen($0.0) }
            return (linears.map { $0.0 + ".weight" }, experts.map(\.0))
        }
    }

    static func expertErrors(_ container: ModelContainer, path: String) async -> [String] {
        await container.perform { context in
            guard let module = context.model.namedModules().first(where: { $0.0 == path })?.1 else { return ["\(path) not found"] }
            let parameters = Dictionary(module.parameters().flattened()) { first, _ in first }
            guard let weight = parameters["weight"] else { return ["\(path) has no weight"] }
            let count = min(sampledExperts, weight.dim(0))
            var dense = weight
            var source = "\(weight.dtype)"
            if let quantized = module as? QuantizedSwitchLinear, let scales = parameters["scales"] {
                dense = dequantized(
                    weight[0 ..< count], scales: scales[0 ..< count], biases: parameters["biases"].map { $0[0 ..< count] },
                    groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
                source = "dequantized \(quantized.bits)-bit \(quantized.mode.rawValue) g\(quantized.groupSize)"
            }
            return errors("\(path) from \(source)", dense)
        }
    }

    static func errors(_ key: String, _ tensor: MLXArray) -> [String] {
        let dense = tensor.ndim == 3 ? tensor[0 ..< min(sampledExperts, tensor.dim(0))] : tensor
        let flat = dense.reshaped(-1, dense.dim(-1))
        return formats.map { format in
            do {
                let error = TensorPacker.measure(flat, try TensorPacker.converted(flat, bias: nil, to: format))
                return "\(key) \(tensor.shape) from \(tensor.dtype) → \(TensorPacker.describe(format)) relative \(error.relativeError) rmse \(error.rmse) maxAbs \(error.maxAbsError)"
            } catch {
                return "\(key) → \(TensorPacker.describe(format)) refused: \(error)"
            }
        }
    }

    static func denseErrors(_ directory: URL) throws -> [String] {
        let shards = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".safetensors") }.sorted()
            .map { directory.appendingPathComponent($0) }
        let keys = try shards.map { try loadArrays(url: $0).keys }
        let last = keys.joined().compactMap(layerIndex).max() ?? 0
        let chosen: Set<Int> = [0, last / 2, last]
        var lines: [String] = []
        for shard in shards {
            let arrays = try loadArrays(url: shard)
            for key in arrays.keys.sorted() where layerIndex(key).map(chosen.contains) ?? key.hasSuffix("lm_head.weight") {
                guard let tensor = arrays[key], WeightConverter.isLinearLike(key, tensor) else { continue }
                lines += errors(key, tensor)
                Memory.clearCache()
            }
        }
        return lines
    }

    static func perplexity(_ container: ModelContainer, text: String) async throws -> Double {
        try await container.perform { context in
            let tokens = Array(context.tokenizer.encode(text: text).prefix(2049))
            let cache = try context.model.newCache(parameters: nil)
            var nll = 0.0
            var start = 0
            while start < tokens.count - 1 {
                let end = min(tokens.count - 1, start + 512)
                let logits = context.model(MLXArray(tokens[start ..< end].map(Int32.init))[.newAxis], cache: cache)[0].asType(.float32)
                let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
                let targets = MLXArray(tokens[(start + 1) ... end].map(Int32.init))
                nll -= Double(takeAlong(logp, targets[0..., .newAxis], axis: -1).sum().item(Float.self))
                eval(cache)
                start = end
            }
            return exp(nll / Double(max(tokens.count - 1, 1)))
        }
    }

    static func decodeSpeed(_ engine: MLXInferenceEngine) async throws -> Double {
        var speeds: [Double] = []
        for run in 0 ..< 3 {
            let prompt = "Write a long, detailed essay on the history of the printing press, part \(run + 1)."
            speeds.append(try await engine.generate(request: GenerationRequest(prompt: prompt, maxTokens: 128, temperature: 0)).tokensPerSecond)
        }
        return speeds.sorted()[1]
    }

    struct Arm {
        var speed: Double
        var perplexity: Double
        var traces: [String: GoldenTrace]
    }

    static let text = (try? String(
        contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("CONTRIBUTING.md"), encoding: .utf8)) ?? ""

    static func measure(_ engine: MLXInferenceEngine, path: String, arm: String, log: ModelRunLog) async throws -> Arm {
        try await log.step("load \(arm)") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let speed = try await log.step("decode \(arm)") { try await decodeSpeed(engine) }
        let perplexity = try await log.step("perplexity \(arm)") { try await NF4Run.perplexity(container, text: text) }
        let traces = try await GoldenRun.traces(engine, run: 0, log: log)
        log.record("arm", "\(arm) decode \(speed) tok/s, perplexity \(perplexity), MLXLM_NF4_KERNEL=\(env["MLXLM_NF4_KERNEL"] ?? "unset")")
        return Arm(speed: speed, perplexity: perplexity, traces: traces)
    }

    static func sourceBits(_ path: String) -> Int? {
        let config = (try? Data(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        return (config?["quantization"] as? [String: Any])?["bits"] as? Int
    }
}

@Suite("NF4 against affine and mxfp4 on real tensors: tensor_pack error per layer, with the source precision", .serialized)
struct NF4PackRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_NF4_PACK_MODELS")

    @Test("Attention, dense FFN and lm_head through tensor_pack (undone after each), MoE experts through the same codec; every format's relative error is logged",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_NF4_PACK_MODELS"), arguments: models)
    func packSweep(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("nf4-pack", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        await engine.skipJournalForDeltaTests()
        let container = try #require(await engine.container)
        let (linears, experts) = await NF4Run.sample(container)
        log.record("sample", "\(linears.count) linear weights, \(experts.count) expert stacks")
        for tensor in linears {
            for format in NF4Run.formats {
                let report = try await log.step("pack \(tensor) \(TensorPacker.describe(format))") {
                    try await engine.pack(PackRequest(tensor: tensor, format: format))
                }
                log.record("result", "\(report.overlay.description) relative \(report.relativeError) rmse \(report.rmse) maxAbs \(report.maxAbsError) bytes \(report.bytesBefore.rawValue) → \(report.bytesAfter.rawValue)")
                try await engine.undo(report.overlay.id)
            }
        }
        for stack in experts {
            for line in await NF4Run.expertErrors(container, path: stack) {
                log.record("result", line)
            }
        }
        #expect(!linears.isEmpty)
    }
}

@Suite("NF4 against affine and mxfp4 on bf16 tensors read one at a time from a source snapshot, without loading the model", .serialized)
struct NF4DenseTensorRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_NF4_BF16_MODELS")

    @Test("Every linear weight of the first, middle and last layer and lm_head, and four experts of each expert stack, re-packed from bf16 with every format's error logged",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_NF4_BF16_MODELS to bf16 snapshots"), arguments: models)
    func denseSweep(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("nf4-bf16", modelPath: path)
        defer { log.releaseGPU() }
        let lines = try await log.step("sweep") { try NF4Run.denseErrors(URL(fileURLWithPath: path)) }
        for line in lines { log.record("result", line) }
        #expect(!lines.isEmpty && lines.allSatisfy { $0.contains("from bfloat16") || $0.contains("from float16") || $0.contains("refused") })
    }
}

@Suite("NF4 g32 checkpoint against its affine 4-bit source: decode speed, perplexity and golden-logit KL", .serialized)
struct NF4SpeedRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_NF4_SPEED_MODELS")
    static let output = URL(fileURLWithPath: NF4Run.env["BASALT_E2E_NF4_OUT"] ?? "/private/tmp/basalt-nf4")

    @Test("Converted with models convert --quantize nf4:32, decode reaches 0.95x of the source and perplexity is no worse",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_NF4_SPEED_MODELS"), arguments: models)
    func speedAndQuality(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("nf4-speed", modelPath: path)
        defer { log.releaseGPU() }
        if let bits = NF4Run.sourceBits(path), bits != 4 {
            log.record("SKIP", "the source is \(bits)-bit; nf4 is compared with an affine 4-bit or dense source, pass a bf16 checkpoint path instead")
            return
        }
        let converted = Self.output.appendingPathComponent(log.model.replacingOccurrences(of: "/", with: "--") + "-nf4g32")
        if !FileManager.default.fileExists(atPath: converted.path) {
            let summary = try await log.step("convert") {
                try WeightConverter(precision: .nf4(groupSize: 32)).run(source: URL(fileURLWithPath: path), output: converted)
            }
            log.record("convert", "\(converted.path): \(summary.outputTensors) tensors, \(summary.bytesBefore) → \(summary.bytesAfter) bytes")
        }
        let inspection = await ModelInspector().inspect(modelIdOrPath: converted.path)
        log.record("inspect nf4", "intact \(inspection.intact), \(inspection.totalSizeFormatted)")
        try #require(inspection.intact)
        let engine = MLXInferenceEngine(promptCacheBytes: 1 << 30)
        let source = try await NF4Run.measure(engine, path: path, arm: "source", log: log)
        let nf4 = try await NF4Run.measure(engine, path: converted.path, arm: "nf4", log: log)
        let kl = GoldenFixture(revision: "source", tolerance: .infinity, prompts: source.traces).compare(nf4.traces)
        let ratio = nf4.speed / source.speed
        log.record("RESULT", "decode ratio \(ratio), perplexity \(source.perplexity) → \(nf4.perplexity), golden \(kl)")
        #expect(ratio >= 0.95, "decode \(nf4.speed) vs \(source.speed) tok/s")
        #expect(nf4.perplexity <= source.perplexity, "perplexity \(nf4.perplexity) vs \(source.perplexity)")
    }
}

final class CapturingLinear: Linear {
    let inner: Linear
    var inputs: [MLXArray] = []

    init(_ inner: Linear) {
        self.inner = inner
        super.init(weight: inner.weight, bias: inner.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        inputs.append(x)
        return inner(x)
    }
}

enum NF4Activation {
    static let calibrationTokens = 512

    static func tail(_ key: String) -> String? {
        if key.hasSuffix("lm_head.weight") { return "lm_head.weight" }
        guard let range = key.range(of: #"layers\.\d+\..+"#, options: .regularExpression) else { return nil }
        return String(key[range])
    }

    static func sourceIndex(_ directory: URL) throws -> [String: (URL, String)] {
        let shards = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".safetensors") }
        var index: [String: (URL, String)] = [:]
        for name in shards {
            let url = directory.appendingPathComponent(name)
            for key in try loadArrays(url: url).keys { if let tail = tail(key) { index[tail] = (url, key) } }
        }
        return index
    }

    static func sweep(_ container: ModelContainer, paths: [String], index: [String: (URL, String)], text: String) async throws -> [String] {
        try await container.perform { context in
            var probes: [(String, Linear, CapturingLinear)] = []
            for path in paths {
                guard let linear = context.model.namedModules().first(where: { $0.0 == path })?.1 as? Linear else { continue }
                let probe = CapturingLinear(linear)
                try MLXInferenceEngine.replace(path, with: probe, in: context.model)
                probes.append((path, linear, probe))
            }
            let tokens = Array(context.tokenizer.encode(text: text).prefix(calibrationTokens)).map(Int32.init)
            eval(context.model(MLXArray(tokens)[.newAxis], cache: try context.model.newCache(parameters: nil)))
            for (path, linear, _) in probes { try MLXInferenceEngine.replace(path, with: linear, in: context.model) }
            var lines: [String] = []
            for (path, _, probe) in probes {
                guard !probe.inputs.isEmpty, let (url, key) = tail(path + ".weight").flatMap({ index[$0] }),
                      let weight = try loadArrays(url: url)[key] else { lines.append("\(path) no input or weight"); continue }
                let input = concatenated(probe.inputs.map { $0.reshaped(-1, $0.dim(-1)) }, axis: 0).asType(.float32)
                lines += errors(path, weight: weight, input: input)
                probe.inputs = []
                Memory.clearCache()
            }
            return lines
        }
    }

    static func errors(_ path: String, weight: MLXArray, input: MLXArray) -> [String] {
        let w = weight.asType(.float32)
        guard w.ndim == 2, w.dim(1) == input.dim(1) else { return ["\(path) shape \(weight.shape) vs input \(input.shape): skipped"] }
        let reference = matmul(input, w.transposed())
        let norm = sqrt((reference * reference).sum()).item(Float.self)
        return NF4Run.formats.map { format in
            do {
                let packed = TensorPacker.dense(try TensorPacker.converted(weight, bias: nil, to: format)).asType(.float32)
                let difference = matmul(input, (w - packed).transposed())
                let error = sqrt((difference * difference).sum()).item(Float.self) / norm
                Memory.clearCache()
                return "act \(path) \(weight.shape) tokens \(input.dim(0)) → \(TensorPacker.describe(format)) relative \(error)"
            } catch {
                return "act \(path) → \(TensorPacker.describe(format)) refused: \(error)"
            }
        }
    }
}

@Suite("NF4 against affine and mxfp4, weighted by real layer inputs: bf16 weights, inputs captured from the shipped checkpoint", .serialized)
struct NF4ActivationRealModelTests {
    static let pairs = NF4Run.models("BASALT_E2E_NF4_ACT_MODELS")

    @Test("‖X(W−Ŵ)ᵀ‖/‖XWᵀ‖ per sampled linear on 512 calibration tokens, W from the bf16 source, X from the quantized checkpoint's forward",
          .enabled(if: !pairs.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_NF4_ACT_MODELS to <checkpoint>=<bf16 directory> pairs"), arguments: pairs)
    func activationSweep(pair: String) async throws {
        let parts = pair.split(separator: "=", maxSplits: 1).map(String.init)
        try #require(parts.count == 2, "expected <checkpoint>=<bf16 directory>")
        let path = NF4Run.path(parts[0])
        let index = try NF4Activation.sourceIndex(URL(fileURLWithPath: parts[1]))
        let log = try await ModelRunLog("nf4-activation", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let paths = await NF4Run.sample(container).linears.map { String($0.dropLast(".weight".count)) }
        let matched = paths.filter { NF4Activation.tail($0 + ".weight").flatMap { index[$0] } != nil }
        log.record("sample", "\(paths.count) linears, \(matched.count) matched in \(index.count) bf16 tensors of \(parts[1])")
        for line in try await NF4Activation.sweep(container, paths: matched, index: index, text: NF4Run.text) { log.record("result", line) }
        #expect(!matched.isEmpty)
    }
}
