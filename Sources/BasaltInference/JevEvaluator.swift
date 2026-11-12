import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import BasaltCore

public struct JevEvaluator: DecisionEngine {
    public let modelID: String
    let bundle: JevBundle
    let container: ModelContainer
    let head: [Float]
    let header: String
    let tail: String

    public static func isBundle(_ directory: URL) -> Bool { JevBundle.isBundle(directory) }

    public static func baseModel(of directory: URL) throws(BasaltError) -> String { try JevBundle(directory: directory).baseModel }

    public init(directory: URL, base: URL, float32: Bool = false) async throws(BasaltError) {
        bundle = try JevBundle(directory: directory)
        modelID = bundle.modelID
        do {
            container = try await loadModelContainer(from: base, using: DefaultTokenizerLoader())
            let adapterFile = directory.appending(component: "adapter_model.safetensors")
            guard let weight = try loadArrays(url: directory.appending(component: "head.safetensors"))["weight"] else {
                throw BasaltError.modelConfigurationInvalid(path: directory.path, reason: "head.safetensors has no weight")
            }
            head = weight.asType(.float32).reshaped(-1).asArray(Float.self)
            let scale = bundle.loraScale
            let prompt = try await container.perform { context in
                guard Self.supports(context.model) else {
                    throw BasaltError.modelConfigurationInvalid(path: base.path, reason: "the Jev base must be a Qwen3 or Qwen3.5 text checkpoint")
                }
                if FileManager.default.fileExists(atPath: adapterFile.path) {
                    try Self.apply(adapter: directory, file: adapterFile, to: context.model, scale: scale, float32: float32)
                }
                eval(context.model)
                return try Self.template(context.tokenizer)
            }
            (header, tail) = prompt
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelConfigurationInvalid(path: directory.path, reason: error.localizedDescription)
        }
    }

    static func supports(_ model: any LanguageModel) -> Bool { model is Qwen3Model || model is Qwen35TextModel }

    static func apply(adapter directory: URL, file: URL, to model: any LanguageModel, scale: Float, float32: Bool) throws {
        guard float32 else { return try LoRAContainer.fromPEFT(directory: directory).load(into: model) }
        model.update(parameters: model.parameters().mapValues { $0.dtype == .bfloat16 || $0.dtype == .float16 ? $0.asType(.float32) : $0 })
        let merged = try JevBundle.mergedLoRA(try loadArrays(url: file), base: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), scale: scale)
        try model.update(parameters: ModuleParameters.unflattened(merged), verify: [])
    }

    static func hidden(_ model: any LanguageModel, _ inputs: MLXArray, cache: [any KVCache]) throws -> MLXArray {
        let probe = FinalResidual()
        _ = withForwardProbe(probe) { model(LMInput.Text(tokens: inputs), cache: cache, state: nil) }
        guard let residual = probe.hidden, let norm = model.namedModules().first(where: { $0.0 == "model.norm" })?.1 as? any UnaryLayer else {
            throw BasaltError.inferenceError("systemone needs a Qwen3 or Qwen3.5 base")
        }
        return norm(residual)
    }

    static func template(_ tokenizer: any MLXLMCommon.Tokenizer) throws -> (String, String) {
        let ids = try tokenizer.applyChatTemplate(
            messages: [["role": "system", "content": JevBundle.system], ["role": "user", "content": JevBundle.marker]],
            tools: nil, additionalContext: ["enable_thinking": false])
        let parts = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false).components(separatedBy: JevBundle.marker)
        guard parts.count == 2 else { throw BasaltError.modelConfigurationInvalid(path: "chat template", reason: "the template does not place the user message exactly once") }
        return (parts[0], parts[1])
    }

    public func decide(_ request: SystemOneRequest) async throws(BasaltError) -> SystemOneResponse {
        try SystemOneCore.validate(request)
        let names = request.questions.keys.sorted()
        let questions = names.map { request.questions[$0]! }
        let (bundle, head, header, tail) = (bundle, head, header, tail)
        let scored: (scores: [[Double]], usage: SystemOneUsage)
        do {
            scored = try await container.perform { context in
                let prefix = context.tokenizer.encode(text: header + "State:\n" + request.state, addSpecialTokens: false)
                guard prefix.count <= bundle.maxPrefixTokens else { throw BasaltError.contextLengthExceeded(maxTokens: bundle.maxPrefixTokens, requested: prefix.count) }
                let suffixes = questions.map { question in
                    SystemOneCore.options(question).descriptions.map { context.tokenizer.encode(text: SystemOneCore.branch(question, candidate: $0, tail: tail), addSpecialTokens: false) }
                }
                let flat = suffixes.flatMap { $0 }
                guard flat.count <= bundle.maxCandidates else { throw BasaltError.invalidRequestStructure(details: "systemone: \(flat.count) candidates exceed the model's \(bundle.maxCandidates)") }
                if let long = flat.first(where: { $0.count > bundle.maxBranchTokens }) {
                    throw BasaltError.contextLengthExceeded(maxTokens: bundle.maxBranchTokens, requested: long.count)
                }
                let values = try Self.scores(context.model, head: MLXArray(head), prefix: prefix, suffixes: flat, batch: bundle.branchBatchSize)
                var offset = 0
                let grouped = suffixes.map { group in
                    defer { offset += group.count }
                    return Array(values[offset ..< offset + group.count])
                }
                let usage = SystemOneUsage(
                    prefixTokens: prefix.count, candidateBranches: flat.count, prefixEvaluations: 1,
                    branchForwardCalls: (flat.count + bundle.branchBatchSize - 1) / bundle.branchBatchSize, suffixTokens: flat.reduce(0) { $0 + $1.count })
                return (grouped, usage)
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("systemone: \(error.localizedDescription)")
        }
        let calibrated = request.calibrated ?? true
        var answers: [String: SystemOneAnswer] = [:]
        for (index, name) in names.enumerated() {
            let question = questions[index]
            let temperature = calibrated ? bundle.temperatures[question.type] ?? 1 : 1
            answers[name] = SystemOneCore.gated(
                SystemOneCore.answer(question, scores: scored.scores[index], temperature: temperature), abstainBelow: request.abstainBelow, minMargin: request.minMargin)
        }
        return SystemOneResponse(model: modelID, answers: answers, usage: scored.usage, calibrated: calibrated && !bundle.temperatures.isEmpty)
    }

    static func scores(_ model: any LanguageModel, head: MLXArray, prefix: [Int], suffixes: [[Int]], batch: Int) throws -> [Double] {
        let cache = try model.newCache(parameters: nil)
        eval(try hidden(model, MLXArray(prefix.map(Int32.init)).reshaped(1, -1), cache: cache))
        let prefixState = cache.map(\.state)
        var values: [Double] = []
        for start in stride(from: 0, to: suffixes.count, by: batch) {
            let items = Array(suffixes[start ..< min(start + batch, suffixes.count)])
            let longest = items.map(\.count).max() ?? 1
            let ids = items.flatMap { $0 + Array(repeating: 0, count: longest - $0.count) }.map(Int32.init)
            let branchCache = try model.newCache(parameters: nil)
            for (fresh, state) in zip(branchCache, prefixState) { var fresh = fresh; fresh.state = state.map { repeated($0, count: items.count, axis: 0) } }
            let hidden = try hidden(model, MLXArray(ids).reshaped(items.count, longest), cache: branchCache)
            let last = MLXArray(items.enumerated().map { Int32($0.offset * longest + $0.element.count - 1) })
            let rows = hidden.reshaped(items.count * longest, -1)[last].asType(.float32)
            values += matmul(rows, head).asArray(Float.self).map(Double.init)
        }
        return values
    }
}

final class FinalResidual: ForwardProbe {
    private(set) var hidden: MLXArray?

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        self.hidden = hidden
        return hidden
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func routerLogits(layer: Int, _ logits: MLXArray) {}
    func expertMask(layer: Int) -> MLXArray? { nil }
}
