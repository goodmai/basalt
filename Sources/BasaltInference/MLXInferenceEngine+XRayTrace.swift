import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let traceTokenLimit = 2048
    static let traceTopKLimit = 50

    public func trace(_ request: TraceRequest) async throws(BasaltError) -> TraceReport {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        guard (1 ... Self.traceTopKLimit).contains(request.topK) else {
            throw .invalidRequestStructure(details: "topK must be between 1 and \(Self.traceTopKLimit), got \(request.topK)")
        }
        if let layers = request.layers, layers.isEmpty || layers.contains(where: { $0 < 0 }) {
            throw .invalidRequestStructure(details: "layers must list at least one layer index, none negative")
        }
        do {
            return try await container.performEvaluating(nonSendable: ()) { context, _ in try Self.capture(request, context: context) }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .modelInferenceError(details: Self.briefly(error))
        }
    }

    static func capture(_ request: TraceRequest, context: ModelContext) throws -> TraceReport {
        let ids = context.tokenizer.encode(text: request.prompt, addSpecialTokens: true)
        guard !ids.isEmpty else { throw BasaltError.invalidRequestStructure(details: "the prompt encodes to no tokens") }
        guard ids.count <= traceTokenLimit else {
            throw BasaltError.contextLengthExceeded(maxTokens: traceTokenLimit, requested: ids.count)
        }
        let position = request.position ?? ids.count - 1
        guard ids.indices.contains(position) else {
            throw BasaltError.invalidRequestStructure(details: "position \(position) is outside the prompt's \(ids.count) tokens")
        }
        let recorder = TraceRecorder(request, position: position, tokens: ids.count)
        var context = context
        context.probe = recorder
        defer { context.probe = nil }
        let input = LMInput.Text(tokens: MLXArray(ids.map(Int32.init)).reshaped(1, -1))
        let logits = context.model(input, cache: try context.model.newCache(parameters: nil), state: nil).logits
        return try recorder.report(logits: logits, readout: readout(of: context.model), ids: ids, tokenizer: context.tokenizer)
    }

    static func readout(of model: Module) -> ((MLXArray) -> MLXArray)? {
        let modules = model.namedModules().map { (path: $0.0, name: $0.0.split(separator: ".").last.map(String.init), module: $0.1) }
            .sorted { $0.path.split(separator: ".").count < $1.path.split(separator: ".").count }
        guard let embedding = ["embed_tokens", "embeddings"].lazy.compactMap({ name in
            modules.first { $0.name == name && $0.module is Embedding }
        }).first, let table = embedding.module as? Embedding else { return nil }
        let parent = embedding.path.split(separator: ".").dropLast().map { $0 + "." }.joined()
        guard let norm = ["norm", "norm_f", "embedding_norm"].lazy.compactMap({ name in
            modules.first { $0.path == parent + name }?.module as? any UnaryLayer
        }).first else { return nil }
        let head = modules.first { $0.name == "lm_head" }?.module as? Linear
        return { hidden in
            let normed = norm(hidden)
            return head.map { $0(normed) } ?? table.asLinear(normed)
        }
    }
}

final class TraceRecorder: ForwardProbe {
    let request: TraceRequest
    let position: Int
    let tokens: Int
    private(set) var layerCount = 0
    private var residuals: [Int: MLXArray] = [:]
    private var attention: [Int: MLXArray] = [:]
    private var routers: [Int: MLXArray] = [:]

    init(_ request: TraceRequest, position: Int, tokens: Int) {
        self.request = request
        self.position = position
        self.tokens = tokens
    }

    func traces(_ layer: Int) -> Bool { request.layers?.contains(layer) ?? true }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        layerCount = max(layerCount, layer + 1)
        if traces(layer) { residuals[layer] = hidden[0, position] }
        return hidden
    }

    func wantsAttention(layer: Int) -> Bool { request.attention && traces(layer) }

    func attentionWeights(layer: Int, _ weights: MLXArray) {
        let query = position - (tokens - weights.dim(2))
        guard wantsAttention(layer: layer), query >= 0 else { return }
        attention[layer] = weights[0, 0..., query].asType(.float32)
    }

    func headMask(layer: Int) -> MLXArray? { nil }

    func routerLogits(layer: Int, _ logits: MLXArray) {
        guard request.router, traces(layer) else { return }
        routers[layer] = softmax(logits.reshaped(-1, logits.dim(-1))[position].asType(.float32), axis: -1)
    }

    func expertMask(layer: Int) -> MLXArray? { nil }

    func report(logits: MLXArray, readout: ((MLXArray) -> MLXArray)?, ids: [Int], tokenizer: any Tokenizer) throws -> TraceReport {
        guard layerCount > 0 else {
            throw BasaltError.invalidRequestStructure(details: "this model's family does not report its layers to the forward probe")
        }
        let layers = request.layers.map { Array(Set($0)).sorted() } ?? Array(0 ..< layerCount)
        if let missing = layers.first(where: { residuals[$0] == nil }) {
            throw BasaltError.invalidRequestStructure(details: "layer \(missing) is outside the model's \(layerCount) layers")
        }
        let states = stacked(layers.compactMap { residuals[$0] })
        let norms = sqrt(square(states.asType(.float32)).sum(axis: -1))
        let lens = request.logitLens ? readout.map { softmax($0(states).asType(.float32), axis: -1) } : nil
        let output = softmax(logits[0, position].asType(.float32), axis: -1)
        eval([norms, output] + (lens.map { [$0] } ?? []) + Array(attention.values) + Array(routers.values))
        let k = request.topK
        func scores(_ row: MLXArray) -> [TokenScore] {
            TraceSummary.top(row.asArray(Float.self), k: k).map {
                TokenScore(token: tokenizer.decode(tokenIds: [$0.index], skipSpecialTokens: false), id: $0.index, probability: $0.value)
            }
        }
        let hiddenNorms = norms.asArray(Float.self)
        let layerTraces = layers.enumerated().map { row, layer in
            LayerTrace(
                index: layer, hiddenNorm: hiddenNorms[row],
                lens: lens.map { scores($0[row]) },
                attention: attention[layer].map {
                    TraceSummary.heads($0.asArray(Float.self), heads: $0.dim(0), k: k, firstPosition: tokens - $0.dim(1))
                },
                experts: routers[layer].map {
                    TraceSummary.top($0.asArray(Float.self), k: k).map { ExpertChoice(expert: $0.index, weight: $0.value) }
                })
        }
        return TraceReport(
            tokens: ids.map { tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false) },
            position: position, layers: layerTraces, final: scores(output))
    }
}
