import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    public func contrast(_ request: ContrastRequest) async throws(BasaltError) -> ContrastReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the contrast was prepared; retry") }
        try Self.validate(request, layers: anatomy.layers.count)
        return try await Self.labRun(container, "the contrast") { context in try ContrastSearch(request, anatomy: anatomy, context: context).run() }
    }

    public func commitSteering(_ commit: SteeringCommit) async throws(BasaltError) -> AdapterVerdict {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        guard (1 ... ContrastRequest.classPrompts.upperBound).contains(commit.positive.count) else {
            throw .invalidRequestStructure(details: "steer_commit needs 1...\(ContrastRequest.classPrompts.upperBound) positive prompts to measure the module input")
        }
        guard !commit.direction.isEmpty, commit.scale.isFinite, commit.direction.allSatisfy(\.isFinite) else {
            throw .invalidRequestStructure(details: "the steering direction must be non-empty and, with its scale, finite")
        }
        let epoch = modelEpoch
        let source = try await Self.labRun(container, "the steering commit") { context in try Self.steeringSource(commit, context: context) }
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the steering commit was measured; retry") }
        return try await applyAdapter(source, gate: commit.gate)
    }

    static func validate(_ request: ContrastRequest, layers: Int) throws(BasaltError) {
        guard ContrastRequest.classPrompts.contains(request.positive.count), ContrastRequest.classPrompts.contains(request.negative.count) else {
            throw .invalidRequestStructure(details: "contrast needs \(ContrastRequest.classPrompts) positive and \(ContrastRequest.classPrompts) negative prompts")
        }
        guard ContrastRequest.heldOutPrompts.contains(request.heldOut.count) else {
            throw .invalidRequestStructure(details: "contrast needs \(ContrastRequest.heldOutPrompts) held-out prompts")
        }
        guard request.topHeads >= 0, ContrastRequest.scaleCount.contains(request.scales.count), request.scales.allSatisfy(\.isFinite) else {
            throw .invalidRequestStructure(details: "topHeads must be non-negative and scales must hold \(ContrastRequest.scaleCount) finite values")
        }
        if let chosen = request.layers, chosen.isEmpty || chosen.contains(where: { !(0 ..< layers).contains($0) }) {
            throw .invalidRequestStructure(details: "layers must list at least one layer in 0..<\(layers)")
        }
    }

    static func encode(_ prompts: [String], tokenizer: any Tokenizer) throws(BasaltError) -> [[Int]] {
        var encoded: [[Int]] = []
        for prompt in prompts {
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: true)
            guard !ids.isEmpty else { throw .invalidRequestStructure(details: "a prompt encodes to no tokens") }
            guard ids.count <= traceTokenLimit else { throw .contextLengthExceeded(maxTokens: traceTokenLimit, requested: ids.count) }
            encoded.append(ids)
        }
        return encoded
    }
}

extension MLXInferenceEngine {
    static let steeringModules = ["mlp.down_proj", "feed_forward.w2", "self_attn.o_proj"]

    static func steeringSource(_ commit: SteeringCommit, context: ModelContext) throws -> AdapterSource {
        let prompts = try encode(commit.positive, tokenizer: context.tokenizer)
        guard let found = decoderLayer(commit.layer, in: context.model) else {
            throw BasaltError.invalidRequestStructure(details: "the model has no decoder layer \(commit.layer)")
        }
        guard let target = steeringTarget(in: found.layer) else {
            throw BasaltError.invalidRequestStructure(
                details: "layer \(commit.layer) has no dense projection into the residual (\(steeringModules.joined(separator: ", "))); steer another layer")
        }
        let mean = try meanInput(target.key, of: found.layer, linear: target.linear, prompts: prompts, context: context)
        return try rankOne(
            module: found.path + "." + target.key, meanInput: mean, direction: commit.direction, scale: commit.scale,
            outputs: target.linear.weight.dim(0))
    }

    static func decoderLayer(_ index: Int, in model: Module) -> (path: String, layer: Module)? {
        model.namedModules().first { path, _ in
            (path == "layers.\(index)" || path.hasSuffix(".layers.\(index)")) && !path.contains("vision") && !path.contains("audio")
        }.map { (path: $0.0, layer: $0.1) }
    }

    static func steeringTarget(in layer: Module) -> (key: String, linear: Linear)? {
        let modules = Dictionary(layer.namedModules(), uniquingKeysWith: { first, _ in first })
        return steeringModules.lazy.compactMap { key in (modules[key] as? Linear).map { (key: key, linear: $0) } }.first
    }

    static func rankOne(module: String, meanInput: [Float], direction: [Float], scale: Float, outputs: Int) throws(BasaltError) -> AdapterSource {
        guard direction.count == outputs else {
            throw .invalidRequestStructure(details: "the direction has \(direction.count) values; \(module) writes \(outputs)")
        }
        let energy = meanInput.reduce(0) { $0 + $1 * $1 }
        guard energy.isFinite, energy > 0 else {
            throw .invalidRequestStructure(details: "the mean input of \(module) over the positive prompts is zero or not finite")
        }
        return .rankOne(module: module, input: meanInput.map { $0 / energy }, output: direction, scale: scale)
    }

    static func meanInput(_ key: String, of layer: Module, linear: Linear, prompts: [[Int]], context: ModelContext) throws -> [Float] {
        var rows: [MLXArray] = []
        layer.update(modules: .unflattened([(key, InputTap(linear, record: { rows.append($0) }) as Module)]))
        context.model.invalidateCompiledTraces()
        defer {
            layer.update(modules: .unflattened([(key, linear as Module)]))
            context.model.invalidateCompiledTraces()
        }
        for ids in prompts {
            _ = forward(context, probe: nil, ids, cache: try context.model.newCache(parameters: nil))
            eval(rows)
        }
        guard rows.count == prompts.count else {
            throw BasaltError.inferenceError("\(key) ran \(rows.count) times for \(prompts.count) prompts; its input cannot be measured")
        }
        return stacked(rows).mean(axis: 0).asArray(Float.self)
    }
}

struct ContrastSearch {
    struct Capture {
        let states: MLXArray
        let logProbabilities: MLXArray
    }

    let request: ContrastRequest
    let anatomy: ModelAnatomy
    let context: ModelContext
    let positive: [[Int]]
    let negative: [[Int]]
    let heldOut: [[Int]]

    init(_ request: ContrastRequest, anatomy: ModelAnatomy, context: ModelContext) throws(BasaltError) {
        self.request = request
        self.anatomy = anatomy
        self.context = context
        positive = try MLXInferenceEngine.encode(request.positive, tokenizer: context.tokenizer)
        negative = try MLXInferenceEngine.encode(request.negative, tokenizer: context.tokenizer)
        heldOut = try MLXInferenceEngine.encode(request.heldOut, tokenizer: context.tokenizer)
    }

    func run() throws -> ContrastReport {
        let (positives, negatives) = try classStates()
        let (unit, norm, separation) = Self.directions(positive: positives, negative: negatives)
        let count = positives.dim(1)
        let layers = request.layers.map { Array(Set($0)).sorted() } ?? Array(0 ..< count)
        if let outside = layers.first(where: { $0 >= count }) {
            throw BasaltError.invalidRequestStructure(details: "layer \(outside) is outside the \(count) layers the forward probe reports")
        }
        let norms = norm.asArray(Float.self), separations = separation.asArray(Float.self)
        let best = layers.max { separations[$0] < separations[$1] } ?? 0
        let direction = unit[best]
        return ContrastReport(
            layers: layers.map { DirectionLayer(index: $0, norm: norms[$0], separation: separations[$0]) },
            bestLayer: best, direction: direction.asArray(Float.self),
            heads: try heads(best: best, direction: direction, gap: norms[best]),
            trials: try trials(best: best, direction: direction))
    }

    static func directions(positive: MLXArray, negative: MLXArray) -> (unit: MLXArray, norm: MLXArray, separation: MLXArray) {
        let difference = positive.mean(axis: 0) - negative.mean(axis: 0)
        let norm = sqrt(square(difference).sum(axis: -1))
        let unit = difference / maximum(norm, MLXArray(Float(1e-12)))[0..., .newAxis]
        func spread(_ states: MLXArray) -> MLXArray {
            let projections = (states * unit).sum(axis: -1)
            return square(projections - projections.mean(axis: 0)).sum(axis: 0)
        }
        let pooled = sqrt((spread(positive) + spread(negative)) / Float(positive.dim(0) + negative.dim(0) - 2))
        let separation = norm / maximum(pooled, MLXArray(Float(1e-6)))
        eval(unit, norm, separation)
        return (unit, norm, separation)
    }

    func classStates(_ edits: [Intervention] = []) throws -> (positive: MLXArray, negative: MLXArray) {
        (stacked(try positive.map { try capture($0, edits).states }), stacked(try negative.map { try capture($0, edits).states }))
    }

    func capture(_ ids: [Int], _ edits: [Intervention] = []) throws -> Capture {
        let probe = ContrastProbe(edits.isEmpty ? nil : InterventionProbe(edits, anatomy: anatomy))
        let logits = MLXInferenceEngine.forward(context, probe: probe, ids, cache: try context.model.newCache(parameters: nil))
        let last = logits[0, ids.count - 1].asType(.float32)
        let states = (0 ..< probe.states.count).compactMap { probe.states[$0] }
        guard !states.isEmpty, states.count == probe.states.count else {
            throw BasaltError.invalidRequestStructure(details: "this model's family does not report its layers to the forward probe")
        }
        let capture = Capture(states: stacked(states), logProbabilities: last - logSumExp(last, axis: -1))
        eval(capture.states, capture.logProbabilities)
        return capture
    }

    func heads(best: Int, direction: MLXArray, gap: Float) throws -> [HeadEffect] {
        guard request.topHeads > 0 else { return [] }
        let layers = [best - 1, best].filter { anatomy.layers.indices.contains($0) && ["full attention", "sliding window"].contains(anatomy.layers[$0].kind) }
        var effects: [HeadEffect] = []
        for layer in layers {
            for head in 0 ..< (anatomy.layers[layer].heads ?? 0) {
                let (positives, negatives) = try classStates([.ablateHead(layer: layer, head: head)])
                let ablated = ((positives[0..., best].mean(axis: 0) - negatives[0..., best].mean(axis: 0)) * direction).sum().item(Float.self)
                effects.append(HeadEffect(layer: layer, head: head, effect: gap - ablated))
            }
        }
        return Array(effects.sorted { $0.effect > $1.effect }.prefix(request.topHeads))
    }

    func trials(best: Int, direction: MLXArray) throws -> [SteerTrial] {
        let vector = direction.asArray(Float.self)
        let baseline = try heldOut.map { try capture($0) }
        let last = baseline[0].states.dim(0) - 1
        return try request.scales.map { scale in
            var divergence: Float = 0, shift: Float = 0
            for (ids, base) in zip(heldOut, baseline) {
                let steered = try capture(ids, [.steer(layer: best, vector: vector, scale: scale)])
                let p = exp(base.logProbabilities)
                divergence += which(p .> 0, p * (base.logProbabilities - steered.logProbabilities), 0).sum().item(Float.self)
                shift += ((steered.states[last] - base.states[last]) * direction).sum().item(Float.self)
            }
            let count = Float(heldOut.count)
            return SteerTrial(scale: scale, klDivergence: divergence / count, projectionShift: shift / count)
        }
    }
}

final class ContrastProbe: ForwardProbe {
    private let edits: InterventionProbe?
    private(set) var states: [Int: MLXArray] = [:]

    init(_ edits: InterventionProbe?) {
        self.edits = edits
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        let edited = edits?.residual(layer: layer, hidden) ?? hidden
        states[layer] = edited[0, edited.dim(1) - 1].asType(.float32)
        return edited
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}

    func wantsAttention(layer: Int) -> Bool { false }

    func headMask(layer: Int) -> MLXArray? { edits?.headMask(layer: layer) }

    func routerLogits(layer: Int, _ logits: MLXArray) {}

    func expertMask(layer: Int) -> MLXArray? { edits?.expertMask(layer: layer) }
}

final class InputTap: Linear {
    private let original: (MLXArray) -> MLXArray
    private let record: (MLXArray) -> Void

    init(_ linear: Linear, record: @escaping (MLXArray) -> Void) {
        original = { linear($0) }
        self.record = record
        super.init(weight: linear.weight, bias: linear.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        record(x[0, x.dim(1) - 1].asType(.float32))
        return original(x)
    }
}
