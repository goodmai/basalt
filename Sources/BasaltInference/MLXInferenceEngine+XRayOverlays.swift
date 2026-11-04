import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    static let compareTokenLimit = 256

    public func intervene(_ intervention: Intervention) async throws(BasaltError) -> OverlayInfo {
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the intervention was checked; retry") }
        try Self.validate(intervention, against: anatomy)
        return register(intervention, description: Self.describe(intervention))
    }

    public func overlays() async -> [OverlayInfo] {
        refreshOverlays(at: Date())
        return experiments.registry.overlays.map(\.info)
    }

    public func undo(_ overlay: OverlayID) async throws(BasaltError) {
        let restored = try await undoDelta(overlay)
        guard remove(overlay) != nil || restored else {
            throw .invalidRequestStructure(details: "no overlay \(overlay.rawValue): it expired, was undone or belonged to a previous model")
        }
    }

    public func compare(_ request: CompareRequest) async throws(BasaltError) -> CompareReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        refreshOverlays(at: Date())
        let interventions = try self.interventions(request.overlays)
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed before the comparison started; retry") }
        let prompts = request.prompts
        let maxTokens = min(max(request.maxTokens, 1), Self.compareTokenLimit)
        do {
            let rows = try await container.performEvaluating(nonSendable: ()) { context, _ in
                let probe = interventions.isEmpty ? nil : InterventionProbe(interventions, anatomy: anatomy)
                return try prompts.map { try Self.compareRow($0, context: context, probe: probe, maxTokens: maxTokens) }
            }
            return CompareReport(rows: rows)
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("the comparison failed: \(Self.briefly(error))")
        }
    }

    func register(_ intervention: Intervention?, description: String) -> OverlayInfo {
        let now = Date()
        refreshOverlays(at: now)
        let info = experiments.registry.add(intervention, description: description, at: now)
        journal("add", info, at: now)
        return info
    }

    @discardableResult
    func remove(_ id: OverlayID) -> OverlayRegistry.Overlay? {
        let now = Date()
        refreshOverlays(at: now)
        guard let removed = experiments.registry.remove(id) else { return nil }
        journal("undo", removed.info, at: now)
        return removed
    }

    private func refreshOverlays(at now: Date) {
        if experiments.epoch != modelEpoch {
            experiments.epoch = modelEpoch
            experiments.registry.removeAll()
        }
        for overlay in experiments.registry.expire(at: now) { journal("expire", overlay.info, at: now) }
    }

    private func interventions(_ ids: [OverlayID]) throws(BasaltError) -> [Intervention] {
        var found: [Intervention] = []
        for id in ids {
            guard let overlay = experiments.registry.overlays.first(where: { $0.info.id == id }) else {
                throw .invalidRequestStructure(details: "no overlay \(id.rawValue): it expired, was undone or belonged to a previous model")
            }
            if let intervention = overlay.intervention { found.append(intervention) }
        }
        return found
    }

    private func journal(_ event: String, _ info: OverlayInfo, at now: Date) {
        let fingerprint = modelFingerprint ?? PromptCacheDiskStore.fingerprint(modelDirectory: URL(fileURLWithPath: modelDescription))
        do {
            try experiments.registry.record(event, info, at: now, fingerprint: fingerprint)
        } catch {
            BasaltLogger(module: "MLXInferenceEngine").warn("The experiment journal was not written: \(Self.briefly(error))")
        }
    }

    static func validate(_ intervention: Intervention, against anatomy: ModelAnatomy) throws(BasaltError) {
        func layer(_ index: Int) throws(BasaltError) -> LayerAnatomy {
            guard anatomy.layers.indices.contains(index) else {
                throw .invalidRequestStructure(details: "layer \(index) does not exist: the model has layers 0..<\(anatomy.layers.count)")
            }
            return anatomy.layers[index]
        }
        switch intervention {
        case .steer(let index, let vector, let scale):
            _ = try layer(index)
            guard vector.count == anatomy.hiddenSize else {
                throw .invalidRequestStructure(details: "the steering vector has \(vector.count) values; the hidden size is \(anatomy.hiddenSize)")
            }
            guard scale.isFinite, vector.allSatisfy(\.isFinite) else {
                throw .invalidRequestStructure(details: "the steering vector and its scale must be finite")
            }
        case .ablateHead(let index, let head):
            let heads = try layer(index).heads ?? 0
            guard (0 ..< heads).contains(head) else {
                throw .invalidRequestStructure(details: "layer \(index) has \(heads) attention heads; head \(head) does not exist")
            }
        case .ablateExpert(let index, let expert):
            let experts = try layer(index).experts ?? 0
            guard (0 ..< experts).contains(expert) else {
                throw .invalidRequestStructure(details: "layer \(index) has \(experts) experts; expert \(expert) does not exist")
            }
        }
    }

    static func describe(_ intervention: Intervention) -> String {
        switch intervention {
        case .steer(let layer, let vector, let scale): "steer the residual after layer \(layer) by \(scale) × a \(vector.count)-value vector"
        case .ablateHead(let layer, let head): "ablate head \(head) of layer \(layer)"
        case .ablateExpert(let layer, let expert): "ablate expert \(expert) of layer \(layer)"
        }
    }

}

extension MLXInferenceEngine {
    static func compareRow(_ prompt: String, context: ModelContext, probe: (any ForwardProbe)?, maxTokens: Int) throws -> CompareRow {
        let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !ids.isEmpty else { throw BasaltError.invalidRequestStructure(details: "an empty prompt has no next token to compare") }
        let stops = stopTokenIds(context)
        let baseline = try greedy(ids, context: context, probe: nil, maxTokens: maxTokens, stops: stops)
        let overlay = try greedy(ids, context: context, probe: probe, maxTokens: maxTokens, stops: stops)
        let base = try logProbabilities(ids + baseline, from: ids.count - 1, context: context, probe: nil)
        let steered = try logProbabilities(ids + baseline, from: ids.count - 1, context: context, probe: probe)
        let p = exp(base)
        let kl = which(p .> 0, p * (base - steered), 0).sum(axis: -1).mean()
        let targets = MLXArray(baseline)[0..., .newAxis]
        let basePerplexity = exp(-takeAlong(base, targets, axis: -1).mean())
        let steeredPerplexity = exp(-takeAlong(steered, targets, axis: -1).mean())
        eval(kl, basePerplexity, steeredPerplexity)
        func text(_ tokens: [Int]) -> String {
            context.tokenizer.decode(tokenIds: tokens.filter { !stops.contains($0) }, skipSpecialTokens: true)
        }
        return CompareRow(
            prompt: prompt, baselineText: text(baseline), overlayText: text(overlay), klDivergence: kl.item(Float.self),
            baselinePerplexity: basePerplexity.item(Float.self), overlayPerplexity: steeredPerplexity.item(Float.self))
    }

    static func greedy(_ ids: [Int], context: ModelContext, probe: (any ForwardProbe)?, maxTokens: Int, stops: Set<Int>) throws -> [Int] {
        let cache = try context.model.newCache(parameters: nil)
        var input = ids
        var tokens: [Int] = []
        while tokens.count < maxTokens {
            let next = argMax(forward(context, probe: probe, input, cache: cache)[0, -1], axis: -1).item(Int.self)
            tokens.append(next)
            if stops.contains(next) { break }
            input = [next]
        }
        return tokens
    }

    static func logProbabilities(_ path: [Int], from start: Int, context: ModelContext, probe: (any ForwardProbe)?) throws -> MLXArray {
        let logits = forward(context, probe: probe, path, cache: try context.model.newCache(parameters: nil))[0, start ..< (path.count - 1)]
            .asType(.float32)
        return logits - logSumExp(logits, axis: -1, keepDims: true)
    }

    static func forward(_ context: ModelContext, probe: (any ForwardProbe)?, _ tokens: [Int], cache: [KVCache]) -> MLXArray {
        withForwardProbe(probe) { context.model(MLXArray(tokens)[.newAxis], cache: cache) }
    }
}

final class InterventionProbe: ForwardProbe {
    private let steering: [Int: MLXArray]
    private let heads: [Int: MLXArray]
    private let experts: [Int: MLXArray]

    init(_ interventions: [Intervention], anatomy: ModelAnatomy) {
        var steering: [Int: [Float]] = [:], heads: [Int: [Bool]] = [:], experts: [Int: [Bool]] = [:]
        for intervention in interventions {
            switch intervention {
            case .steer(let layer, let vector, let scale):
                let sum = steering[layer] ?? Array(repeating: 0, count: vector.count)
                steering[layer] = zip(sum, vector).map { $0 + scale * $1 }
            case .ablateHead(let layer, let head):
                heads[layer, default: Array(repeating: true, count: anatomy.layers[layer].heads ?? 0)][head] = false
            case .ablateExpert(let layer, let expert):
                experts[layer, default: Array(repeating: true, count: anatomy.layers[layer].experts ?? 0)][expert] = false
            }
        }
        self.steering = steering.mapValues { MLXArray($0) }
        self.heads = heads.mapValues { MLXArray($0) }
        self.experts = experts.mapValues { MLXArray($0) }
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        steering[layer].map { hidden + $0.asType(hidden.dtype) } ?? hidden
    }

    func attentionWeights(layer: Int, _ weights: MLXArray) {}

    func wantsAttention(layer: Int) -> Bool { false }

    func headMask(layer: Int) -> MLXArray? { heads[layer] }

    func routerLogits(layer: Int, _ logits: MLXArray) {}

    func expertMask(layer: Int) -> MLXArray? { experts[layer] }
}
