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
        try Self.validate(request, against: anatomy)
        return try await Self.labRun(container, "the contrast") { context in try ContrastSearch(request, anatomy: anatomy, context: context).run() }
    }

    public func commitSteering(_ commit: SteeringCommit) async throws(BasaltError) -> AdapterVerdict {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        try Self.validateCommit(commit)
        let epoch = modelEpoch
        let preLive = modelFacts.deltas.filter { !$0.value.slice.isEmpty || $0.value.pack != nil }
        let source = try await Self.labRun(container, "the steering commit") { context in try Self.steeringSource(commit, context: context) }
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the steering commit was measured; retry") }
        let verdict = try await applyAdapter(source, gate: commit.gate)
        guard let maxAnti = commit.maxAntiSteeringRate, let overlay = verdict.overlay else { return verdict }
        return try await enforceAntiSteeringGate(
            commit,
            maxRate: maxAnti,
            overlay: overlay,
            verdict: verdict,
            epoch: epoch,
            preLive: preLive
        )
    }

    private func enforceAntiSteeringGate(
        _ commit: SteeringCommit,
        maxRate: Float,
        overlay: OverlayInfo,
        verdict: AdapterVerdict,
        epoch: ModelEpoch,
        preLive: [String: AppliedDelta]
    ) async throws(BasaltError) -> AdapterVerdict {
        guard let container, modelEpoch == epoch else {
            try? await withTaskCancellationShield { () async throws(BasaltError) in try await undo(overlay.id) }
            throw .inferenceError("the model changed while the anti-steering gate ran; retry")
        }
        let check: GateCheck
        do throws(BasaltError) {
            check = try await Self.labRun(container, "the anti-steering gate") { context in
                try Self.antiSteeringCheck(commit, maxRate: maxRate, live: preLive, context: context)
            }
        } catch {
            try? await withTaskCancellationShield { () async throws(BasaltError) in try await undo(overlay.id) }
            throw error
        }
        let checks = (verdict.report?.checks ?? []) + [check]
        let updated = GateReport(checks: checks, compare: verdict.report?.compare, fisherRaoDistance: verdict.report?.fisherRaoDistance)
        guard updated.passed else {
            try await withTaskCancellationShield { () async throws(BasaltError) in try await undo(overlay.id) }
            return AdapterVerdict(overlay: nil, report: updated)
        }
        return AdapterVerdict(overlay: overlay, report: updated)
    }

    static func validate(_ request: ContrastRequest, against anatomy: ModelAnatomy) throws(BasaltError) {
        let layers = anatomy.layers.count
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
        guard request.protectedPrompts.count <= ContrastRequest.classPrompts.upperBound else {
            throw .invalidRequestStructure(details: "protectedPrompts must hold at most \(ContrastRequest.classPrompts.upperBound) prompts")
        }
        if let maxRank = request.nullspaceMaxRank, maxRank <= 0 {
            throw .invalidRequestStructure(details: "nullspaceMaxRank must be positive")
        }
        if let contextGate = request.contextGate {
            try contextGate.validate(hiddenSize: anatomy.hiddenSize)
        }
        if let probes = request.polarityProbes {
            guard (1 ... PolarityProbe.maxProbes).contains(probes.count), probes.allSatisfy({ !$0.prompt.isEmpty }) else {
                throw .invalidRequestStructure(details: "polarityProbes must hold 1...\(PolarityProbe.maxProbes) non-empty prompts")
            }
        }
    }

    static func validateCommit(_ commit: SteeringCommit) throws(BasaltError) {
        guard (1 ... ContrastRequest.classPrompts.upperBound).contains(commit.positive.count) else {
            throw .invalidRequestStructure(details: "steer_commit needs 1...\(ContrastRequest.classPrompts.upperBound) positive prompts to measure the module input")
        }
        guard !commit.direction.isEmpty, commit.scale.isFinite, commit.direction.allSatisfy(\.isFinite) else {
            throw .invalidRequestStructure(details: "the steering direction must be non-empty and, with its scale, finite")
        }
        guard commit.protectedPrompts.count <= ContrastRequest.classPrompts.upperBound,
              commit.negative.count <= ContrastRequest.classPrompts.upperBound else {
            throw .invalidRequestStructure(details: "protectedPrompts and negative must hold at most \(ContrastRequest.classPrompts.upperBound) prompts")
        }
        if let maxRank = commit.nullspaceMaxRank, maxRank <= 0 {
            throw .invalidRequestStructure(details: "nullspaceMaxRank must be positive")
        }
        if let maxAnti = commit.maxAntiSteeringRate, !maxAnti.isFinite || !(0 ... 1).contains(maxAnti) {
            throw .invalidRequestStructure(details: "maxAntiSteeringRate must be in 0...1")
        }
        if let probes = commit.polarityProbes {
            guard (1 ... PolarityProbe.maxProbes).contains(probes.count), probes.allSatisfy({ !$0.prompt.isEmpty }) else {
                throw .invalidRequestStructure(details: "polarityProbes must hold 1...\(PolarityProbe.maxProbes) non-empty prompts")
            }
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
        let outputs = target.linear.weight.dim(0)
        guard commit.direction.count == outputs else {
            throw BasaltError.invalidRequestStructure(details: "the direction has \(commit.direction.count) values; \(found.path).\(target.key) writes \(outputs)")
        }
        let mean = try meanInput(target.key, of: found.layer, linear: target.linear, prompts: prompts, context: context)
        guard !commit.protectedPrompts.isEmpty else {
            return try rankOne(module: found.path + "." + target.key, meanInput: mean, direction: commit.direction, scale: commit.scale, outputs: outputs)
        }
        let protectedIds = try encode(commit.protectedPrompts, tokenizer: context.tokenizer)
        let residualNullspace = try CognitiveNullspaceExtractor.residualNullspace(
            layer: commit.layer, prompts: protectedIds, maxRank: commit.nullspaceMaxRank, context: context)
        let projectedDirection = try residualNullspace.project(commit.direction)
        let outEnergy = projectedDirection.reduce(0.0) { $0 + Double($1) * Double($1) }
        guard outEnergy.isFinite, outEnergy > 1e-16 else {
            throw BasaltError.invalidRequestStructure(details: "the steering direction lies entirely within the protected cognitive subspace")
        }
        let inputSpec = CognitiveNullspaceExtractor.ProtectedInputSpec(
            key: target.key, layer: found.layer, linear: target.linear, context: context)
        let projectedInput = try CognitiveNullspaceExtractor.projectedRankOneInput(
            meanInput: mean,
            protectedPrompts: protectedIds,
            spec: inputSpec
        )
        return .rankOne(module: found.path + "." + target.key, input: projectedInput, output: projectedDirection, scale: commit.scale)
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

    static func antiSteeringCheck(
        _ commit: SteeringCommit,
        maxRate: Float,
        live: [String: AppliedDelta],
        context: ModelContext
    ) throws -> GateCheck {
        let probes = commit.polarityProbes ?? PolarityProbe.expand(commit.gate.heldOut)
        let probeIds = try encode(probes.map(\.prompt), tokenizer: context.tokenizer)
        let posIds = try encode(commit.positive, tokenizer: context.tokenizer)
        let negIds = commit.negative.isEmpty ? [] : try encode(commit.negative, tokenizer: context.tokenizer)
        let pristine = try lifted(live, in: context.model) {
            (
                pos: try posIds.map { try lastTokenLogProbs($0, context: context) },
                neg: try negIds.map { try lastTokenLogProbs($0, context: context) },
                probes: try probeIds.map { try lastTokenLogProbs($0, context: context) }
            )
        }
        let steeredPos = try posIds.map { try lastTokenLogProbs($0, context: context) }
        let steeredProbes = try probeIds.map { try lastTokenLogProbs($0, context: context) }
        let pair = !pristine.neg.isEmpty
            ? ContrastSearch.tokenPair(positive: pristine.pos, negative: pristine.neg)
            : ContrastSearch.shiftTokenPair(baseline: pristine.pos, steered: steeredPos, scale: commit.scale)
        let posLD0 = pristine.pos.map { ContrastSearch.logitDiff($0, pair: pair) }
        let posLDSteered = steeredPos.map { ContrastSearch.logitDiff($0, pair: pair) }
        let refMeanDelta = (posLDSteered.reduce(0, +) - posLD0.reduce(0, +)) / Float(max(posIds.count, 1))
        let refSlope = abs(commit.scale) > 1e-12 ? refMeanDelta / commit.scale : 0
        let baseLD = pristine.probes.map { ContrastSearch.logitDiff($0, pair: pair) }
        let steeredLD = steeredProbes.map { ContrastSearch.logitDiff($0, pair: pair) }
        let summary = AntiSteeringGuard.summarize(
            baseline: baseLD,
            trials: [(scale: commit.scale, steered: steeredLD)],
            kinds: probes.map(\.kind),
            referenceSlope: refSlope
        )
        return AntiSteeringGuard.gateCheck(rate: summary.antiSteeringRate, limit: maxRate)
    }

    private static func lastTokenLogProbs(_ ids: [Int], context: ModelContext) throws -> MLXArray {
        let logits = forward(context, probe: nil, ids, cache: try context.model.newCache(parameters: nil))
        let last = logits[0, ids.count - 1].asType(.float32)
        let logP = last - logSumExp(last, axis: -1)
        eval(logP)
        return logP
    }
}

struct ContrastSearch {
    struct Capture {
        let states: MLXArray
        let logProbabilities: MLXArray
    }

    struct TokenPair {
        let positive: Int
        let negative: Int
    }

    let request: ContrastRequest
    let anatomy: ModelAnatomy
    let context: ModelContext
    let positive: [[Int]]
    let negative: [[Int]]
    let heldOut: [[Int]]
    let protectedIds: [[Int]]

    init(_ request: ContrastRequest, anatomy: ModelAnatomy, context: ModelContext) throws(BasaltError) {
        self.request = request
        self.anatomy = anatomy
        self.context = context
        positive = try MLXInferenceEngine.encode(request.positive, tokenizer: context.tokenizer)
        negative = try MLXInferenceEngine.encode(request.negative, tokenizer: context.tokenizer)
        heldOut = try MLXInferenceEngine.encode(request.heldOut, tokenizer: context.tokenizer)
        protectedIds = request.protectedPrompts.isEmpty ? [] : try MLXInferenceEngine.encode(request.protectedPrompts, tokenizer: context.tokenizer)
    }

    func run() throws -> ContrastReport {
        let posCaptures = try positive.map { try capture($0) }
        let negCaptures = try negative.map { try capture($0) }
        let positives = stacked(posCaptures.map(\.states))
        let negatives = stacked(negCaptures.map(\.states))
        let (unit, norm, separation) = Self.directions(positive: positives, negative: negatives)
        let count = positives.dim(1)
        let layers = request.layers.map { Array(Set($0)).sorted() } ?? Array(0 ..< count)
        if let outside = layers.first(where: { $0 >= count }) {
            throw BasaltError.invalidRequestStructure(details: "layer \(outside) is outside the \(count) layers the forward probe reports")
        }
        let norms = norm.asArray(Float.self), separations = separation.asArray(Float.self)
        let best = layers.max { separations[$0] < separations[$1] } ?? 0
        let rawUnit = unit[best]
        let rawVector = rawUnit.asArray(Float.self)
        let (activeVector, activeTensor, nullspaceRank) = try resolveDirection(best: best, rawVector: rawVector, rawTensor: rawUnit)
        let attn = try attentionDiagnostics(best: best, direction: rawUnit, gap: norms[best])
        let pair = Self.tokenPair(positive: posCaptures.map(\.logProbabilities), negative: negCaptures.map(\.logProbabilities))
        let (trials, steerability) = try trialsAndSteerability(best: best, vector: activeVector, direction: activeTensor, pair: pair)
        var report = ContrastReport(
            layers: layers.map { DirectionLayer(index: $0, norm: norms[$0], separation: separations[$0]) },
            bestLayer: best, direction: activeVector,
            heads: attn.heads,
            trials: trials)
        report.rawDirection = nullspaceRank == nil ? nil : rawVector
        report.nullspaceRank = nullspaceRank
        report.headKurtosis = attn.kurtosis
        report.gqaGroups = attn.groups
        report.recommendedIntervention = attn.mode
        report.steerability = steerability
        return report
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
}

extension ContrastSearch {
    func resolveDirection(
        best: Int,
        rawVector: [Float],
        rawTensor: MLXArray
    ) throws -> (vector: [Float], tensor: MLXArray, rank: Int?) {
        guard !protectedIds.isEmpty else { return (rawVector, rawTensor, nil) }
        let nullspace = try CognitiveNullspaceExtractor.residualNullspace(
            layer: best, prompts: protectedIds, maxRank: request.nullspaceMaxRank, context: context)
        let projected = try nullspace.project(rawVector)
        let tensor = MLXArray(projected)
        eval(tensor)
        return (projected, tensor, nullspace.rank)
    }

    func attentionDiagnostics(
        best: Int,
        direction: MLXArray,
        gap: Float
    ) throws -> (heads: [HeadEffect], kurtosis: Float?, groups: [GQAGroupEffect], mode: HeadInterventionMode?) {
        guard request.topHeads > 0 else { return ([], nil, [], nil) }
        let layers = [best - 1, best].filter {
            anatomy.layers.indices.contains($0) && ["full attention", "sliding window"].contains(anatomy.layers[$0].kind)
        }
        guard !layers.isEmpty else { return ([], nil, [], nil) }
        var allHeads: [HeadEffect] = []
        for layer in layers {
            for head in 0 ..< (anatomy.layers[layer].heads ?? 0) {
                let (positives, negatives) = try classStates([.ablateHead(layer: layer, head: head)])
                let ablated = ((positives[0..., best].mean(axis: 0) - negatives[0..., best].mean(axis: 0)) * direction).sum().item(Float.self)
                allHeads.append(HeadEffect(layer: layer, head: head, effect: gap - ablated))
            }
        }
        let rankedHeads = Array(allHeads.sorted { $0.effect > $1.effect }.prefix(request.topHeads))
        let kurtosis = allHeads.count >= 2 ? ActivationStatistics.excessKurtosis(allHeads.map { abs($0.effect) }) : nil
        let groupInput = GQAGroupPatcher.SearchInput(layers: layers, best: best, direction: direction, gap: gap, search: self)
        let groups = try GQAGroupPatcher.rankGroups(groupInput)
        let refLayer = layers.last ?? best
        let headsCount = anatomy.layers[refLayer].heads ?? 0
        let kvCount = anatomy.layers[refLayer].kvHeads ?? headsCount
        let mode = kurtosis.map { GQAGroupPatcher.recommendMode(heads: headsCount, kvHeads: kvCount, kurtosis: $0) }
        return (rankedHeads, kurtosis, groups, mode)
    }
}

extension ContrastSearch {
    func trialsAndSteerability(
        best: Int,
        vector: [Float],
        direction: MLXArray,
        pair: TokenPair
    ) throws -> ([SteerTrial], SteerabilitySummary) {
        let baseline = try heldOut.map { try capture($0) }
        let baseLD = baseline.map { Self.logitDiff($0.logProbabilities, pair: pair) }
        let last = baseline[0].states.dim(0) - 1
        var steeredLDByScale: [(scale: Float, steered: [Float])] = []
        var rawTrials: [(scale: Float, kl: Float, fr: Float, shift: Float, meanLD: Float)] = []
        for scale in request.scales {
            var divergence: Float = 0, fisherRao: Float = 0, shift: Float = 0
            var lds: [Float] = []
            for (ids, base) in zip(heldOut, baseline) {
                let edit = Intervention.steer(layer: best, vector: vector, scale: scale, geometry: request.geometry, contextGate: request.contextGate)
                let steered = try capture(ids, [edit])
                let drift = LogitDrift.metrics(logP: base.logProbabilities, logQ: steered.logProbabilities)
                divergence += drift.klDivergence
                fisherRao += drift.meanFisherRao
                shift += ((steered.states[last] - base.states[last]) * direction).sum().item(Float.self)
                lds.append(Self.logitDiff(steered.logProbabilities, pair: pair))
            }
            let count = Float(heldOut.count)
            let meanLD = lds.reduce(0, +) / count
            steeredLDByScale.append((scale: scale, steered: lds))
            rawTrials.append((scale, divergence / count, fisherRao / count, shift / count, meanLD))
        }
        let heldOutSummary = AntiSteeringGuard.summarize(baseline: baseLD, trials: steeredLDByScale)
        let finalSummary = try polaritySteerability(best: best, vector: vector, pair: pair, heldOutSummary: heldOutSummary)
        let trials = zip(rawTrials, finalSummary.perScaleAntiSteering).map { raw, anti in
            SteerTrial(
                scale: raw.scale,
                klDivergence: raw.kl,
                projectionShift: raw.shift,
                fisherRaoDistance: raw.fr,
                meanLogitDiff: raw.meanLD,
                antiSteeringRate: anti
            )
        }
        return (trials, finalSummary)
    }

    private func polaritySteerability(
        best: Int,
        vector: [Float],
        pair: TokenPair,
        heldOutSummary: SteerabilitySummary
    ) throws -> SteerabilitySummary {
        guard request.polarityProbes != nil || request.evaluatePolarityFlips else {
            var copy = heldOutSummary
            copy.positiveTokenId = pair.positive
            copy.negativeTokenId = pair.negative
            return copy
        }
        let probes = request.polarityProbes ?? PolarityProbe.expand(request.heldOut)
        let encoded = try MLXInferenceEngine.encode(probes.map(\.prompt), tokenizer: context.tokenizer)
        let baseLD = try encoded.map { Self.logitDiff(try capture($0).logProbabilities, pair: pair) }
        var scaleSeries: [(scale: Float, steered: [Float])] = []
        for scale in request.scales {
            let edit = Intervention.steer(layer: best, vector: vector, scale: scale, geometry: request.geometry, contextGate: request.contextGate)
            let lds = try encoded.map { Self.logitDiff(try capture($0, [edit]).logProbabilities, pair: pair) }
            scaleSeries.append((scale: scale, steered: lds))
        }
        var summary = AntiSteeringGuard.summarize(
            baseline: baseLD,
            trials: scaleSeries,
            kinds: probes.map(\.kind),
            referenceSlope: heldOutSummary.slope
        )
        summary.intercept = heldOutSummary.intercept
        summary.rSquared = heldOutSummary.rSquared
        summary.positiveTokenId = pair.positive
        summary.negativeTokenId = pair.negative
        return summary
    }

    static func tokenPair(positive: [MLXArray], negative: [MLXArray]) -> TokenPair {
        let diff = stacked(positive).mean(axis: 0) - stacked(negative).mean(axis: 0)
        return distinctPair(from: diff)
    }

    static func shiftTokenPair(baseline: [MLXArray], steered: [MLXArray], scale: Float) -> TokenPair {
        let sign: Float = scale >= 0 ? 1 : -1
        let diff = (stacked(steered).mean(axis: 0) - stacked(baseline).mean(axis: 0)) * sign
        return distinctPair(from: diff)
    }

    private static func distinctPair(from diff: MLXArray) -> TokenPair {
        let maxIdx = argMax(diff, axis: -1)
        let minIdx = argMin(diff, axis: -1)
        eval(maxIdx, minIdx)
        let pos = maxIdx.item(Int.self)
        var neg = minIdx.item(Int.self)
        if neg == pos { neg = (pos + 1) % max(diff.dim(0), 2) }
        return TokenPair(positive: pos, negative: neg)
    }

    static func logitDiff(_ logProbs: MLXArray, pair: TokenPair) -> Float {
        let delta = logProbs[pair.positive] - logProbs[pair.negative]
        eval(delta)
        return delta.item(Float.self)
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
