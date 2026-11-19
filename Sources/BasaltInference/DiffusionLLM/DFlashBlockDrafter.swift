import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DFlashConfig: Codable, Equatable, Sendable {
    public var hiddenSize: Int
    public var targetHiddenSize: Int
    public var vocabularySize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var blockSize: Int
    public var maskTokenId: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float

    public init(
        hiddenSize: Int = 64,
        vocabularySize: Int = 256,
        numHiddenLayers: Int = 2,
        numAttentionHeads: Int = 4,
        blockSize: Int = 8,
        maskTokenId: Int = 0
    ) {
        let clampedHeads = max(1, numAttentionHeads)
        let clampedHidden = max(clampedHeads, hiddenSize)
        self.hiddenSize = clampedHidden
        self.targetHiddenSize = clampedHidden
        self.vocabularySize = max(1, vocabularySize)
        self.numHiddenLayers = max(1, numHiddenLayers)
        self.numAttentionHeads = clampedHeads
        self.numKeyValueHeads = clampedHeads.isMultiple(of: 2) ? max(1, clampedHeads / 2) : clampedHeads
        self.intermediateSize = clampedHidden * 2
        self.blockSize = max(1, blockSize)
        self.maskTokenId = maskTokenId
        self.rmsNormEps = 1e-5
        self.ropeTheta = 10_000.0
    }

    public var headDim: Int { max(1, hiddenSize / max(1, numAttentionHeads)) }
}

final class DFlashAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlashConfig) {
        let heads = max(1, config.numAttentionHeads)
        let rawKV = max(1, min(heads, config.numKeyValueHeads))
        let kvHeads = heads.isMultiple(of: rawKV) ? rawKV : heads
        let dim = config.headDim
        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = dim
        self.scale = 1.0 / Float(dim).squareRoot()
        self.rope = RoPE(dimensions: dim, traditional: false, base: config.ropeTheta)
        self._wq.wrappedValue = Linear(config.hiddenSize, heads * dim, bias: false)
        self._wk.wrappedValue = Linear(config.hiddenSize, kvHeads * dim, bias: false)
        self._wv.wrappedValue = Linear(config.hiddenSize, kvHeads * dim, bias: false)
        self._wo.wrappedValue = Linear(heads * dim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
    }
}

extension DFlashAttention {
    func callAsFunction(_ x: MLXArray, context: MLXArray, queryOffset: Int) -> MLXArray {
        let (batch, blockLen) = (x.dim(0), x.dim(1))
        let ctxLen = context.dim(1)
        let joint = ctxLen > 0 ? concatenated([context, x], axis: 1) : x
        let totalLen = joint.dim(1)

        let rawQ = qNorm(wq(x).reshaped(batch, blockLen, numHeads, headDim)).transposed(0, 2, 1, 3)
        let q = rope(rawQ, offset: max(0, queryOffset))
        let rawK = kNorm(wk(joint).reshaped(batch, totalLen, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let k = applyJointRoPE(rawK, ctxLen: ctxLen, queryOffset: queryOffset)
        let v = wv(joint).reshaped(batch, totalLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        return wo(attended.transposed(0, 2, 1, 3).reshaped(batch, blockLen, -1))
    }

    private func applyJointRoPE(_ keys: MLXArray, ctxLen: Int, queryOffset: Int) -> MLXArray {
        if ctxLen > 0, queryOffset >= ctxLen {
            return rope(keys, offset: queryOffset - ctxLen)
        }
        if ctxLen > 0 {
            let kCtx = rope(keys[0..., 0..., ..<ctxLen, 0...], offset: 0)
            let kBlk = rope(keys[0..., 0..., ctxLen..., 0...], offset: max(0, queryOffset))
            return concatenated([kCtx, kBlk], axis: 2)
        }
        return rope(keys, offset: max(0, queryOffset))
    }
}

final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: DFlashConfig) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class DFlashLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DFlashConfig) {
        self._attention.wrappedValue = DFlashAttention(config)
        self._mlp.wrappedValue = DFlashMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, queryOffset: Int) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), context: context, queryOffset: queryOffset)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public struct DFlashProposal {
    public let tokens: [Int]
    public let logits: MLXArray

    public init(tokens: [Int], logits: MLXArray) {
        self.tokens = tokens
        self.logits = logits
    }
}

public final class DFlashDraftModel: Module {
    public let config: DFlashConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "context_norm") public var contextNorm: RMSNorm
    @ModuleInfo(key: "context_proj") public var contextProj: Linear
    @ModuleInfo(key: "layers") var layers: [DFlashLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public init(_ config: DFlashConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._contextNorm.wrappedValue = RMSNorm(dimensions: config.targetHiddenSize, eps: config.rmsNormEps)
        self._contextProj.wrappedValue = Linear(config.targetHiddenSize, config.hiddenSize, bias: false)
        var builtLayers: [DFlashLayer] = []
        for _ in 0 ..< config.numHiddenLayers { builtLayers.append(DFlashLayer(config)) }
        self._layers.wrappedValue = builtLayers
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }
}

extension DFlashDraftModel {
    public func encodeDraftHidden(
        anchorTokens: MLXArray,
        targetHidden: MLXArray,
        queryOffset: Int = 0,
        draftCount: Int? = nil
    ) -> MLXArray {
        let count = max(1, draftCount ?? config.blockSize)
        let anchor = anchorTokens.ndim == 1 ? anchorTokens.reshaped(-1, 1) : anchorTokens
        let masks = MLXArray.full([anchor.dim(0), count], values: MLXArray(Int32(config.maskTokenId)))
        let blockInput = concatenated([anchor.asType(.int32), masks], axis: 1)

        var hidden = embedTokens(blockInput)
        let context = contextProj(contextNorm(targetHidden))
        for layer in layers {
            hidden = layer(hidden, context: context, queryOffset: queryOffset)
        }
        return norm(hidden[0..., 1..., 0...])
    }

    public func callAsFunction(
        anchorTokens: MLXArray,
        targetHidden: MLXArray,
        queryOffset: Int = 0,
        draftCount: Int? = nil
    ) -> MLXArray {
        lmHead(encodeDraftHidden(
            anchorTokens: anchorTokens,
            targetHidden: targetHidden,
            queryOffset: queryOffset,
            draftCount: draftCount
        ))
    }

    public func propose(
        anchorToken: Int,
        targetHidden: MLXArray,
        queryOffset: Int = 0,
        draftCount: Int? = nil
    ) -> DFlashProposal {
        let anchor = MLXArray([Int32(anchorToken)]).reshaped(1, 1)
        let logits = self(anchorTokens: anchor, targetHidden: targetHidden, queryOffset: queryOffset, draftCount: draftCount)
        let greedy = MLX.argMax(logits[0], axis: -1)
        eval(logits, greedy)
        return DFlashProposal(tokens: greedy.asArray(Int32.self).map { Int($0) }, logits: logits)
    }
}

public struct DFlashVerificationResult: Equatable, Sendable {
    public let acceptedDraftCount: Int
    public let emittedTokens: [Int]
    public let cacheTrimCount: Int

    public init(acceptedDraftCount: Int, emittedTokens: [Int], cacheTrimCount: Int) {
        self.acceptedDraftCount = acceptedDraftCount
        self.emittedTokens = emittedTokens
        self.cacheTrimCount = cacheTrimCount
    }
}

public enum DFlashVerifier: Sendable {
    private struct ProbabilityTables {
        let vocab: Int
        let limit: Int
        let pFlat: [Float]
        let qFlat: [Float]
    }

    public static func verifyGreedy(draftTokens: [Int], targetLogits: MLXArray) -> DFlashVerificationResult {
        let rows = targetLogits.ndim == 3 ? targetLogits[0] : targetLogits
        let greedyTensor = MLX.argMax(rows, axis: -1)
        eval(greedyTensor)
        let targetGreedy = greedyTensor.asArray(Int32.self).map { Int($0) }
        let limit = min(draftTokens.count, max(0, targetGreedy.count - 1))

        var accepted = 0
        var emitted: [Int] = []
        emitted.reserveCapacity(limit + 1)

        for index in 0 ..< limit {
            let token = targetGreedy[index]
            emitted.append(token)
            if token == draftTokens[index] { accepted += 1 } else { break }
        }
        if accepted == limit, limit < targetGreedy.count {
            emitted.append(targetGreedy[limit])
        }
        return DFlashVerificationResult(
            acceptedDraftCount: accepted,
            emittedTokens: emitted,
            cacheTrimCount: max(0, draftTokens.count - accepted)
        )
    }

    public static func verifyRejection(
        draftTokens: [Int],
        draftLogits: MLXArray,
        targetLogits: MLXArray,
        temperature: Float,
        acceptanceUniforms: [Float],
        resampleUniforms: [Float]
    ) -> DFlashVerificationResult {
        guard temperature > 0 else {
            return verifyGreedy(draftTokens: draftTokens, targetLogits: targetLogits)
        }
        let tables = computeTables(
            draftCount: draftTokens.count,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: temperature
        )
        return runRejectionLoop(
            draftTokens: draftTokens,
            tables: tables,
            acceptanceUniforms: acceptanceUniforms,
            resampleUniforms: resampleUniforms
        )
    }

    private static func computeTables(
        draftCount: Int,
        draftLogits: MLXArray,
        targetLogits: MLXArray,
        temperature: Float
    ) -> ProbabilityTables {
        let tRows = (targetLogits.ndim == 3 ? targetLogits[0] : targetLogits).asType(.float32) / temperature
        let dRows = (draftLogits.ndim == 3 ? draftLogits[0] : draftLogits).asType(.float32) / temperature
        let targetProbs = MLX.softmax(tRows, axis: -1)
        let draftProbs = MLX.softmax(dRows, axis: -1)
        eval(targetProbs, draftProbs)
        let limit = min(draftCount, min(draftProbs.dim(0), max(0, targetProbs.dim(0) - 1)))
        return ProbabilityTables(
            vocab: targetProbs.dim(-1),
            limit: limit,
            pFlat: targetProbs.asArray(Float.self),
            qFlat: draftProbs.asArray(Float.self)
        )
    }

    private static func runRejectionLoop(
        draftTokens: [Int],
        tables: ProbabilityTables,
        acceptanceUniforms: [Float],
        resampleUniforms: [Float]
    ) -> DFlashVerificationResult {
        var accepted = 0
        var emitted: [Int] = []
        emitted.reserveCapacity(tables.limit + 1)

        for index in 0 ..< tables.limit {
            let candidate = draftTokens[index]
            let pVal = (0 ..< tables.vocab).contains(candidate) ? tables.pFlat[index * tables.vocab + candidate] : 0
            let qVal = (0 ..< tables.vocab).contains(candidate) ? max(tables.qFlat[index * tables.vocab + candidate], 1e-12) : 1e-12
            let u = index < acceptanceUniforms.count ? acceptanceUniforms[index] : 0
            if u < min(Float(1.0), pVal / qVal) {
                emitted.append(candidate)
                accepted += 1
            } else {
                let r = index < resampleUniforms.count ? resampleUniforms[index] : 0
                let base = index * tables.vocab
                emitted.append(sampleResidual(
                    pSlice: Array(tables.pFlat[base ..< (base + tables.vocab)]),
                    qSlice: Array(tables.qFlat[base ..< (base + tables.vocab)]),
                    uniform: r
                ))
                break
            }
        }
        if accepted == tables.limit, (tables.limit + 1) * tables.vocab <= tables.pFlat.count {
            let rBonus = tables.limit < resampleUniforms.count ? resampleUniforms[tables.limit] : 0
            let base = tables.limit * tables.vocab
            emitted.append(sampleCategorical(probs: Array(tables.pFlat[base ..< (base + tables.vocab)]), uniform: rBonus))
        }
        return DFlashVerificationResult(
            acceptedDraftCount: accepted,
            emittedTokens: emitted,
            cacheTrimCount: max(0, draftTokens.count - accepted)
        )
    }

    private static func sampleResidual(pSlice: [Float], qSlice: [Float], uniform: Float) -> Int {
        var residual = [Float](repeating: 0, count: pSlice.count)
        var sum: Float = 0
        for index in 0 ..< pSlice.count {
            let diff = max(Float(0), pSlice[index] - qSlice[index])
            residual[index] = diff
            sum += diff
        }
        guard sum > 1e-12 else { return sampleCategorical(probs: pSlice, uniform: uniform) }
        let invSum = 1.0 / sum
        for index in 0 ..< residual.count { residual[index] *= invSum }
        return sampleCategorical(probs: residual, uniform: uniform)
    }

    private static func sampleCategorical(probs: [Float], uniform: Float) -> Int {
        guard !probs.isEmpty else { return 0 }
        let target = min(max(uniform, 0), 0.99999994)
        var cumulative: Float = 0
        for (index, prob) in probs.enumerated() {
            cumulative += max(0, prob)
            if target < cumulative { return index }
        }
        return probs.count - 1
    }
}

public protocol DFlashTargetModel {
    func newCache() throws -> [any KVCache]
    func forwardWithHidden(tokens: MLXArray, cache: [any KVCache]?) -> (logits: MLXArray, hiddenStates: MLXArray)
    func trimCache(_ cache: [any KVCache], count: Int)
}

final class DFlashResidualProbe: ForwardProbe {
    private(set) var lastResidual: MLXArray?

    func residual(layer _: Int, _ hidden: MLXArray) -> MLXArray {
        lastResidual = hidden
        return hidden
    }

    func attentionWeights(layer _: Int, _: MLXArray) {}
    func wantsAttention(layer _: Int) -> Bool { false }
    func headMask(layer _: Int) -> MLXArray? { nil }
    func routerLogits(layer _: Int, _: MLXArray) {}
    func expertMask(layer _: Int) -> MLXArray? { nil }
}

public struct LanguageModelDFlashTarget: DFlashTargetModel {
    public let model: any LanguageModel
    public let hiddenSize: Int

    public init(_ model: any LanguageModel, hiddenSize: Int) {
        self.model = model
        self.hiddenSize = max(1, hiddenSize)
    }

    public func newCache() throws -> [any KVCache] { try model.newCache(parameters: nil) }

    public func forwardWithHidden(tokens: MLXArray, cache: [any KVCache]?) -> (logits: MLXArray, hiddenStates: MLXArray) {
        let probe = DFlashResidualProbe()
        let output = withForwardProbe(probe) { model(LMInput.Text(tokens: tokens), cache: cache, state: nil) }
        if let residual = probe.lastResidual {
            return (output.logits, residual)
        }
        return (output.logits, MLXArray.zeros([tokens.dim(0), tokens.dim(-1), hiddenSize], type: Float.self))
    }

    public func trimCache(_ cache: [any KVCache], count: Int) {
        guard count > 0 else { return }
        _ = trimPromptCache(cache, numTokens: count)
    }
}

public struct DFlashGenerationResult: Equatable, Sendable {
    public let tokens: [Int]
    public let generatedTokens: [Int]
    public let telemetry: SpeculativeDecodingTelemetry

    public init(tokens: [Int], generatedTokens: [Int], telemetry: SpeculativeDecodingTelemetry) {
        self.tokens = tokens
        self.generatedTokens = generatedTokens
        self.telemetry = telemetry
    }
}

public enum DFlashBlockDrafter {
    fileprivate struct RolloutSession {
        let cache: [any KVCache]
        var anchor: Int
        var generated: [Int]
        var contextHidden: MLXArray
        var currentOffset: Int
        var rounds = 0
        var draftedCount = 0
        var acceptedCount = 0
        var verifiedPositions = 0
    }

    public static func generateGreedy(
        target: some DFlashTargetModel,
        drafter: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        eosTokenId: Int? = nil
    ) throws -> DFlashGenerationResult {
        guard !promptTokens.isEmpty, maxNewTokens > 0 else {
            return DFlashGenerationResult(tokens: promptTokens, generatedTokens: [], telemetry: SpeculativeDecodingTelemetry())
        }
        var session = try startSession(target: target, promptTokens: promptTokens)
        if eosTokenId == session.anchor || session.generated.count >= maxNewTokens {
            return DFlashGenerationResult(
                tokens: promptTokens + session.generated,
                generatedTokens: session.generated,
                telemetry: SpeculativeDecodingTelemetry(emittedTokenCount: session.generated.count)
            )
        }
        while session.generated.count < maxNewTokens {
            if runDraftRound(target: target, drafter: drafter, session: &session, maxNewTokens: maxNewTokens, eosTokenId: eosTokenId) {
                break
            }
        }
        return finishSession(promptTokens: promptTokens, session: session)
    }

    public static func soloGreedy(
        target: some DFlashTargetModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        eosTokenId: Int? = nil
    ) throws -> [Int] {
        guard !promptTokens.isEmpty, maxNewTokens > 0 else { return promptTokens }
        let cache = try target.newCache()
        let promptTensor = MLXArray(promptTokens.map { Int32($0) })[.newAxis, 0...]
        var (logits, _) = target.forwardWithHidden(tokens: promptTensor, cache: cache)
        var nextTensor = MLX.argMax(logits[0, -1, 0...], axis: -1)
        eval(cache.flatMap(\.state) + [nextTensor])

        var generated: [Int] = []
        generated.reserveCapacity(maxNewTokens)
        for index in 0 ..< maxNewTokens {
            let nextToken = Int(nextTensor.item(Int32.self))
            generated.append(nextToken)
            if eosTokenId == nextToken || index == maxNewTokens - 1 { break }
            let stepTensor = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            (logits, _) = target.forwardWithHidden(tokens: stepTensor, cache: cache)
            nextTensor = MLX.argMax(logits[0, -1, 0...], axis: -1)
            eval(cache.flatMap(\.state) + [nextTensor])
        }
        return promptTokens + generated
    }
}

extension DFlashBlockDrafter {
    fileprivate static func startSession(target: some DFlashTargetModel, promptTokens: [Int]) throws -> RolloutSession {
        let cache = try target.newCache()
        let promptTensor = MLXArray(promptTokens.map { Int32($0) })[.newAxis, 0...]
        let (prefillLogits, prefillHidden) = target.forwardWithHidden(tokens: promptTensor, cache: cache)
        let firstTokenTensor = MLX.argMax(prefillLogits[0, -1, 0...], axis: -1)
        eval(cache.flatMap(\.state) + [firstTokenTensor, prefillHidden])
        let firstToken = Int(firstTokenTensor.item(Int32.self))
        return RolloutSession(
            cache: cache,
            anchor: firstToken,
            generated: [firstToken],
            contextHidden: prefillHidden,
            currentOffset: promptTokens.count
        )
    }

    fileprivate static func runDraftRound(
        target: some DFlashTargetModel,
        drafter: DFlashDraftModel,
        session: inout RolloutSession,
        maxNewTokens: Int,
        eosTokenId: Int?
    ) -> Bool {
        let remaining = maxNewTokens - session.generated.count
        let blockLen = min(max(1, drafter.config.blockSize), remaining)
        let proposal = drafter.propose(
            anchorToken: session.anchor,
            targetHidden: session.contextHidden,
            queryOffset: session.currentOffset,
            draftCount: blockLen
        )
        let fed = [session.anchor] + proposal.tokens
        let fedTensor = MLXArray(fed.map { Int32($0) })[.newAxis, 0...]
        let (verifyLogits, verifyHidden) = target.forwardWithHidden(tokens: fedTensor, cache: session.cache)
        let verification = DFlashVerifier.verifyGreedy(draftTokens: proposal.tokens, targetLogits: verifyLogits)
        if verification.cacheTrimCount > 0 { target.trimCache(session.cache, count: verification.cacheTrimCount) }

        let accepted = verification.acceptedDraftCount
        session.contextHidden = verifyHidden[0..., 0 ... accepted, 0...]
        eval(session.cache.flatMap(\.state) + [session.contextHidden])
        session.currentOffset += accepted + 1

        let (roundEmitted, hitEOS) = clampEmitted(verification.emittedTokens, remaining: remaining, eosTokenId: eosTokenId)
        session.generated.append(contentsOf: roundEmitted)
        session.anchor = roundEmitted[roundEmitted.count - 1]
        session.rounds += 1
        session.draftedCount += blockLen
        session.acceptedCount += accepted
        session.verifiedPositions += fed.count
        return hitEOS
    }

    fileprivate static func finishSession(promptTokens: [Int], session: RolloutSession) -> DFlashGenerationResult {
        let telemetry = SpeculativeDecodingTelemetry(
            roundCount: session.rounds,
            draftTokenCount: session.draftedCount,
            acceptedDraftTokenCount: session.acceptedCount,
            targetModelCallCount: session.rounds,
            draftModelCallCount: session.rounds,
            targetVerifiedTokenCount: session.verifiedPositions,
            emittedTokenCount: session.generated.count
        )
        return DFlashGenerationResult(
            tokens: promptTokens + session.generated,
            generatedTokens: session.generated,
            telemetry: telemetry
        )
    }

    private static func clampEmitted(_ emitted: [Int], remaining: Int, eosTokenId: Int?) -> (tokens: [Int], hitEOS: Bool) {
        let prefix = Array(emitted.prefix(remaining))
        if let eos = eosTokenId, let eosIndex = prefix.firstIndex(of: eos) {
            return (Array(prefix[...eosIndex]), true)
        }
        return (prefix, false)
    }
}
