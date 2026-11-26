import Foundation
import MLX
import MLXLMCommon
import MLXNN

public protocol BidirectionalBlockModel {
    var layerCount: Int { get }
    func forward(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray
    func forwardHidden(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray?
    func projectLogits(hidden: MLXArray) -> MLXArray?
}

extension BidirectionalBlockModel {
    public func forwardHidden(tokens _: MLXArray, caches _: [any KVCache]?) -> MLXArray? { nil }
    public func projectLogits(hidden _: MLXArray) -> MLXArray? { nil }
}

public struct LanguageModelBlockAdapter: BidirectionalBlockModel {
    public let model: any LanguageModel
    public let layerCount: Int

    public init(_ model: any LanguageModel) throws {
        self.model = model
        self.layerCount = try model.newCache(parameters: nil).count
    }

    public func forward(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray {
        model(LMInput.Text(tokens: tokens), cache: caches, state: nil).logits
    }

    public func forwardHidden(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray? {
        guard let diffusion = model as? any DiffusionLanguageModel else { return nil }
        return diffusion.hiddenStates(tokens, cache: caches)
    }

    public func projectLogits(hidden: MLXArray) -> MLXArray? {
        guard let diffusion = model as? any DiffusionLanguageModel else { return nil }
        return diffusion.projectLogits(hidden)
    }
}

public struct MLXModuleBlockAdapter<M: Module>: BidirectionalBlockModel {
    public let module: M
    public let layerCount: Int
    private let forwardImpl: (M, MLXArray, [any KVCache]?) -> MLXArray

    public init(
        module: M,
        layerCount: Int,
        forward: @escaping (M, MLXArray, [any KVCache]?) -> MLXArray
    ) {
        self.module = module
        self.layerCount = max(0, layerCount)
        self.forwardImpl = forward
    }

    public func forward(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray {
        forwardImpl(module, tokens, caches)
    }
}

public struct FastDLLMRequest: Equatable, Sendable {
    public var promptTokens: [Int]
    public var maxNewTokens: Int
    public var suffixTokens: [Int]
    public var includeTrailingMaskSuffix: Bool
    public var eosTokenId: Int?
    public var stopTokenIds: Set<Int> = []

    public init(
        promptTokens: [Int],
        maxNewTokens: Int,
        suffixTokens: [Int] = [],
        includeTrailingMaskSuffix: Bool = true,
        eosTokenId: Int? = nil
    ) {
        self.promptTokens = promptTokens
        self.maxNewTokens = max(0, maxNewTokens)
        self.suffixTokens = suffixTokens
        self.includeTrailingMaskSuffix = includeTrailingMaskSuffix
        self.eosTokenId = eosTokenId
    }
}

public struct FastDLLMGenerationResult: Equatable, Sendable {
    public let tokens: [Int]
    public let generatedTokens: [Int]
    public let forwardEvaluations: Int
    public let denoisingSteps: Int
    public let remaskedTokenCount: Int
    public let evaluatedTokenPositions: Int
    public var projectedLogitPositions: Int = 0

    public init(
        tokens: [Int],
        generatedTokens: [Int],
        forwardEvaluations: Int,
        denoisingSteps: Int,
        remaskedTokenCount: Int,
        evaluatedTokenPositions: Int
    ) {
        self.tokens = tokens
        self.generatedTokens = generatedTokens
        self.forwardEvaluations = forwardEvaluations
        self.denoisingSteps = denoisingSteps
        self.remaskedTokenCount = remaskedTokenCount
        self.evaluatedTokenPositions = evaluatedTokenPositions
    }
}

public struct FastDLLMSession<Model: BidirectionalBlockModel> {
    let model: Model
    let config: BlockDiffusionConfig
    let dualCache: DualBlockKVCache
    var state: FastDLLMRunner.RunState
    var nextBlockStart: Int
    var prefilled = false
    var suffixPrimed = false
    var finished: Bool
    var earlyCutoff: Int?

    public init(model: Model, request: FastDLLMRequest, config: BlockDiffusionConfig) {
        self.model = model
        self.config = config
        self.dualCache = DualBlockKVCache(layerCount: model.layerCount, bidirectionalPrompt: config.bidirectionalPrompt)
        self.state = FastDLLMRunner.makeInitialState(for: request, maskTokenId: config.maskTokenId)
        self.nextBlockStart = request.promptTokens.count
        self.finished = request.maxNewTokens <= 0
    }

    public var isFinished: Bool { finished }
    package var totalKernelInvocations: Int { dualCache.totalKernelInvocations }

    public mutating func prefillIfNeeded() {
        guard !prefilled else { return }
        prefilled = true
        guard !finished, config.dualCacheEnabled, state.promptLength > 0 else { return }
        FastDLLMRunner.prefillPrompt(model: model, state: &state, dualCache: dualCache)
    }

    public mutating func nextBlock() -> [Int]? {
        guard !finished else { return nil }
        prefillIfNeeded()
        guard nextBlockStart < state.generationEnd else {
            finished = true
            return nil
        }
        let blockSize = max(1, config.blockSize)
        let blockEnd = min(nextBlockStart + blockSize, state.generationEnd)
        let activeRange = nextBlockStart ..< blockEnd
        nextBlockStart = blockEnd
        let cutoff = FastDLLMRunner.runSingleBlock(
            model: model,
            state: &state,
            dualCache: dualCache,
            activeRange: activeRange,
            suffixPrimed: &suffixPrimed,
            config: config
        )
        if let cutoff {
            earlyCutoff = cutoff
            finished = true
            let clamped = min(max(cutoff, activeRange.lowerBound), activeRange.upperBound)
            return clamped > activeRange.lowerBound ? Array(state.canvas[activeRange.lowerBound ..< clamped]) : []
        }
        if nextBlockStart >= state.generationEnd {
            finished = true
        }
        return Array(state.canvas[activeRange])
    }

    public func buildResult() -> FastDLLMGenerationResult {
        FastDLLMRunner.buildResult(from: state, cutoff: earlyCutoff)
    }
}

public enum FastDLLMRunner {
    struct RunState {
        let promptLength: Int
        let generationEnd: Int
        let usesTrailingSuffix: Bool
        let stopTokenIds: Set<Int>
        var canvas: [Int]
        var finalizedBlockEnds: [Int] = []
        var forwardEvaluations = 0
        var denoisingSteps = 0
        var remaskedTokenCount = 0
        var evaluatedTokenPositions = 0
        var projectedLogitPositions = 0
    }

    fileprivate struct BlockSpan {
        let range: Range<Int>
        let totalLength: Int
    }

    fileprivate struct BlockPredictions {
        let tokens: [Int]
        let confidences: [Float]
        let margins: [Float]
    }

    public static func generate(
        model: some BidirectionalBlockModel,
        request: FastDLLMRequest,
        config: BlockDiffusionConfig
    ) -> FastDLLMGenerationResult {
        guard request.maxNewTokens > 0 else {
            return emptyResult(for: request)
        }
        var session = FastDLLMSession(model: model, request: request, config: config)
        while session.nextBlock() != nil {}
        return session.buildResult()
    }

    private static func emptyResult(for request: FastDLLMRequest) -> FastDLLMGenerationResult {
        FastDLLMGenerationResult(
            tokens: request.promptTokens + request.suffixTokens,
            generatedTokens: [],
            forwardEvaluations: 0,
            denoisingSteps: 0,
            remaskedTokenCount: 0,
            evaluatedTokenPositions: 0
        )
    }

    static func makeInitialState(for request: FastDLLMRequest, maskTokenId: Int) -> RunState {
        let promptLength = request.promptTokens.count
        let middle = Array(repeating: maskTokenId, count: request.maxNewTokens)
        return RunState(
            promptLength: promptLength,
            generationEnd: promptLength + request.maxNewTokens,
            usesTrailingSuffix: request.includeTrailingMaskSuffix || !request.suffixTokens.isEmpty,
            stopTokenIds: request.suffixTokens.isEmpty ? request.stopTokenIds.union(request.eosTokenId.map { [$0] } ?? []) : [],
            canvas: request.promptTokens + middle + request.suffixTokens
        )
    }

    static func buildResult(from state: RunState, cutoff: Int?) -> FastDLLMGenerationResult {
        let end = cutoff ?? state.generationEnd
        let fullTokens = cutoff.map { Array(state.canvas[..<$0]) } ?? state.canvas
        var result = FastDLLMGenerationResult(
            tokens: fullTokens,
            generatedTokens: Array(state.canvas[state.promptLength ..< end]),
            forwardEvaluations: state.forwardEvaluations,
            denoisingSteps: state.denoisingSteps,
            remaskedTokenCount: state.remaskedTokenCount,
            evaluatedTokenPositions: state.evaluatedTokenPositions
        )
        result.projectedLogitPositions = state.projectedLogitPositions
        return result
    }
}

extension FastDLLMRunner {
    static func runSingleBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>,
        suffixPrimed: inout Bool,
        config: BlockDiffusionConfig
    ) -> Int? {
        let span = BlockSpan(range: activeRange, totalLength: state.usesTrailingSuffix ? state.canvas.count : activeRange.upperBound)
        if config.dualCacheEnabled {
            prepareSuffixForBlock(
                model: model, state: &state, dualCache: dualCache,
                span: span, suffixPrimed: &suffixPrimed, config: config
            )
        }
        denoiseBlock(model: model, state: &state, dualCache: dualCache, span: span, config: config)
        state.finalizedBlockEnds.append(activeRange.upperBound)
        if let eosCutoff = completedEOSCutoff(state: state, upTo: activeRange.upperBound, maskTokenId: config.maskTokenId) {
            return eosCutoff
        }
        if config.dualCacheEnabled, activeRange.upperBound < state.generationEnd {
            commitBlock(model: model, state: &state, dualCache: dualCache, activeRange: activeRange, keepSuffix: config.reuseSuffixCache)
        }
        return nil
    }

    private static func prepareSuffixForBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        span: BlockSpan,
        suffixPrimed: inout Bool,
        config: BlockDiffusionConfig
    ) {
        let suffixEnd = config.maxSuffixLength.map { min(span.totalLength, span.range.upperBound + $0) } ?? span.totalLength
        guard span.range.upperBound < suffixEnd else {
            dualCache.clearSuffix()
            return
        }
        if config.reuseSuffixCache, suffixPrimed {
            dualCache.advanceSuffix(by: span.range.count)
            if dualCache.suffixLength > 0 { return }
        }
        dualCache.prepareSuffixPrefill(at: span.range.upperBound)
        let suffixTensor = MLXArray(state.canvas[span.range.upperBound ..< suffixEnd].map { Int32($0) })[.newAxis, 0...]
        evaluateCacheOnlyForward(model: model, tokens: suffixTensor, dualCache: dualCache, state: &state)
        suffixPrimed = true
    }

    static func prefillPrompt(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache
    ) {
        dualCache.preparePromptPrefill()
        let prompt = MLXArray(state.canvas[0 ..< state.promptLength].map { Int32($0) })[.newAxis, 0...]
        evaluateCacheOnlyForward(model: model, tokens: prompt, dualCache: dualCache, state: &state)
    }

    private static func commitBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>,
        keepSuffix: Bool
    ) {
        if !keepSuffix {
            dualCache.clearSuffix()
        }
        dualCache.prepareBlockCommit(at: activeRange.lowerBound)
        let blockTensor = MLXArray(state.canvas[activeRange].map { Int32($0) })[.newAxis, 0...]
        evaluateCacheOnlyForward(model: model, tokens: blockTensor, dualCache: dualCache, state: &state)
    }

    private static func evaluateCacheOnlyForward(
        model: some BidirectionalBlockModel,
        tokens: MLXArray,
        dualCache: DualBlockKVCache,
        state: inout RunState
    ) {
        if let hidden = model.forwardHidden(tokens: tokens, caches: dualCache.kvCaches) {
            let cacheState = dualCache.layers.flatMap { $0.innerState() }
            eval(cacheState.isEmpty ? [hidden] : cacheState)
        } else {
            let logits = model.forward(tokens: tokens, caches: dualCache.kvCaches)
            eval(dualCache.layers.flatMap { $0.innerState() } + [logits])
            state.projectedLogitPositions += tokens.dim(1)
        }
        state.forwardEvaluations += 1
        state.evaluatedTokenPositions += tokens.dim(1)
    }
}

extension FastDLLMRunner {
    fileprivate static func denoiseBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        span: BlockSpan,
        config: BlockDiffusionConfig
    ) {
        let maxSteps = max(1, config.maxStepsPerBlock)
        let remaskEnabled = (config.remaskThreshold ?? 0) > 0
        for step in 0 ..< maxSteps {
            let maskedMask = span.range.map { state.canvas[$0] == config.maskTokenId }
            let hasMasks = maskedMask.contains(true)
            if !hasMasks && (!remaskEnabled || step == 0 || step >= maxSteps - 1) { break }
            let predictions = evaluateActivePredictions(
                model: model, state: &state, dualCache: dualCache,
                span: span, maskedMask: maskedMask, config: config
            )
            applyStep(
                state: &state, activeRange: span.range, maskedMask: maskedMask,
                predictions: predictions, step: step, config: config
            )
            if !hasMasks && !span.range.contains(where: { state.canvas[$0] == config.maskTokenId }) { break }
            if completedEOSCutoff(state: state, upTo: span.range.upperBound, maskTokenId: config.maskTokenId) != nil {
                break
            }
        }
    }

    private static func evaluateActivePredictions(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        span: BlockSpan,
        maskedMask: [Bool],
        config: BlockDiffusionConfig
    ) -> BlockPredictions {
        state.forwardEvaluations += 1
        state.denoisingSteps += 1
        let remaskEnabled = (config.remaskThreshold ?? 0) > 0
        let maskedOffsets = remaskEnabled ? [] : maskedMask.enumerated().compactMap { $0.element ? $0.offset : nil }
        if config.dualCacheEnabled {
            dualCache.prepareActiveBlock(at: span.range.lowerBound)
            let activeTensor = MLXArray(state.canvas[span.range].map { Int32($0) })[.newAxis, 0...]
            state.evaluatedTokenPositions += span.range.count
            if !remaskEnabled, !maskedOffsets.isEmpty,
               let hidden = model.forwardHidden(tokens: activeTensor, caches: dualCache.kvCaches)
            {
                return projectSparsePredictions(
                    model: model, hidden: hidden, blockCount: span.range.count,
                    maskedOffsets: maskedOffsets, state: &state
                )
            }
            let logits = model.forward(tokens: activeTensor, caches: dualCache.kvCaches)
            state.projectedLogitPositions += span.range.count
            return extractPredictions(from: logits)
        }
        return evaluateUncachedPredictions(
            model: model, state: &state, span: span,
            maskedOffsets: maskedOffsets, remaskEnabled: remaskEnabled
        )
    }

    private static func evaluateUncachedPredictions(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        span: BlockSpan,
        maskedOffsets: [Int],
        remaskEnabled: Bool
    ) -> BlockPredictions {
        let mask = DualBlockKVCache.makeBlockDiffusionMask(
            promptLength: state.promptLength, finalizedBlockEnds: state.finalizedBlockEnds,
            activeRange: span.range, totalLength: span.totalLength
        )
        let uncached = DualBlockKVCache.uncachedMaskCaches(layerCount: model.layerCount, mask: mask)
        let fullTensor = MLXArray(state.canvas[0 ..< span.totalLength].map { Int32($0) })[.newAxis, 0...]
        state.evaluatedTokenPositions += span.totalLength
        if !remaskEnabled, !maskedOffsets.isEmpty,
           let fullHidden = model.forwardHidden(tokens: fullTensor, caches: uncached)
        {
            let blockHidden = fullHidden[0..., span.range.lowerBound ..< span.range.upperBound, 0...]
            return projectSparsePredictions(
                model: model, hidden: blockHidden, blockCount: span.range.count,
                maskedOffsets: maskedOffsets, state: &state
            )
        }
        let fullLogits = model.forward(tokens: fullTensor, caches: uncached)
        state.projectedLogitPositions += span.totalLength
        return extractPredictions(from: fullLogits[0..., span.range.lowerBound ..< span.range.upperBound, 0...])
    }
}

extension FastDLLMRunner {
    private static func projectSparsePredictions(
        model: some BidirectionalBlockModel,
        hidden: MLXArray,
        blockCount: Int,
        maskedOffsets: [Int],
        state: inout RunState
    ) -> BlockPredictions {
        let allMasked = maskedOffsets.count == blockCount
        let selectedHidden = allMasked ? hidden : hidden[0..., MLXArray(maskedOffsets.map(Int32.init)), 0...]
        guard let logits = model.projectLogits(hidden: selectedHidden) else {
            return BlockPredictions(
                tokens: [Int](repeating: 0, count: blockCount),
                confidences: [Float](repeating: 0, count: blockCount),
                margins: [Float](repeating: 0, count: blockCount)
            )
        }
        state.projectedLogitPositions += maskedOffsets.count
        let compact = extractPredictions(from: logits)
        if allMasked { return compact }
        var tokens = [Int](repeating: 0, count: blockCount)
        var confidences = [Float](repeating: 1.0, count: blockCount)
        var margins = [Float](repeating: 1.0, count: blockCount)
        for (compactIdx, offset) in maskedOffsets.enumerated() {
            tokens[offset] = compact.tokens[compactIdx]
            confidences[offset] = compact.confidences[compactIdx]
            margins[offset] = compact.margins[compactIdx]
        }
        return BlockPredictions(tokens: tokens, confidences: confidences, margins: margins)
    }

    private static func extractPredictions(from blockLogits: MLXArray) -> BlockPredictions {
        let confidence = Device.defaultDevice().deviceType == .gpu
            ? FusedLogitConfidence.compute(blockLogits)
            : FusedLogitConfidence.reference(blockLogits)
        let probs = MLX.exp(confidence.top1Logprob)
        let margins = probs - MLX.exp(confidence.top1Logprob - confidence.top2Margin)
        eval(confidence.top1TokenID, probs, margins)
        return BlockPredictions(
            tokens: confidence.top1TokenID[0].asArray(Int32.self).map { Int($0) },
            confidences: probs[0].asArray(Float.self),
            margins: margins[0].asArray(Float.self)
        )
    }

    fileprivate static func applyStep(
        state: inout RunState,
        activeRange: Range<Int>,
        maskedMask: [Bool],
        predictions: BlockPredictions,
        step: Int,
        config: BlockDiffusionConfig
    ) {
        let maxSteps = max(1, config.maxStepsPerBlock)
        if step >= maxSteps - 1 {
            for (offset, isMasked) in maskedMask.enumerated() where isMasked {
                state.canvas[activeRange.lowerBound + offset] = predictions.tokens[offset]
            }
            return
        }
        let selected = FastDLLMUnmasker.selectPositions(
            confidences: predictions.confidences,
            maskedMask: maskedMask,
            config: config,
            top2Margins: predictions.margins
        )
        let newlyUnmasked = Set(selected)
        for offset in selected {
            state.canvas[activeRange.lowerBound + offset] = predictions.tokens[offset]
        }
        let remainingMasks = maskedMask.enumerated().filter { $0.element && !newlyUnmasked.contains($0.offset) }.count
        let remaskBudget = max(0, (maxSteps - 1 - step) - remainingMasks)
        guard remaskBudget > 0, config.remaskThreshold != nil else { return }

        let ineligibleMask = maskedMask.enumerated().map { $0.element || newlyUnmasked.contains($0.offset) }
        let remasked = FastDLLMUnmasker.selectRemaskPositions(
            confidences: predictions.confidences,
            maskedMask: ineligibleMask,
            config: config,
            maxRemaskCount: remaskBudget
        )
        for offset in remasked {
            state.canvas[activeRange.lowerBound + offset] = config.maskTokenId
        }
        state.remaskedTokenCount += remasked.count
    }

    fileprivate static func completedEOSCutoff(state: RunState, upTo upperBound: Int, maskTokenId: Int) -> Int? {
        guard !state.stopTokenIds.isEmpty else { return nil }
        for index in state.promptLength ..< upperBound {
            let token = state.canvas[index]
            if token == maskTokenId { return nil }
            if state.stopTokenIds.contains(token) { return index + 1 }
        }
        return nil
    }
}
