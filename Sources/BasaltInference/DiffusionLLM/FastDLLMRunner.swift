import Foundation
import MLX
import MLXLMCommon
import MLXNN

public protocol BidirectionalBlockModel {
    var layerCount: Int { get }
    func forward(tokens: MLXArray, caches: [any KVCache]?) -> MLXArray
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

public enum FastDLLMRunner {
    fileprivate struct RunState {
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
        var state = makeInitialState(for: request, maskTokenId: config.maskTokenId)
        let dualCache = DualBlockKVCache(layerCount: model.layerCount, bidirectionalPrompt: config.bidirectionalPrompt)
        if config.dualCacheEnabled, state.promptLength > 0 {
            prefillPrompt(model: model, state: &state, dualCache: dualCache)
        }

        let blockSize = max(1, config.blockSize)
        for blockStart in stride(from: state.promptLength, to: state.generationEnd, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, state.generationEnd)
            if let earlyResult = runSingleBlock(
                model: model,
                state: &state,
                dualCache: dualCache,
                activeRange: blockStart ..< blockEnd,
                config: config
            ) {
                return earlyResult
            }
        }
        return buildResult(from: state, cutoff: nil)
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

    private static func makeInitialState(for request: FastDLLMRequest, maskTokenId: Int) -> RunState {
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

    fileprivate static func buildResult(from state: RunState, cutoff: Int?) -> FastDLLMGenerationResult {
        let end = cutoff ?? state.generationEnd
        let fullTokens = cutoff.map { Array(state.canvas[..<$0]) } ?? state.canvas
        return FastDLLMGenerationResult(
            tokens: fullTokens,
            generatedTokens: Array(state.canvas[state.promptLength ..< end]),
            forwardEvaluations: state.forwardEvaluations,
            denoisingSteps: state.denoisingSteps,
            remaskedTokenCount: state.remaskedTokenCount,
            evaluatedTokenPositions: state.evaluatedTokenPositions
        )
    }
}

extension FastDLLMRunner {
    fileprivate static func runSingleBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>,
        config: BlockDiffusionConfig
    ) -> FastDLLMGenerationResult? {
        let totalLength = state.usesTrailingSuffix ? state.canvas.count : activeRange.upperBound
        if config.dualCacheEnabled {
            prepareSuffixCache(
                model: model,
                state: &state,
                dualCache: dualCache,
                activeUpper: activeRange.upperBound,
                totalLength: totalLength
            )
        }
        denoiseBlock(
            model: model,
            state: &state,
            dualCache: dualCache,
            activeRange: activeRange,
            totalLength: totalLength,
            config: config
        )
        state.finalizedBlockEnds.append(activeRange.upperBound)
        if let eosCutoff = completedEOSCutoff(state: state, upTo: activeRange.upperBound, maskTokenId: config.maskTokenId) {
            return buildResult(from: state, cutoff: eosCutoff)
        }
        if config.dualCacheEnabled, activeRange.upperBound < state.generationEnd {
            commitBlock(model: model, state: &state, dualCache: dualCache, activeRange: activeRange)
        }
        return nil
    }

    fileprivate static func prefillPrompt(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache
    ) {
        dualCache.preparePromptPrefill()
        let prompt = MLXArray(state.canvas[0 ..< state.promptLength].map { Int32($0) })[.newAxis, 0...]
        let logits = model.forward(tokens: prompt, caches: dualCache.kvCaches)
        eval(dualCache.layers.flatMap { $0.innerState() } + [logits])
        state.forwardEvaluations += 1
        state.evaluatedTokenPositions += state.promptLength
    }

    fileprivate static func prepareSuffixCache(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeUpper: Int,
        totalLength: Int
    ) {
        guard activeUpper < totalLength else {
            dualCache.clearSuffix()
            return
        }
        dualCache.prepareSuffixPrefill(at: activeUpper)
        let suffixSlice = state.canvas[activeUpper ..< totalLength]
        let suffixTensor = MLXArray(suffixSlice.map { Int32($0) })[.newAxis, 0...]
        let logits = model.forward(tokens: suffixTensor, caches: dualCache.kvCaches)
        eval(dualCache.layers.flatMap { $0.innerState() } + [logits])
        state.forwardEvaluations += 1
        state.evaluatedTokenPositions += suffixSlice.count
    }

    fileprivate static func commitBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>
    ) {
        dualCache.clearSuffix()
        dualCache.prepareBlockCommit(at: activeRange.lowerBound)
        let blockTensor = MLXArray(state.canvas[activeRange].map { Int32($0) })[.newAxis, 0...]
        let logits = model.forward(tokens: blockTensor, caches: dualCache.kvCaches)
        eval(dualCache.layers.flatMap { $0.innerState() } + [logits])
        state.forwardEvaluations += 1
        state.evaluatedTokenPositions += activeRange.count
    }
}

extension FastDLLMRunner {
    fileprivate static func denoiseBlock(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>,
        totalLength: Int,
        config: BlockDiffusionConfig
    ) {
        let maxSteps = max(1, config.maxStepsPerBlock)
        let remaskEnabled = (config.remaskThreshold ?? 0) > 0
        for step in 0 ..< maxSteps {
            let maskedMask = activeRange.map { state.canvas[$0] == config.maskTokenId }
            let hasMasks = maskedMask.contains(true)
            if !hasMasks && (!remaskEnabled || step == 0 || step >= maxSteps - 1) { break }
            let blockLogits = evaluateActiveLogits(
                model: model,
                state: &state,
                dualCache: dualCache,
                activeRange: activeRange,
                totalLength: totalLength,
                dualCacheEnabled: config.dualCacheEnabled
            )
            let predictions = extractPredictions(from: blockLogits)
            applyStep(
                state: &state,
                activeRange: activeRange,
                maskedMask: maskedMask,
                predictions: predictions,
                step: step,
                config: config
            )
            if !hasMasks && !activeRange.contains(where: { state.canvas[$0] == config.maskTokenId }) { break }
            if completedEOSCutoff(state: state, upTo: activeRange.upperBound, maskTokenId: config.maskTokenId) != nil {
                break
            }
        }
    }

    fileprivate static func evaluateActiveLogits(
        model: some BidirectionalBlockModel,
        state: inout RunState,
        dualCache: DualBlockKVCache,
        activeRange: Range<Int>,
        totalLength: Int,
        dualCacheEnabled: Bool
    ) -> MLXArray {
        state.forwardEvaluations += 1
        state.denoisingSteps += 1
        if dualCacheEnabled {
            dualCache.prepareActiveBlock(at: activeRange.lowerBound)
            let activeTensor = MLXArray(state.canvas[activeRange].map { Int32($0) })[.newAxis, 0...]
            state.evaluatedTokenPositions += activeRange.count
            return model.forward(tokens: activeTensor, caches: dualCache.kvCaches)
        }
        let mask = DualBlockKVCache.makeBlockDiffusionMask(
            promptLength: state.promptLength,
            finalizedBlockEnds: state.finalizedBlockEnds,
            activeRange: activeRange,
            totalLength: totalLength
        )
        let uncached = DualBlockKVCache.uncachedMaskCaches(layerCount: model.layerCount, mask: mask)
        let fullTensor = MLXArray(state.canvas[0 ..< totalLength].map { Int32($0) })[.newAxis, 0...]
        state.evaluatedTokenPositions += totalLength
        let fullLogits = model.forward(tokens: fullTensor, caches: uncached)
        return fullLogits[0..., activeRange.lowerBound ..< activeRange.upperBound, 0...]
    }

    fileprivate static func extractPredictions(from blockLogits: MLXArray) -> BlockPredictions {
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
