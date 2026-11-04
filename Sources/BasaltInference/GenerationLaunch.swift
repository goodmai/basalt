import Foundation
import MLX
import MLXLMCommon
import BasaltCore

enum FreshCache: Equatable {
    case headSplit(KVHeadMask)
    case quest(topK: Int)
    case zoned(KVZones, KVPrecision)
    case stock

    static func plan(
        kvBitsSet: Bool, headMask: KVHeadMask?, questTopK: Int?, zones: KVZones?, precision: KVPrecision?
    ) -> FreshCache {
        if !kvBitsSet, let headMask { return .headSplit(headMask) }
        if !kvBitsSet, let questTopK { return .quest(topK: questTopK) }
        if let zones, let precision { return .zoned(zones, precision) }
        return .stock
    }
}

struct LookupOptions {
    var speculative = false
    var recycling = false
    var jumpForward = false
    var routesExperts = true

    var requested: Bool { speculative || jumpForward }
}

struct CachedPrompt {
    var reuse: (cache: PromptCache, reusedTokens: Int)?
    var cuts: [Int] = []
    var boundary: Int
    var blend: (store: CacheBlendStore, namespace: String)?
}

struct LaunchOptions {
    var params: GenerateParameters
    var components = GenerationComponents()
    var tools: [ToolSpec]?
    var keepsCache = false
    var batch: BatchedDecoder?
    var lookup = LookupOptions()
    var kvZones: KVZones?
    var kvHeadMask: KVHeadMask?
    var questTopK: Int?
    var kvPrecision: KVPrecision?
    var mtp = false
    var mtpBlockSize = 4
    var mtpPrefix: MTPPrefix?
}

struct MTPRequest {
    var drafter: any MTPDrafterModel
    var blockSize: Int
    var prefix: MTPPrefix?
}

struct GenerationLauncher {
    let input: LMInput
    let params: GenerateParameters
    let components: GenerationComponents
    let tools: [ToolSpec]?
    let context: ModelContext
    let started: ContinuousClock.Instant

    func handing(_ iterator: some TokenIteratorProtocol, promptTokens: Int) -> (AsyncStream<Generation>, Task<Void, Never>, TimeInterval) {
        MLXInferenceEngine.handOff(
            promptTokenCount: promptTokens, modelConfiguration: context.configuration, tokenizer: context.tokenizer,
            iterator: iterator, tools: tools, toolCallPolicy: params.toolCallPolicy, since: started)
    }

    func uncached(_ iterator: some TokenIteratorProtocol) -> MLXInferenceEngine.GenerationLaunch {
        let (stream, task, prefill) = handing(iterator, promptTokens: input.text.tokens.dim(-1))
        return MLXInferenceEngine.GenerationLaunch(stream: stream, task: task, cache: nil, reusedTokens: 0, prefillSeconds: prefill)
    }

    func fresh(_ plan: FreshCache, batch: BatchedDecoder?, lookup: LookupOptions) throws -> MLXInferenceEngine.GenerationLaunch {
        let model = context.model
        switch plan {
        case .headSplit(let mask):
            let cache = mask.replacingFullAttention(try model.newCache(parameters: params))
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: params, components: components))
        case .quest(let topK):
            let cache = QuestKVCache.replacingFullAttention(try model.newCache(parameters: params), topK: topK)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: params, components: components))
        case .zoned(let zones, let precision):
            var exact = params
            exact.kvBits = nil
            let cache = ZonedKVCache.replacingFullAttention(try model.newCache(parameters: exact), zones: zones, precision: precision)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: exact, components: components))
        case .stock:
            return try stock(batch: batch, lookup: lookup)
        }
    }

    func stock(batch: BatchedDecoder?, lookup: LookupOptions) throws -> MLXInferenceEngine.GenerationLaunch {
        let model = context.model
        if let batch, let layers = BatchedTokenIterator.batchableLayers(try model.newCache(parameters: params)), let seat = batch.seat() {
            try components.validate(parameters: params)
            return uncached(try BatchedTokenIterator(
                input: input, model: model, layers: layers,
                decoding: DecodingSettings(processor: components.logitProcessor(parameters: params), parameters: params),
                stopTokens: MLXInferenceEngine.batchStopTokens(context), seat: seat))
        }
        let fresh = lookup.requested && params.kvBits == nil && !MLXInferenceEngine.carriesMoreThanTokens(input)
            ? try model.newCache(parameters: params) : []
        guard PromptLookupTokenIterator.supports(fresh, routesExperts: lookup.routesExperts) else {
            return uncached(try TokenIterator(input: input, model: model, parameters: params, components: components))
        }
        try components.validate(parameters: params)
        return uncached(try PromptLookupTokenIterator(
            prompt: PromptLookupTokenIterator.Prompt(input: input, history: input.text.tokens.asArray(Int.self)), model: model, cache: fresh,
            decoding: DecodingSettings(processor: components.makeProcessor(parameters: params), parameters: params),
            lookup: lookup))
    }

    func mtp(_ request: MTPRequest, prompt: CachedPrompt, keepsCache: Bool) throws -> MLXInferenceEngine.GenerationLaunch {
        let (reuse, boundary) = (prompt.reuse, prompt.boundary)
        let reused = reuse?.reusedTokens ?? 0
        let stateful = request.drafter as? any StatefulMTPDrafterModel
        let staged = keepsCache && stateful != nil && boundary >= reused && boundary < input.text.tokens.dim(-1)
        var cache: PromptCache?
        var drafterState: MTPDrafterState?
        var kept: (cache: PromptCache, prefix: MTPPrefix)?
        if staged, let stateful {
            let layers = try reuse?.cache ?? PromptCache(context.model.newCache(parameters: params))
            var state = request.prefix?.state ?? stateful.makeState(parameters: params)
            if boundary > reused {
                kept = stage(layers, drafter: stateful, state: &state, tokens: reused ..< boundary)
            }
            cache = layers
            drafterState = state
        }
        let start = staged ? boundary : 0
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: input.text.tokens[.ellipsis, start...]), mainModel: context.model,
            drafter: request.drafter, mainCache: cache?.layers, drafterState: drafterState, parameters: params,
            blockSize: request.blockSize, components: components)
        let (stream, task, prefill) = handing(iterator, promptTokens: input.text.tokens.dim(-1) - start)
        return MLXInferenceEngine.GenerationLaunch(
            stream: stream, task: task, cache: kept?.cache, reusedTokens: staged ? reused : 0,
            prefillSeconds: prefill, mtpPrefix: kept?.prefix)
    }

    func stage(
        _ prompt: PromptCache, drafter: any StatefulMTPDrafterModel, state: inout MTPDrafterState, tokens: Range<Int>
    ) -> (cache: PromptCache, prefix: MTPPrefix)? {
        let head = input.text.tokens[.ellipsis, tokens]
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        let output = context.model(LMInput.Text(tokens: head)[text: .newAxis], cache: prompt.layers, state: emit)
        if let hidden = output.state?[mtpLastHiddenStatesKey] {
            drafter.prepareDrafterState(
                target: context.model, input: MTPPrefillInput(promptTokens: head, targetHidden: hidden, firstBonus: nil),
                state: &state, sampler: ArgMaxSampler())
        }
        eval(prompt.layers.flatMap { $0.innerState() } + state.cache.flatMap { $0.innerState() })
        return MTPPrefix(state).map { (prompt.snapshot(at: tokens.upperBound, state: nil), $0) }
    }

    func cached(_ prompt: CachedPrompt, batch: BatchedDecoder?, lookup: LookupOptions) throws -> MLXInferenceEngine.GenerationLaunch {
        let (reuse, blend) = (prompt.reuse, prompt.blend)
        try components.validate(parameters: params)
        var cache = try reuse?.cache ?? PromptCache(context.model.newCache(parameters: params))
        let reused = reuse?.reusedTokens ?? 0
        let blended = cache.isHybrid ? nil : blend.flatMap {
            $0.store.extend(
                cache, prompt: input.text.tokens.asArray(Int.self), from: reused, namespace: $0.namespace,
                model: context.model, tokenizer: context.tokenizer, prefillStep: params.prefill.stepSize ?? 512)
        }
        try Task.checkCancellation()
        if let blended {
            cache = blended.cache
            if blended.reused > 0 {
                BasaltLogger(module: "CacheBlend").info(
                    "Blended \(blended.tokens) prompt tokens: \(blended.reused) from cached chunks, \(blended.recomputed) of them recomputed")
            }
        }
        let head = try blended.map { (anchors: [PromptCache](), kept: $0.cache as PromptCache?, start: reused + $0.tokens, state: $0.cache.state) }
            ?? HybridPrefixSnapshot.prefillHead(input.text.tokens, into: cache, plan: prompt, model: context.model, prefill: params.prefill)
        let tail = LMInput(tokens: input.text.tokens[.ellipsis, head.start...])
        let processor = components.makeProcessor(parameters: params).map {
            PromptAnchoredProcessor(base: $0, prompt: input.text.tokens)
        }
        let hybrid = cache.isHybrid
        func launching(_ iterator: some TokenIteratorProtocol) -> MLXInferenceEngine.GenerationLaunch {
            let keepable = iterator.state == nil || hybrid
            let (stream, task, prefill) = handing(iterator, promptTokens: input.text.tokens.dim(-1) - reused)
            return MLXInferenceEngine.GenerationLaunch(
                stream: stream, task: task, cache: keepable ? head.kept : nil, anchors: keepable ? head.anchors : [],
                reusedTokens: reused, exactTokens: blended?.exactPrefix ?? Int.max, prefillSeconds: prefill)
        }
        if let batch, head.state == nil, let layers = BatchedTokenIterator.batchableLayers(cache.layers), let seat = batch.seat() {
            return launching(try BatchedTokenIterator(
                input: tail, model: context.model, layers: layers, decoding: DecodingSettings(processor: processor, parameters: params),
                stopTokens: MLXInferenceEngine.batchStopTokens(context), seat: seat))
        }
        if lookup.requested, PromptLookupTokenIterator.supports(cache.layers, routesExperts: lookup.routesExperts) {
            return launching(try PromptLookupTokenIterator(
                prompt: PromptLookupTokenIterator.Prompt(input: tail, history: input.text.tokens.asArray(Int.self)), model: context.model,
                cache: cache.layers, state: head.state, decoding: DecodingSettings(processor: processor, parameters: params),
                lookup: lookup))
        }
        return launching(try TokenIterator(
            input: tail, model: context.model, cache: cache.layers, state: head.state,
            processor: processor, sampler: params.sampler(), prefill: params.prefill, maxTokens: params.maxTokens))
    }
}
