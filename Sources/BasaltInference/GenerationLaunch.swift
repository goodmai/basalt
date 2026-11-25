import Foundation
import MLX
import MLXLMCommon
import BasaltCore

enum FreshCache: Equatable {
    case headSplit(KVHeadMask, HeadSplitKVCache.Zoning?)
    case quest(topK: Int)
    case sageMX(SageMXKVCache.Mode)
    case kvarn(useQJL: Bool)
    case zoned(KVZones, KVPrecision)
    case evicting(KVZones)
    case kEqVRow(KEqVRowLayout)
    case stock

    static func plan(
        kvBitsSet: Bool, headMask: KVHeadMask?, questTopK: Int?, zones: KVZones?, precision: KVPrecision?
    ) -> FreshCache {
        plan(options: LaunchOptions(
            params: GenerateParameters(kvBits: kvBitsSet ? 4 : nil),
            kvZones: zones, kvHeadMask: headMask, questTopK: questTopK, kvPrecision: precision))
    }

    static func plan(options: LaunchOptions) -> FreshCache {
        let kvBitsSet = options.params.kvBits != nil || options.params.kvScheme != nil
        if let zones = options.kvZones, zones.middleStaysFullPrecision, options.kvPrecision != nil { return .evicting(zones) }
        let zoning = options.kvZones.flatMap { zones in options.kvPrecision.map { HeadSplitKVCache.Zoning(zones: zones, precision: $0) } }
        if let headMask = options.kvHeadMask, !kvBitsSet || zoning != nil { return .headSplit(headMask, zoning) }
        if let precision = options.kvPrecision, precision.isKVarN, !kvBitsSet { return .kvarn(useQJL: precision.useQJL) }
        if !kvBitsSet, let questTopK = options.questTopK { return .quest(topK: questTopK) }
        if !kvBitsSet, options.attentionBackend != .standard {
            let mode: SageMXKVCache.Mode = options.attentionBackend == .sagemx ? .sagemx : .moba(topBlocks: options.mobaTopK)
            return .sageMX(mode)
        }
        if let zones = options.kvZones, let precision = options.kvPrecision { return .zoned(zones, precision) }
        if let layout = options.kEqVRow { return .kEqVRow(layout) }
        return .stock
    }
}

struct DraftConditions: Sendable, Equatable {
    var thermalLevel: ThermalLevel
    var isLowPowerModeEnabled: Bool

    static let nominal = DraftConditions(thermalLevel: .nominal, isLowPowerModeEnabled: false)

    static var live: DraftConditions {
        DraftConditions(
            thermalLevel: ThermalLevel(ProcessInfo.processInfo.thermalState),
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}

struct LookupOptions {
    var speculative = false
    var recycling = false
    var jumpForward = false
    var routesExperts = true
    var kvQuantization: GenerateParameters?
    var draftConditions: DraftConditions?

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
    var attentionBackend: AttentionBackend = .standard
    var mobaTopK = 8
    var kvPrecision: KVPrecision?
    var mtp = false
    var mtpBlockSize: Int?
    var mtpPrefix: MTPPrefix?
    var pacer: PrefillPacer?
    var kEqVRow: KEqVRowLayout?
    var prefillScratch: PrefillAttentionScratch?
}

struct MTPRequest {
    var drafter: any MTPDrafterModel
    var blockSize: Int
    var prefix: MTPPrefix?
}

extension MLXInferenceEngine {
    @TaskLocal static var prefillProgress: (@Sendable (_ processed: Int, _ total: Int) -> Void)?
}

extension GenerateParameters {
    var reportingPrefill: GenerateParameters {
        var reporting = self
        reporting.prefill.progress = MLXInferenceEngine.prefillProgress ?? prefill.progress
        return reporting
    }

    func capped(for model: any LanguageModel, promptTokens: Int, scratch: PrefillAttentionScratch? = nil) -> GenerateParameters {
        guard let cap = (model as? Inkling2Model)?.prefillStepCap(keys: promptTokens) ?? scratch?.stepCap(keys: promptTokens) else { return self }
        var capped = self
        capped.prefill.stepSize = min(prefill.stepSize ?? PrefillParameters.defaultStepSize, cap)
        return capped
    }

    func paced(by pacer: PrefillPacer?) -> GenerateParameters {
        guard let pacer else { return self }
        var paced = self
        paced.prefill.stepSize = pacer.stepSize(under: prefill.stepSize)
        paced.prefill.progress = pacer.progress(chaining: prefill.progress)
        return paced
    }
}

struct GenerationLauncher {
    let input: LMInput
    let params: GenerateParameters
    let components: GenerationComponents
    let tools: [ToolSpec]?
    let context: ModelContext
    let started: ContinuousClock.Instant

    func handing(_ iterator: some TokenIteratorProtocol, promptTokens: Int) -> (AsyncStream<Generation>, Task<Void, Never>, TimeInterval, OnyxReasoningBox?) {
        MLXInferenceEngine.handOffWithOnyx(
            promptTokenCount: promptTokens, context: context, iterator: iterator, tools: tools, toolCallPolicy: params.toolCallPolicy, since: started)
    }

    func uncached(_ iterator: some TokenIteratorProtocol) -> MLXInferenceEngine.GenerationLaunch {
        let (stream, task, prefill, onyx) = handing(iterator, promptTokens: input.text.tokens.dim(-1))
        return MLXInferenceEngine.GenerationLaunch(
            stream: stream, task: task, cache: nil, reusedTokens: 0, prefillSeconds: prefill, onyxReasoning: onyx)
    }

    func fresh(_ plan: FreshCache, batch: BatchedDecoder?, lookup: LookupOptions) throws -> MLXInferenceEngine.GenerationLaunch {
        let model = context.model
        switch plan {
        case .headSplit(let mask, let zoning):
            var exact = params
            (exact.kvBits, exact.kvScheme) = (nil, nil)
            let cache = mask.replacingFullAttention(try model.newCache(parameters: exact), zoning: zoning)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: exact, components: components))
        case .quest(let topK):
            let cache = QuestKVCache.replacingFullAttention(try model.newCache(parameters: params), topK: topK)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: params, components: components))
        case .sageMX(let mode):
            let cache = SageMXKVCache.replacingFullAttention(try model.newCache(parameters: params), mode: mode)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: params, components: components))
        case .kvarn(let useQJL):
            var exact = params
            (exact.kvBits, exact.kvScheme) = (nil, nil)
            let cache = KVarNKVCache.replacingFullAttention(try model.newCache(parameters: exact), useQJL: useQJL)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: exact, components: components))
        case .zoned(let zones, let precision):
            var exact = params
            (exact.kvBits, exact.kvScheme) = (nil, nil)
            let cache = ZonedKVCache.replacingFullAttention(try model.newCache(parameters: exact), zones: zones, precision: precision)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: exact, components: components))
        case .evicting(let zones):
            var exact = params
            (exact.kvBits, exact.kvScheme) = (nil, nil)
            let cache = EvictingKVCache.replacingFullAttention(try model.newCache(parameters: exact), zones: zones)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: exact, components: components))
        case .kEqVRow(let layout):
            let cache = KEqVRowKVCache.replacingFullAttention(try model.newCache(parameters: params), model: model, layout: layout)
            return uncached(try TokenIterator(input: input, model: model, cache: cache, parameters: params, components: components))
        case .stock:
            return try stock(batch: batch, lookup: lookup)
        }
    }

    func stock(batch: BatchedDecoder?, lookup: LookupOptions) throws -> MLXInferenceEngine.GenerationLaunch {
        let model = context.model
        let fresh = try model.newCache(parameters: params)
        let plan = prefillKVPlan(layers: fresh)
        plan.apply(to: fresh)
        defer { plan.restore(fresh) }
        let launchParams = parameters(for: plan)
        if let batch, let layers = BatchedTokenIterator.batchableLayers(fresh),
           let seat = batch.seat(in: context, kvBits: launchParams.kvBits) {
            try components.validate(parameters: launchParams)
            return uncached(try BatchedTokenIterator(
                input: input, model: model, layers: layers,
                decoding: DecodingSettings(processor: components.logitProcessor(parameters: launchParams), parameters: launchParams),
                stopTokens: MLXInferenceEngine.batchStopTokens(context), seat: seat))
        }
        guard lookup.requested, !MLXInferenceEngine.carriesMoreThanTokens(input),
              PromptLookupTokenIterator.supports(fresh, routesExperts: lookup.routesExperts) else {
            return uncached(try TokenIterator(input: input, model: model, cache: fresh, parameters: launchParams, components: components))
        }
        try components.validate(parameters: launchParams)
        var quantizing = lookup
        quantizing.kvQuantization = launchParams.kvBits == nil && launchParams.kvScheme == nil ? nil : launchParams
        return uncached(try PromptLookupTokenIterator(
            prompt: PromptLookupTokenIterator.Prompt(input: input, history: input.text.tokens.asArray(Int.self)), model: model, cache: fresh,
            decoding: DecodingSettings(processor: components.makeProcessor(parameters: launchParams), parameters: launchParams),
            lookup: quantizing))
    }

    func mtp(_ request: MTPRequest, prompt: CachedPrompt, keepsCache: Bool) throws -> MLXInferenceEngine.GenerationLaunch {
        try components.validate(parameters: params)
        let drafter = MTPSidecar.probing(request.drafter, processor: components.makeProcessor(parameters: params))
        let (reuse, boundary) = (prompt.reuse, prompt.boundary)
        let reused = reuse?.reusedTokens ?? 0
        let stateful = drafter as? any StatefulMTPDrafterModel
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
        try Task.checkCancellation()
        let start = staged ? boundary : 0
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: input.text.tokens[.ellipsis, start...]), mainModel: context.model,
            drafter: drafter, mainCache: cache?.layers, drafterState: drafterState, parameters: params,
            blockSize: request.blockSize, components: components)
        let (stream, task, prefill, onyx) = handing(iterator, promptTokens: input.text.tokens.dim(-1) - (staged ? reused : 0))
        return MLXInferenceEngine.GenerationLaunch(
            stream: stream, task: task, cache: kept?.cache, reusedTokens: TokenCount(staged ? reused : 0),
            prefillSeconds: prefill, mtpPrefix: kept?.prefix, onyxReasoning: onyx)
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

    func freshLayers(rows: KEqVRowLayout?) throws -> [any KVCache] {
        let layers = try context.model.newCache(parameters: params)
        return rows.map { KEqVRowKVCache.replacingFullAttention(layers, model: context.model, layout: $0) } ?? layers
    }

    func cached(_ prompt: CachedPrompt, batch: BatchedDecoder?, lookup: LookupOptions, rows: KEqVRowLayout? = nil) throws -> MLXInferenceEngine.GenerationLaunch {
        let (reuse, blend) = (prompt.reuse, prompt.blend)
        try components.validate(parameters: params)
        var cache = try reuse?.cache ?? PromptCache(freshLayers(rows: rows))
        let reused = reuse?.reusedTokens ?? 0
        let blended = cache.isHybrid ? nil : blend.flatMap {
            $0.store.extend(
                cache, prompt: input.text.tokens.asArray(Int.self), from: reused, namespace: $0.namespace,
                model: context.model, tokenizer: context.tokenizer, prefill: params.prefill)
        }
        try Task.checkCancellation()
        if let blended {
            cache = blended.cache
            if blended.reused > 0 {
                BasaltLogger(module: "CacheBlend").info(
                    "Blended \(blended.tokens) prompt tokens: \(blended.reused) from cached chunks, \(blended.recomputed) of them recomputed")
            }
        }
        let plan = prefillKVPlan(layers: cache.layers, reusedTokens: blended.map { reused + $0.tokens } ?? reused)
        plan.apply(to: cache.layers)
        defer { plan.restore(cache.layers) }
        let launchParams = parameters(for: plan)
        let head = try blended.map { (anchors: [PromptCache](), kept: $0.cache as PromptCache?, start: reused + $0.tokens, state: $0.cache.state) }
            ?? HybridPrefixSnapshot.prefillHead(
                input.text.tokens, into: cache, plan: prompt, model: context.model, prefill: launchParams.prefill, image: input.image)
        let tail = LMInput(text: LMInput.Text(tokens: input.text.tokens[.ellipsis, head.start...]), image: head.start == 0 ? input.image : nil)
        let processor = components.makeProcessor(parameters: launchParams).map {
            PromptAnchoredProcessor(base: $0, prompt: input.text.tokens)
        }
        let hybrid = cache.isHybrid
        func launching(_ iterator: some TokenIteratorProtocol) -> MLXInferenceEngine.GenerationLaunch {
            let keepable = iterator.state == nil || hybrid
            let (stream, task, prefill, onyx) = handing(iterator, promptTokens: input.text.tokens.dim(-1) - reused)
            return MLXInferenceEngine.GenerationLaunch(
                stream: stream, task: task, cache: keepable ? head.kept : nil, anchors: keepable ? head.anchors : [],
                reusedTokens: TokenCount(reused), exactTokens: TokenCount(blended?.exactPrefix ?? Int.max), prefillSeconds: prefill,
                onyxReasoning: onyx)
        }
        if let batch, head.state == nil, let layers = BatchedTokenIterator.batchableLayers(cache.layers),
           let seat = batch.seat(in: context, kvBits: launchParams.kvBits) {
            return launching(try BatchedTokenIterator(
                input: tail, model: context.model, layers: layers, decoding: DecodingSettings(processor: processor, parameters: launchParams),
                stopTokens: MLXInferenceEngine.batchStopTokens(context), seat: seat))
        }
        if lookup.requested, PromptLookupTokenIterator.supports(cache.layers, routesExperts: lookup.routesExperts) {
            return launching(try PromptLookupTokenIterator(
                prompt: PromptLookupTokenIterator.Prompt(input: tail, history: input.text.tokens.asArray(Int.self)), model: context.model,
                cache: cache.layers, state: head.state, decoding: DecodingSettings(processor: processor, parameters: launchParams),
                lookup: lookup))
        }
        return launching(try TokenIterator(
            input: tail, model: context.model, cache: cache.layers, state: head.state,
            processor: processor, sampler: launchParams.sampler(), prefill: launchParams.prefill, maxTokens: launchParams.maxTokens))
    }
}
