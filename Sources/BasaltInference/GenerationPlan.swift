import BasaltCore

enum PromptMedia: Equatable, Sendable {
    case text
    case images(digests: [String], end: Int)
    case uncacheable

    var end: Int {
        switch self {
        case .images(_, let end): end
        case .text, .uncacheable: 0
        }
    }

    static func of(_ messages: [ChatMessageInput]?, carriesMoreThanTokens: Bool, tokens: [Int], imageToken: Int?) -> PromptMedia {
        guard carriesMoreThanTokens else { return .text }
        let digests = messages?.flatMap(\.images).map(\.digest) ?? []
        guard !digests.isEmpty, let last = imageToken.flatMap({ tokens.lastIndex(of: $0) }) else { return .uncacheable }
        return .images(digests: digests, end: last + 1)
    }
}

struct GenerationPlan: Equatable, Sendable {
    struct Settings: Sendable {
        var kvPrecision: KVPrecision?
        var kvZones: KVZones?
        var attendsThroughZones = false
        var prefillsInChunks = false
        var evictsThroughUpdate = false
        var kvHeadMask: KVHeadMask?
        var questTopK: Int?
        var speculativeLookup = false
        var capabilities = ModelCapabilities.unrestricted
        var adapter: String?
    }

    struct Facts: Sendable {
        var namespace: String?
        var promptTokens = 0
        var carriesMoreThanTokens = false
        var constrained = false
        var jumpsForward = false
        var hasMTPDrafter = false
        var blendAvailable = false
        var modelIsHybrid: Bool?
        var modelHasRecurrentState: Bool?
    }

    var kvZones: KVZones?
    var kvHeadMask: KVHeadMask?
    var questTopK: Int?
    var kvPrecision: KVPrecision?
    var speculativeLookup = false
    var drafts = false
    var findsAnchors = false
    var blends = false
    var batchable = false

    static func cacheNamespace(
        _ request: GenerationRequest, _ settings: Settings, cacheEnabled: Bool, media: PromptMedia, sameEpoch: Bool
    ) -> String? {
        guard sameEpoch, cacheEnabled, settings.capabilities.allows(.promptCacheReuse),
              (request.kvPrecision ?? settings.kvPrecision) == nil else { return nil }
        let key: String? = switch media {
        case .text: request.cacheKey
        case .images(let digests, _): request.cacheKey.map { "\($0)#images:\(digests.joined(separator: ","))" }
        case .uncacheable: nil
        }
        return key.map { key in settings.adapter.map { "\(key)|adapter:\($0)" } ?? key }
    }

    static func plainPrompt(_ request: GenerationRequest, _ settings: Settings, carriesMoreThanTokens: Bool) -> Bool {
        settings.capabilities.allows(.grammarJumpForward) && (request.kvPrecision ?? settings.kvPrecision) == nil && !carriesMoreThanTokens
    }

    static func make(_ request: GenerationRequest, _ settings: Settings, _ facts: Facts) -> GenerationPlan {
        let capabilities = settings.capabilities
        let precision = request.kvPrecision ?? settings.kvPrecision
        var plan = cache(precision: precision, settings)
        let tokensOnly = !facts.carriesMoreThanTokens
        plan.speculativeLookup = tokensOnly && capabilities.allows(.speculativeLookup) && (request.speculativeLookup ?? settings.speculativeLookup)
        plan.drafts = tokensOnly && capabilities.allows(.mtp) && drafts(plan, precision: precision, facts)
        plan.findsAnchors = tokensOnly && facts.namespace != nil && !plan.drafts && facts.modelHasRecurrentState == true
        plan.blends = tokensOnly && capabilities.allows(.cacheBlend) && facts.namespace != nil && facts.modelIsHybrid == false && facts.blendAvailable
        plan.batchable = capabilities.allows(.batchedDecode) && batchable(plan, request, precision: precision, facts)
        return plan
    }

    private static func drafts(_ plan: GenerationPlan, precision: KVPrecision?, _ facts: Facts) -> Bool {
        !facts.constrained && !plan.speculativeLookup && plan.kvZones == nil && precision == nil
            && facts.promptTokens <= MLXInferenceEngine.mtpPromptLimit && facts.hasMTPDrafter
    }

    private static func batchable(_ plan: GenerationPlan, _ request: GenerationRequest, precision: KVPrecision?, _ facts: Facts) -> Bool {
        !facts.jumpsForward && !plan.speculativeLookup && !plan.drafts
            && !plan.blends && precision == nil && !facts.carriesMoreThanTokens
    }

    static func cache(precision: KVPrecision?, _ settings: Settings) -> GenerationPlan {
        let capabilities = settings.capabilities
        let zones = zones(settings)
        let mask = capabilities.allows(.kvHeadSplit) ? settings.kvHeadMask : nil
        guard zones == nil else { return GenerationPlan(kvZones: zones, kvHeadMask: mask, kvPrecision: precision) }
        guard precision == nil else { return GenerationPlan() }
        return mask.map { GenerationPlan(kvHeadMask: $0) } ?? GenerationPlan(questTopK: capabilities.allows(.quest) ? settings.questTopK : nil)
    }

    private static func zones(_ settings: Settings) -> KVZones? {
        guard settings.capabilities.allows(.kvZones) else { return nil }
        if settings.attendsThroughZones { return settings.prefillsInChunks ? settings.kvZones : settings.kvZones?.evicting(0) }
        return settings.evictsThroughUpdate ? settings.kvZones?.keepingMiddleFullPrecision() : nil
    }
}
