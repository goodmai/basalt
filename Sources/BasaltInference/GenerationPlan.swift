import BasaltCore

struct GenerationPlan: Equatable, Sendable {
    struct Settings: Sendable {
        var kvPrecision: KVPrecision?
        var kvZones: KVZones?
        var attendsThroughZones = false
        var prefillsInChunks = false
        var kvHeadMask: KVHeadMask?
        var questTopK: Int?
        var speculativeLookup = false
        var capabilities = ModelCapabilities.unrestricted
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
        _ request: GenerationRequest, _ settings: Settings, cacheEnabled: Bool, carriesMoreThanTokens: Bool, sameEpoch: Bool
    ) -> String? {
        guard sameEpoch, !carriesMoreThanTokens, cacheEnabled, settings.capabilities.allows(.promptCacheReuse),
              (request.kvPrecision ?? settings.kvPrecision) == nil else { return nil }
        return request.cacheKey
    }

    static func plainPrompt(_ request: GenerationRequest, _ settings: Settings, carriesMoreThanTokens: Bool) -> Bool {
        settings.capabilities.allows(.grammarJumpForward) && (request.kvPrecision ?? settings.kvPrecision) == nil && !carriesMoreThanTokens
    }

    static func make(_ request: GenerationRequest, _ settings: Settings, _ facts: Facts) -> GenerationPlan {
        let capabilities = settings.capabilities
        let precision = request.kvPrecision ?? settings.kvPrecision
        var plan = cache(precision: precision, settings)
        plan.speculativeLookup = capabilities.allows(.speculativeLookup) && (request.speculativeLookup ?? settings.speculativeLookup)
        plan.drafts = capabilities.allows(.mtp) && drafts(plan, precision: precision, facts)
        plan.findsAnchors = facts.namespace != nil && !plan.drafts && facts.modelHasRecurrentState == true
        plan.blends = capabilities.allows(.cacheBlend) && facts.namespace != nil && facts.modelIsHybrid == false && facts.blendAvailable
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
        let attends = settings.attendsThroughZones && capabilities.allows(.kvZones)
        let zones = attends ? (settings.prefillsInChunks ? settings.kvZones : settings.kvZones?.evicting(0)) : nil
        guard zones == nil else { return GenerationPlan(kvZones: zones, kvPrecision: precision) }
        guard precision == nil else { return GenerationPlan() }
        let mask = capabilities.allows(.kvHeadSplit) ? settings.kvHeadMask : nil
        return mask.map { GenerationPlan(kvHeadMask: $0) } ?? GenerationPlan(questTopK: capabilities.allows(.quest) ? settings.questTopK : nil)
    }
}
