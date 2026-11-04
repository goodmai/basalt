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
    }

    struct Facts: Sendable {
        var namespace: String?
        var promptTokens = 0
        var carriesMoreThanTokens = false
        var constrained = false
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
        guard sameEpoch, !carriesMoreThanTokens, cacheEnabled, (request.kvPrecision ?? settings.kvPrecision) == nil else { return nil }
        return request.cacheKey
    }

    static func plainPrompt(_ request: GenerationRequest, _ settings: Settings, carriesMoreThanTokens: Bool) -> Bool {
        (request.kvPrecision ?? settings.kvPrecision) == nil && !carriesMoreThanTokens
    }

    static func make(_ request: GenerationRequest, _ settings: Settings, _ facts: Facts) -> GenerationPlan {
        let precision = request.kvPrecision ?? settings.kvPrecision
        var plan = cache(precision: precision, settings)
        plan.speculativeLookup = request.speculativeLookup ?? settings.speculativeLookup
        plan.drafts = !facts.constrained && !plan.speculativeLookup && plan.kvZones == nil && precision == nil
            && facts.promptTokens <= MLXInferenceEngine.mtpPromptLimit && facts.hasMTPDrafter
        plan.findsAnchors = facts.namespace != nil && !plan.drafts && facts.modelHasRecurrentState == true
        plan.blends = facts.namespace != nil && facts.modelIsHybrid == false && facts.blendAvailable
        plan.batchable = !facts.constrained && request.toolChoice?.forcedTool == nil && !plan.speculativeLookup && !plan.drafts
            && !plan.blends && plainPrompt(request, settings, carriesMoreThanTokens: facts.carriesMoreThanTokens)
        return plan
    }

    static func cache(precision: KVPrecision?, _ settings: Settings) -> GenerationPlan {
        let zones = settings.attendsThroughZones ? (settings.prefillsInChunks ? settings.kvZones : settings.kvZones?.evicting(0)) : nil
        guard zones == nil else { return GenerationPlan(kvZones: zones, kvPrecision: precision) }
        guard precision == nil else { return GenerationPlan() }
        return settings.kvHeadMask.map { GenerationPlan(kvHeadMask: $0) } ?? GenerationPlan(questTopK: settings.questTopK)
    }
}
