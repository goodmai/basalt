import Testing
import Foundation
@testable import BasaltCore

@Suite("KVPlanner")
struct KVPlannerTests {
    static let geometry = KVCacheGeometry(parameterProxy: 4)
    static let adaptive = KVPlanner(maxTokens: 4096, precisionLadder: KVCacheGeometry.adaptiveKVPrecisionLadder, zones: nil)

    @Test("A request with room and relaxed memory is admitted")
    func roomAdmits() {
        #expect(KVRoom(promptTokens: 100, fitting: 500, kvPrecision: nil).refusal(under: .relaxed) == nil)
    }

    @Test("A request with no room left is refused for context length, naming the fitting tokens and prompt + 1")
    func noRoomRefusesForContext() throws {
        let refusal = try #require(KVRoom(promptTokens: 500, fitting: 500, kvPrecision: nil).refusal(under: .tight))
        guard case .contextLengthExceeded(let maxTokens, let requested) = refusal else {
            Issue.record("expected contextLengthExceeded, got \(refusal)")
            return
        }
        #expect(maxTokens == 500)
        #expect(requested == 501)
    }

    @Test("Critical memory refuses long prompts and full rooms with 503, and admits short prompts that fit",
          arguments: [(MemoryDirective.longContextTokens, 100_000, true), (100, 100, true), (100, 1_000, false)])
    func criticalMemoryRefusesLongOrFull(promptTokens: Int, fitting: Int, refused: Bool) {
        let refusal = KVRoom(promptTokens: promptTokens, fitting: fitting, kvPrecision: nil).refusal(under: .critical)
        guard refused else {
            #expect(refusal == nil)
            return
        }
        guard case .memoryPressure(let retryAfter)? = refusal else {
            Issue.record("expected memoryPressure, got \(String(describing: refusal))")
            return
        }
        #expect(retryAfter == MemoryDirective.retryAfterSeconds)
    }

    @Test("With a budget that holds fp16 the adaptive ladder keeps fp16")
    func adaptiveKeepsFP16WhenItFits() {
        let budget = Self.geometry.bytes(tokens: 8_192) * 2
        let room = Self.adaptive.fit(promptTokens: 1_000, wantedTokens: 2_000, budget: budget)
        #expect(room.kvPrecision == nil)
        #expect(room.fitting >= 2_000)
        #expect(room.promptTokens == 1_000)
    }

    @Test("When fp16 cannot hold the request the ladder steps down to the first precision that does")
    func adaptiveStepsDown() {
        let wanted = 50_000
        let budget = Self.geometry.bytes(tokens: TokenCount(wanted), kvPrecision: .q4) * 2
        let fp16 = Self.geometry.maxTokens(fitting: budget, kvPrecision: nil, promptTokens: 1_000, zones: nil).rawValue
        let room = Self.adaptive.fit(promptTokens: 1_000, wantedTokens: wanted, budget: budget)
        #expect(fp16 < wanted)
        #expect(room.kvPrecision != nil)
        #expect(room.fitting >= wanted)
    }

    @Test("When no precision holds the request the ladder returns the one that fits most")
    func adaptiveFallsBackToLargest() {
        let budget = Self.geometry.bytes(tokens: 2_000)
        let room = Self.adaptive.fit(promptTokens: 1_000, wantedTokens: 10_000_000, budget: budget)
        let best = KVCacheGeometry.adaptiveKVPrecisionLadder
            .map { Self.geometry.maxTokens(fitting: budget, kvPrecision: $0?.uniform, promptTokens: 1_000, zones: nil).rawValue }
            .max()
        #expect(room.fitting == best)
    }

    @Test("Without zones a mixed precision is planned as its uniform equivalent")
    func mixedPrecisionWithoutZonesIsUniform() {
        let planner = KVPlanner(maxTokens: 4096, precisionLadder: [.k8v4], zones: nil)
        #expect(planner.fit(promptTokens: 10, wantedTokens: 20, budget: ByteCount(1 << 30)).kvPrecision == KVPrecision.k8v4.uniform)
    }

    @Test("Zones apply only when the model takes them, and without eviction support they evict nothing")
    func activeZonesFollowModelSupport() {
        let zones = KVZones.standard.evicting(0.5)
        var planner = KVPlanner(maxTokens: 4096, precisionLadder: [nil], zones: zones)
        #expect(planner.activeZones == zones)
        planner.takesEviction = false
        #expect(planner.activeZones == zones.evicting(0))
        planner.takesZones = false
        #expect(planner.activeZones == nil)
    }

    @Test("Room asks for the prompt plus the requested tokens, at most --max-tokens, and one token when none is asked")
    func roomWantsPromptPlusRequested() {
        let planner = KVPlanner(maxTokens: 100, precisionLadder: [nil, .q4], zones: nil)
        let budget = Self.geometry.bytes(tokens: 1_000) / 2
        #expect(planner.room(promptTokens: 10, requestedMaxTokens: nil, budget: budget)
            == planner.fit(promptTokens: 10, wantedTokens: 11, budget: budget))
        #expect(planner.room(promptTokens: 10, requestedMaxTokens: 1_000_000, budget: budget)
            == planner.fit(promptTokens: 10, wantedTokens: 110, budget: budget))
    }

    @Test("A request over the cap is capped, one without max_tokens gets the cap, and both carry the room's precision")
    func validatedCapsAndCarriesPrecision() throws {
        let planner = KVPlanner(maxTokens: 300, precisionLadder: [nil], zones: nil)
        let room = KVRoom(promptTokens: 100, fitting: 600, kvPrecision: .q8)
        let over = try planner.validated(GenerationRequest(prompt: "p", maxTokens: 1_000), in: room)
        #expect(over.maxTokens == 300)
        #expect(over.kvPrecision == .q8)
        #expect(try planner.validated(GenerationRequest(prompt: "p"), in: room).maxTokens == 300)
        #expect(try planner.validated(GenerationRequest(prompt: "p", maxTokens: 50), in: room).maxTokens == 50)
        #expect(planner.cap(for: KVRoom(promptTokens: 100, fitting: 200, kvPrecision: nil)) == 100)
    }

    @Test("Effective max tokens is the smallest of the request, --max-tokens and the room, never negative")
    func effectiveMaxTokens() {
        let planner = KVPlanner(maxTokens: 300, precisionLadder: [nil], zones: nil)
        let room = KVRoom(promptTokens: 100, fitting: 350, kvPrecision: nil)
        #expect(planner.effectiveMaxTokens(requested: nil, room: room) == 250)
        #expect(planner.effectiveMaxTokens(requested: 40, room: room) == 40)
        #expect(planner.effectiveMaxTokens(requested: 40, room: KVRoom(promptTokens: 100, fitting: 50, kvPrecision: nil)) == 0)
    }

    @Test("A reservation covers the prompt plus max_tokens, or --max-tokens when the request names none")
    func reservationBytes() {
        let planner = KVPlanner(maxTokens: 300, precisionLadder: [nil], zones: nil)
        #expect(planner.reservationBytes(promptTokens: 100, for: GenerationRequest(prompt: "p", maxTokens: 50))
            == Self.geometry.bytes(tokens: 150))
        #expect(planner.reservationBytes(promptTokens: 100, for: GenerationRequest(prompt: "p"))
            == Self.geometry.bytes(tokens: 400))
    }

    @Test("The first reservation is always admitted; later ones only while the total stays within the budget")
    func reservationsAdmitWithinBudget() {
        var reservations = KVReservations()
        let first = reservations.reserve(500, for: 1, within: 100)
        let over = reservations.reserve(1, for: 2, within: 100)
        let within = reservations.reserve(100, for: 2, within: 600)
        #expect(first)
        #expect(!over)
        #expect(within)
        #expect(reservations.held == 600)
        reservations.release(1)
        #expect(reservations.bytes == [2: 100])
    }
}

@Suite("HybridArbiter local estimate")
struct HybridArbiterEstimateTests {
    static let freeCloud = CloudAssessment(capabilityGap: 0, estimatedCost: 0, remainingBudget: nil, latencySeconds: 1)

    static func estimate(promptTokens: Int = 100, wanted: Int = 200, fitting: Int = 1_000) -> LocalEstimate {
        LocalEstimate(promptTokens: promptTokens, wantedTokens: wanted, fittingTokens: fitting, decodeTokens: 100, tokensPerSecond: 10)
    }

    @Test("A request the local KV budget cannot hold goes to the cloud for memory")
    func overflowGoesToCloud() {
        #expect(HybridArbiter().cloudReason(Self.freeCloud, local: Self.estimate(wanted: 2_000)) == .memory)
    }

    @Test("A cloud answer the remaining budget cannot pay for stays local")
    func overBudgetStaysLocal() {
        let costly = CloudAssessment(capabilityGap: 1, estimatedCost: 2, remainingBudget: 1, latencySeconds: 1)
        #expect(HybridArbiter().cloudReason(costly, local: Self.estimate(wanted: 2_000)) == nil)
    }

    @Test("A capability gap sends the request to the cloud; past the local context limit it is context degradation")
    func capabilityAndDegradation() {
        let gap = CloudAssessment(capabilityGap: 0.5, estimatedCost: 0, remainingBudget: nil, latencySeconds: 1)
        #expect(HybridArbiter().cloudReason(gap, local: Self.estimate()) == .capability)
        #expect(HybridArbiter(localContextLimit: 50).cloudReason(gap, local: Self.estimate()) == .contextDegradation)
        #expect(HybridArbiter().cloudReason(Self.freeCloud, local: Self.estimate()) == nil)
    }
}
