import Foundation

struct KVRoom: Sendable, Equatable {
    let promptTokens: Int
    let fitting: Int
    let kvPrecision: KVPrecision?

    var room: Int { fitting - promptTokens }

    func refusal(under directive: MemoryDirective) -> BasaltError? {
        if directive.limitsAdmission, promptTokens >= MemoryDirective.longContextTokens || room <= 0 {
            return .memoryPressure(retryAfterSeconds: MemoryDirective.retryAfterSeconds)
        }
        return room > 0 ? nil : .contextLengthExceeded(maxTokens: fitting, requested: promptTokens + 1)
    }
}

struct KVPlanner: Sendable {
    let maxTokens: Int
    let precisionLadder: [KVPrecision?]
    let zones: KVZones?
    var takesZones = true
    var takesEviction = true
    var takesQuantization = true
    var geometry = KVCacheGeometry(parameterProxy: 4)

    var isAdaptive: Bool { precisionLadder.count > 1 }

    var activeZones: KVZones? {
        guard takesZones else { return nil }
        return takesEviction ? zones : zones?.evicting(0)
    }

    func fit(promptTokens: Int, wantedTokens: Int, budget: ByteCount) -> KVRoom {
        let zones = activeZones
        let fits = (takesQuantization ? precisionLadder : [nil]).map { wanted in
            let precision = zones == nil ? wanted?.uniform : wanted
            let fitting = geometry.maxTokens(
                fitting: budget, kvPrecision: precision, promptTokens: TokenCount(promptTokens), zones: zones).rawValue
            return KVRoom(promptTokens: promptTokens, fitting: fitting, kvPrecision: precision)
        }
        return fits.first { $0.fitting >= wantedTokens }
            ?? fits.max { $0.fitting < $1.fitting }
            ?? KVRoom(promptTokens: promptTokens, fitting: 0, kvPrecision: nil)
    }

    func room(promptTokens: Int, requestedMaxTokens: Int?, budget: ByteCount) -> KVRoom {
        fit(promptTokens: promptTokens, wantedTokens: promptTokens + min(requestedMaxTokens ?? 1, maxTokens), budget: budget)
    }

    func cap(for room: KVRoom) -> Int {
        min(maxTokens, room.room)
    }

    func effectiveMaxTokens(requested: Int?, room: KVRoom) -> Int {
        max(0, min(requested ?? maxTokens, maxTokens, room.room))
    }

    func validated(_ request: GenerationRequest, in room: KVRoom) throws(BasaltError) -> GenerationRequest {
        let cap = cap(for: room)
        let validated = try request.validated(defaultMaxTokens: cap).withKVPrecision(room.kvPrecision)
        guard let wanted = request.maxTokens, wanted > cap else { return validated }
        return validated.withMaxTokens(cap)
    }

    func reservationBytes(promptTokens: Int, for request: GenerationRequest) -> ByteCount {
        geometry.bytes(tokens: TokenCount(promptTokens + (request.maxTokens ?? maxTokens)), kvPrecision: request.kvPrecision, zones: activeZones)
    }
}

struct KVReservations: Sendable {
    private(set) var bytes: [Int: ByteCount] = [:]

    var held: ByteCount { bytes.values.reduce(.zero, +) }

    mutating func reserve(_ wanted: ByteCount, for ticket: Int, within budget: ByteCount) -> Bool {
        let held = held
        guard held == 0 || held + wanted <= budget else { return false }
        bytes[ticket] = wanted
        return true
    }

    mutating func release(_ ticket: Int) {
        bytes[ticket] = nil
    }
}
