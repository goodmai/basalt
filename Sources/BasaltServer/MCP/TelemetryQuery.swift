import Foundation
import BasaltCore

struct TelemetryQuery: Encodable {
    static let sections = [
        "windows", "decisions", "experts", "confidence", "cache",
        "calibration", "health", "memory", "slo", "reasoning", "state_health",
    ]

    struct Totals: Encodable {
        let requests: Int
        let tokens: Int
        let activeRequests: Int
        let gpuActiveBytes: Int64

        enum CodingKeys: String, CodingKey {
            case requests, tokens
            case activeRequests = "active_requests", gpuActiveBytes = "gpu_active_bytes"
        }
    }

    let totals: Totals
    let windows: [TelemetryWindow]?
    let decisions: [String: DecisionTelemetry]?
    let experts: ExpertTelemetry?
    let confidence: ConfidenceTelemetry?
    let cache: CacheTelemetry?
    let calibration: DecisionCalibrationTelemetry?
    let health: HealthTelemetry?
    let memory: MemoryTelemetry?
    let slo: ServingSLOSnapshot?
    let reasoning: ReasoningTelemetrySnapshot?
    let stateHealth: RecurrentStateHealthSnapshot?

    enum CodingKeys: String, CodingKey {
        case totals, windows, decisions, experts, confidence, cache, calibration, health, memory, slo, reasoning
        case stateHealth = "state_health"
    }

    init(_ snapshot: ServerTelemetrySnapshot, sections: Set<String>) {
        totals = Totals(requests: snapshot.totalRequests, tokens: snapshot.totalTokensGenerated, activeRequests: snapshot.activeRequests, gpuActiveBytes: snapshot.gpuActiveBytes)
        windows = sections.contains("windows") ? snapshot.windows : nil
        decisions = sections.contains("decisions") ? snapshot.decisions : nil
        experts = sections.contains("experts") ? snapshot.experts : nil
        confidence = sections.contains("confidence") ? snapshot.confidence : nil
        cache = sections.contains("cache") ? snapshot.cache : nil
        calibration = sections.contains("calibration") ? snapshot.calibration : nil
        health = sections.contains("health") ? snapshot.health : nil
        memory = sections.contains("memory") ? snapshot.memory : nil
        slo = sections.contains("slo") ? snapshot.servingSLO : nil
        reasoning = sections.contains("reasoning") ? snapshot.reasoning : nil
        stateHealth = sections.contains("state_health") ? snapshot.stateHealth : nil
    }

    static func validated(_ requested: [String]) -> Set<String>? {
        let wanted = Set(requested)
        return wanted.isSubset(of: Set(sections)) ? wanted : nil
    }
}
