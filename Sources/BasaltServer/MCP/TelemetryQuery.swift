import Foundation
import BasaltCore

struct TelemetryQuery: Encodable {
    static let sections = ["windows", "decisions", "experts", "confidence", "health", "memory"]

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
    let health: HealthTelemetry?
    let memory: MemoryTelemetry?

    init(_ snapshot: ServerTelemetrySnapshot, sections: Set<String>) {
        totals = Totals(requests: snapshot.totalRequests, tokens: snapshot.totalTokensGenerated, activeRequests: snapshot.activeRequests, gpuActiveBytes: snapshot.gpuActiveBytes)
        windows = sections.contains("windows") ? snapshot.windows : nil
        decisions = sections.contains("decisions") ? snapshot.decisions : nil
        experts = sections.contains("experts") ? snapshot.experts : nil
        confidence = sections.contains("confidence") ? snapshot.confidence : nil
        health = sections.contains("health") ? snapshot.health : nil
        memory = sections.contains("memory") ? snapshot.memory : nil
    }

    static func validated(_ requested: [String]) -> Set<String>? {
        let wanted = Set(requested)
        return wanted.isSubset(of: Set(sections)) ? wanted : nil
    }
}
