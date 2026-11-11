import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

package enum ComposeModelState: String, Codable, Sendable, CaseIterable {
    case loading, ready, unloading, unloaded, decision, cloud, failed, refused
    case idleUnloaded = "idle-unloaded", noKey = "no-key", overBudget = "over-budget", rateLimited = "rate-limited"

    package init(_ phase: OnDemandCore.Phase) {
        switch phase {
        case .unloaded: self = .idleUnloaded
        case .loading: self = .loading
        case .loaded: self = .ready
        case .unloading: self = .unloading
        }
    }

    package var serves: Bool {
        [.ready, .decision, .cloud, .idleUnloaded, .overBudget, .rateLimited].contains(self)
    }
}

extension ComposeStatus {
    package var waiting: [String] { models.filter { !$0.state.serves }.map { "\($0.name) (\($0.state.rawValue))" } }
}

struct HealthRoutes: Sendable {
    let orchestrator: ModelOrchestratorActor
    let compose: ComposeControl?

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/health/live") { _, _ in Self.json(.ok, ["status": "live"]) }
        group.get("/health/ready") { _, _ in
            let waiting = await waiting()
            return Self.json(waiting.isEmpty ? .ok : .serviceUnavailable, ["status": waiting.isEmpty ? "ready" : "loading", "waiting": waiting])
        }
    }

    func waiting() async -> [String] {
        if let compose { return await compose.status().waiting }
        return await orchestrator.isLoaded ? [] : ["chat (\(ComposeModelState.unloaded.rawValue))"]
    }

    static func json(_ status: HTTPResponse.Status, _ body: [String: any Sendable]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])) ?? Data()
        return Response(status: status, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
    }
}
