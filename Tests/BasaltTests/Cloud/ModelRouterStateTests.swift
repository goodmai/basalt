import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

@Suite("Model router state: reducer transitions and stale model syncs (#300)")
struct ModelRouterStateTests {
    typealias Model = OpenRouterClient.OpenRouterModel

    static func priced(_ price: String) -> Model {
        .init(id: "m", name: "M", contextLength: 8_192, pricing: .init(prompt: price, completion: price))
    }

    static let cheap = priced("0.000002")
    static let dear = priced("0.00001")

    static func pricePer1M(_ state: RouterState) -> Double {
        state.catalog.estimateCost(model: "m", promptTokens: 1_000_000, completionTokens: 0)
    }

    static let cases: [([RouterEvent], Double, Int)] = [
        ([.syncStarted, .synced(1, [cheap])], 2.0, 1),
        ([.syncStarted, .syncStarted, .synced(1, [dear]), .synced(2, [cheap])], 2.0, 2),
        ([.syncStarted, .syncStarted, .synced(2, [cheap]), .synced(1, [dear])], 2.0, 2),
        ([.syncStarted, .synced(1, [cheap]), .synced(1, [dear])], 2.0, 1),
        ([.syncStarted, .localRegistered(id: "m", estimatedRAM: 4_000)], 0.0, 0),
        ([.localModelSeen("q"), .syncStarted, .synced(1, [dear])], 10.0, 1)
    ]

    @Test("A sync applies only if no newer sync has been applied", arguments: cases)
    func reduce(events: [RouterEvent], price: Double, applied: Int) {
        var state = RouterState()
        for event in events {
            let next = RouterState.reduce(state, event)
            #expect(next.checkInvariants(after: state) == [])
            state = next
        }
        #expect(abs(Self.pricePer1M(state) - price) < 1e-9)
        #expect(state.syncApplied == applied)
    }

    static func startSync(_ router: ModelRouter) async -> (Task<Void, Error>, AsyncStream<[Model]>.Continuation) {
        let (reply, replier) = AsyncStream.makeStream(of: [Model].self)
        let (started, signal) = AsyncStream.makeStream(of: Void.self)
        let task = Task {
            try await router.syncCloudModels {
                signal.yield()
                return await reply.first { _ in true } ?? []
            }
        }
        _ = await started.first { _ in true }
        return (task, replier)
    }

    @Test("An older model sync that lands after a newer one is dropped, so the price never goes back")
    func staleSyncIsDropped() async throws {
        let router = ModelRouter(localOrchestrator: ModelOrchestratorActor(engine: MockInferenceEngine()), cloudClient: nil)
        let (older, olderReply) = await Self.startSync(router)
        let (newer, newerReply) = await Self.startSync(router)
        newerReply.yield([Self.cheap])
        try await newer.value
        olderReply.yield([Self.dear])
        try await older.value
        #expect(abs(await router.estimateCost(model: "m", promptTokens: 1_000_000, completionTokens: 0) - 2.0) < 1e-9)
    }
}
