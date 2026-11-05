import Foundation
import Testing
@testable import BasaltCore

@Suite("HuggingFaceHub state: one transfer per file, stale results dropped (#300)")
struct HuggingFaceHubStateTests {
    static let running = URL(fileURLWithPath: "/hub/running.safetensors")
    static let other = URL(fileURLWithPath: "/hub/other.safetensors")
    static let task = Task<Void, any Error> {}

    static let reduceCases: [(HubEvent, Int, Int, [String: Int])] = [
        (.fetched(bytes: 5), 5, 1, ["running.safetensors": 1]),
        (.transferStarted(running, generation: 2, task), 0, 1, ["running.safetensors": 1]),
        (.transferStarted(other, generation: 2, task), 0, 2, ["running.safetensors": 1, "other.safetensors": 2]),
        (.transferStarted(other, generation: 1, task), 0, 1, ["running.safetensors": 1]),
        (.transferFinished(running, generation: 1), 0, 1, [:]),
        (.transferFinished(running, generation: 0), 0, 1, ["running.safetensors": 1]),
        (.transferFinished(other, generation: 1), 0, 1, ["running.safetensors": 1]),
    ]

    @Test("reduce counts bytes, starts one transfer per file and drops a duplicate start or a stale finish", arguments: reduceCases)
    func reduce(event: HubEvent, bytes: Int, generation: Int, transfers: [String: Int]) {
        var base = HubState()
        base.generation = 1
        base.transfers[Self.running] = HubState.Transfer(generation: 1, task: Self.task)
        let next = HubState.reduce(base, event)
        #expect(next.bytesFetched == bytes)
        #expect(next.generation == generation)
        #expect(Dictionary(uniqueKeysWithValues: next.transfers.map { ($0.key.lastPathComponent, $0.value.generation) }) == transfers)
        #expect(next.checkInvariants(after: base).isEmpty)
        #expect(base.checkInvariants(after: HubState.reduce(base, .fetched(bytes: 1))).count == 1)
    }

    @Test("Two requests for one file while the first is in flight share one transfer: one HEAD, one GET, an intact file, nothing left running")
    func concurrentRequestsShareOneTransfer() async throws {
        let body = Data((0 ..< 200_000).map { UInt8(truncatingIfNeeded: $0) })
        let host = StubURLProtocol.host { request, _ in
            (200, ["Content-Length": "\(body.count)"], request.httpMethod == "HEAD" ? Data() : body)
        }
        let hub = HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("hub-state-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = HubFile(
            name: "model.safetensors",
            url: try #require(URL(string: "https://\(host)/org/tiny/resolve/main/model.safetensors")),
            destination: directory.appendingPathComponent("model.safetensors"))

        let (first, second) = await Self.requestTwice(file, on: hub)
        #expect(first == second)
        #expect(await hub.state.transfers.count == 1)
        try await first.value
        try await second.value

        let methods = StubURLProtocol.requests(for: host).map { $0.0.httpMethod ?? "GET" }
        #expect(methods.sorted() == ["GET", "HEAD"])
        #expect(try Data(contentsOf: file.destination) == body)
        #expect(await hub.state.transfers.isEmpty)
        #expect(await hub.state.generation == 1)
    }

    static func requestTwice(_ file: HubFile, on hub: isolated HuggingFaceHub) -> (Task<Void, any Error>, Task<Void, any Error>) {
        (hub.transferTask(file, token: nil) { _, _, _ in }, hub.transferTask(file, token: nil) { _, _, _ in })
    }
}
