import Foundation
import Hummingbird
import BasaltCore

package struct ComposeModelStatus: Codable, Sendable, Equatable {
    package var name: String
    package var source: String
    package var keep: String
    package var state: ComposeModelState
    package var lastUsed: Date?
    package var idleUnload: Int?
    package var spentToday: Double?
    package var limit: Double?
    package var refusal: String?

    package init(name: String, source: String, keep: String, state: ComposeModelState) {
        (self.name, self.source, self.keep, self.state) = (name, source, keep, state)
    }

    enum CodingKeys: String, CodingKey {
        case name, source, keep, state, limit, refusal
        case lastUsed = "last_used", idleUnload = "idle_unload", spentToday = "spent_today"
    }
}

package struct ComposeStatus: Codable, Sendable, Equatable {
    package var name: String
    package var pipelines: [String]
    package var models: [ComposeModelStatus]
    package var spentToday: Double

    package init(name: String, pipelines: [String], models: [ComposeModelStatus], spentToday: Double) {
        (self.name, self.pipelines, self.models, self.spentToday) = (name, pipelines, models, spentToday)
    }

    enum CodingKeys: String, CodingKey {
        case name, pipelines, models
        case spentToday = "spent_today"
    }
}

package struct ComposeApply: Codable, Sendable, Equatable {
    package var yaml: String
    package var base: String?
    package var variables: [String: String]?
    package var profiles: [String]?

    package init(yaml: String, base: String? = nil, variables: [String: String]? = nil, profiles: [String]? = nil) {
        (self.yaml, self.base, self.variables, self.profiles) = (yaml, base, variables, profiles)
    }
}

package struct ComposeControl: Sendable {
    package let status: @Sendable () async -> ComposeStatus
    package let apply: @Sendable (ComposeApply) async throws(BasaltError) -> ComposeDiff
    package var preview: @Sendable (ComposeApply) async throws(BasaltError) -> ComposeDiff = { _ throws(BasaltError) in throw .invalidRequestStructure(details: "compose diff is not available") }
    package var events: ComposeEvents?
    package var heartbeat = Duration.seconds(15)

    package init(status: @escaping @Sendable () async -> ComposeStatus, apply: @escaping @Sendable (ComposeApply) async throws(BasaltError) -> ComposeDiff) {
        (self.status, self.apply) = (status, apply)
    }
}

struct ComposeController: Sendable {
    static let bodyLimit = 256 << 10

    let control: ComposeControl

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/compose") { _, _ in Self.json(await control.status()) }
        group.post("/compose/apply") { request, _ in await Self.change(request, path: "/v1/compose/apply", control.apply) }
        group.post("/compose/diff") { request, _ in await Self.change(request, path: "/v1/compose/diff", control.preview) }
        group.get("/compose/events") { _, _ in Self.events(control.events, heartbeat: control.heartbeat) }
    }

    static func change(_ request: Request, path: String, _ run: @Sendable (ComposeApply) async throws(BasaltError) -> ComposeDiff) async -> Response {
        do {
            let data = Data(try await request.body.collect(upTo: Self.bodyLimit).readableBytesView)
            guard let request = try? JSONDecoder().decode(ComposeApply.self, from: data) else {
                throw BasaltError.invalidRequestStructure(details: "\(path) takes {\"yaml\": \"<compose file>\", \"base\", \"variables\", \"profiles\"}")
            }
            return Self.json(try await run(request))
        } catch {
            return ProviderErrorResponse.response(for: error, path: path)
        }
    }

    static func events(_ events: ComposeEvents?, heartbeat: Duration) -> Response {
        guard let events else { return ProviderErrorResponse.response(status: .notFound, message: "this system has no compose events", shape: .openAI) }
        let encoder = JSONEncoder()
        (encoder.dateEncodingStrategy, encoder.outputFormatting) = (.iso8601, [.sortedKeys])
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                for await event in await events.stream() {
                    guard let data = try? encoder.encode(event) else { continue }
                    continuation.yield(ByteBuffer(string: "event: \(event.kind)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"))
                }
                continuation.finish()
            }
            let heartbeat = Task {
                while (try? await Task.sleep(for: heartbeat)) != nil { continuation.yield(ByteBuffer(string: ": keep-alive\n\n")) }
            }
            continuation.onTermination = { _ in
                task.cancel()
                heartbeat.cancel()
            }
        }
        return Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"], body: ResponseBody(asyncSequence: body))
    }

    static func json(_ value: some Encodable) -> Response {
        let encoder = JSONEncoder()
        (encoder.dateEncodingStrategy, encoder.outputFormatting) = (.iso8601, [.sortedKeys])
        var buffer = ByteBuffer()
        buffer.writeBytes((try? encoder.encode(value)) ?? Data())
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }
}
