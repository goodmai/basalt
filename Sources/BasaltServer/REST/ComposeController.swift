import Foundation
import Hummingbird
import BasaltCore

package struct ComposeModelStatus: Codable, Sendable, Equatable {
    package var name: String
    package var source: String
    package var keep: String
    package var state: String
    package var lastUsed: Date?
    package var idleUnload: Int?
    package var spentToday: Double?
    package var limit: Double?
    package var refusal: String?

    package init(name: String, source: String, keep: String, state: String) {
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

package struct ComposeControl: Sendable {
    package let status: @Sendable () async -> ComposeStatus
    package let apply: @Sendable (String) async throws(BasaltError) -> ComposeDiff

    package init(status: @escaping @Sendable () async -> ComposeStatus, apply: @escaping @Sendable (String) async throws(BasaltError) -> ComposeDiff) {
        (self.status, self.apply) = (status, apply)
    }
}

struct ComposeController: Sendable {
    static let bodyLimit = 256 << 10

    let control: ComposeControl

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.get("/compose") { _, _ in Self.json(await control.status()) }
        group.post("/compose/apply") { request, _ in
            do {
                let data = Data(try await request.body.collect(upTo: Self.bodyLimit).readableBytesView)
                guard let body = try? JSONDecoder().decode([String: String].self, from: data), let yaml = body["yaml"] else {
                    throw BasaltError.invalidRequestStructure(details: "compose apply takes {\"yaml\": \"<compose file>\"}")
                }
                return Self.json(try await control.apply(yaml))
            } catch {
                return ProviderErrorResponse.response(for: error, path: "/v1/compose/apply")
            }
        }
    }

    static func json(_ value: some Encodable) -> Response {
        let encoder = JSONEncoder()
        (encoder.dateEncodingStrategy, encoder.outputFormatting) = (.iso8601, [.sortedKeys])
        var buffer = ByteBuffer()
        buffer.writeBytes((try? encoder.encode(value)) ?? Data())
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }
}
