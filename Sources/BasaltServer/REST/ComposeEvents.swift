import Foundation
import BasaltCore

package struct ComposeEvent: Codable, Sendable, Equatable {
    package var at: Date
    package var kind: String
    package var model: String?
    package var message: String

    package init(at: Date = Date(), kind: String, model: String? = nil, message: String) { (self.at, self.kind, self.model, self.message) = (at, kind, model, message) }
}

package actor ComposeEvents {
    static let kept = 100
    private var recent: [ComposeEvent] = []
    private var listeners: [UUID: AsyncStream<ComposeEvent>.Continuation] = [:]

    package init() {}

    package func emit(_ kind: String, model: String? = nil, _ message: String) {
        let event = ComposeEvent(kind: kind, model: model, message: message)
        recent = Array((recent + [event]).suffix(Self.kept))
        for listener in listeners.values { listener.yield(event) }
    }

    package func stream() -> AsyncStream<ComposeEvent> {
        let (events, continuation) = AsyncStream.makeStream(of: ComposeEvent.self, bufferingPolicy: .bufferingNewest(Self.kept))
        for event in recent { continuation.yield(event) }
        let id = UUID()
        listeners[id] = continuation
        continuation.onTermination = { [weak self] _ in Task { await self?.drop(id) } }
        return events
    }

    package var history: [ComposeEvent] { recent }

    func drop(_ id: UUID) { listeners[id] = nil }
}
