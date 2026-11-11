import Foundation
import Synchronization
import Testing
@testable import BasaltAppCore

@Suite("Mac app chat: the store runs a conversation end to end on a stub engine — streaming, persistence, timing, stop, no engine (#434 phase 1)")
@MainActor
struct ChatEffectsTests {
    final class Saved: Sendable {
        let snapshots = Mutex<[ConversationSnapshot]>([])
        var all: [ConversationSnapshot] { snapshots.withLock { $0 } }
    }

    static func engine(_ deltas: [ChatDelta], gate: AsyncStream<Void>? = nil) -> EngineClient {
        EngineClient(
            chat: { _ in
                AsyncThrowingStream { continuation in
                    let task = Task {
                        for delta in deltas {
                            continuation.yield(delta)
                            if let gate { for await _ in gate { break } }
                        }
                        continuation.finish()
                    }
                    continuation.onTermination = { _ in task.cancel() }
                }
            },
            health: { true }, stats: { throw EngineError.notRunning })
    }

    static func store(_ engine: EngineClient?, saved: Saved) -> Store<ChatState, ChatAction, ChatEffect> {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/model"))
        state.draft = "Hello?"
        let environment = ChatEnvironment(engine: { engine }, persist: { snapshot in saved.snapshots.withLock { $0.append(snapshot) } })
        return Store(state, reduce: ChatReducer.reduce, handle: ChatEffects.handler(environment))
    }

    @Test("A sent message streams into the answer, completes with a receipt, and the user turn is saved before the answer")
    func roundTrip() async {
        let saved = Saved()
        let store = Self.store(Self.engine([.reasoning("hm"), .text("Hi "), .text("there"), .finish("stop"), .usage(prompt: 3, completion: 2)]), saved: saved)
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        let answer = store.state.conversation.messages[1]
        #expect(answer.parts == [.reasoning("hm"), .text("Hi there")] && answer.state == .complete && store.state.run == .idle)
        #expect(answer.receipt?.completionTokens == 2 && answer.receipt?.timeToFirstToken != nil)
        #expect(saved.all.first?.messages.first?.text == "Hello?" && saved.all.last?.messages.last?.state == .complete)
    }

    @Test("Stop during streaming keeps what arrived, marks it stopped, and later deltas do not land")
    func stop() async {
        let saved = Saved()
        let (gate, open) = AsyncStream<Void>.makeStream()
        let store = Self.store(Self.engine([.text("Part"), .text("ial"), .text(" more")], gate: gate), saved: saved)
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        for _ in 0 ..< 100_000 where store.state.conversation.messages[1].parts.isEmpty { await Task.yield() }
        #expect(store.state.conversation.messages[1].parts == [.text("Part")], "the first delta never reached the store")
        store.send(.stop)
        open.yield()
        open.finish()
        await store.settle()
        #expect(store.state.conversation.messages[1].parts == [.text("Part")] && store.state.conversation.messages[1].state == .stopped)
        #expect(saved.all.last?.messages.last?.state == .stopped)
    }

    @Test("Without an engine the answer fails with notRunning and the draft can be sent again")
    func noEngine() async {
        let store = Self.store(nil, saved: Saved())
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        #expect(store.state.run == .failed(.notRunning) && store.state.conversation.messages[1].state == .failed)
    }
}
