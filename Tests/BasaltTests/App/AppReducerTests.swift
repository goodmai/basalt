import Foundation
import Testing
@testable import BasaltAppCore
@testable import BasaltCore

@Suite("Mac app core: the chat and engine reducers, without a server, a clock or a view (#434 phase 1)")
struct AppReducerTests {
    static func chat(draft: String = "Hi") -> ChatState {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/model"))
        state.draft = draft
        return state
    }

    static func started(_ effects: [ChatEffect], _ run: RunID) -> ChatRequest? {
        guard effects.count == 2, case .persist = effects[0], case .start(let id, let request) = effects[1], id == run else { return nil }
        return request
    }

    @Test("Send appends the user turn and an empty streaming answer, titles the conversation, and asks to persist and start with the turns so far")
    func send() {
        var state = Self.chat(draft: "  What is basalt?  ")
        let (message, reply, run) = (MessageID(), MessageID(), RunID())
        let effects = ChatReducer.reduce(&state, .send(message: message, reply: reply, run: run))
        #expect(state.conversation.messages.map(\.id) == [message, reply] && state.conversation.messages[1].state == .streaming)
        #expect(state.conversation.title == "What is basalt?" && state.draft.isEmpty && state.run == .streaming(run) && !state.canSend)
        let request = Self.started(effects, run)
        #expect(request?.messages == [.init(role: .user, content: "What is basalt?")] && request?.model == "org/model")
        state.draft = "again"
        #expect(ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: RunID())).isEmpty)
        var blank = Self.chat(draft: " \n ")
        #expect(ChatReducer.reduce(&blank, .send(message: message, reply: reply, run: run)).isEmpty && blank.conversation.messages.isEmpty)
    }

    @Test("Deltas build parts in order; usage and timing fill the receipt; finish completes the answer and persists")
    func streaming() {
        var state = Self.chat()
        let run = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: run))
        let call = ToolCallOutput(id: "c", name: "f", arguments: "{}")
        for delta in [ChatDelta.reasoning("th"), .reasoning("ink"), .text("Hel"), .text("lo"), .toolCall(call), .finish("stop"), .usage(prompt: 7, completion: 20)] {
            #expect(ChatReducer.reduce(&state, .delta(run, delta)).isEmpty)
        }
        #expect(state.conversation.messages[1].parts == [.reasoning("think"), .text("Hello"), .toolCall(call)])
        let effects = ChatReducer.reduce(&state, .finished(run, RunTiming(timeToFirstToken: .milliseconds(300), streaming: .seconds(2))))
        let answer = state.conversation.messages[1]
        #expect(answer.state == .complete && state.run == .idle && effects == [.persist(state.conversation)])
        #expect(answer.receipt?.promptTokens == 7 && answer.receipt?.tokensPerSecond == 10 && answer.receipt?.timeToFirstToken == .milliseconds(300))
    }

    @Test("Stop keeps the partial answer as stopped and cancels the run; deltas and a finish of that run arriving later change nothing")
    func stopAndStale() {
        var state = Self.chat()
        let run = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: run))
        _ = ChatReducer.reduce(&state, .delta(run, .text("Par")))
        #expect(ChatReducer.reduce(&state, .stop) == [.cancel(run), .persist(state.conversation)])
        #expect(state.conversation.messages[1].state == .stopped && state.run == .idle)
        let frozen = state
        #expect(ChatReducer.reduce(&state, .delta(run, .text("tial"))).isEmpty)
        #expect(ChatReducer.reduce(&state, .finished(run, RunTiming(timeToFirstToken: nil, streaming: .seconds(1)))).isEmpty)
        #expect(ChatReducer.reduce(&state, .failed(run, .transport("reset"))).isEmpty && state == frozen)
        #expect(ChatReducer.reduce(&state, .stop).isEmpty)
        state.draft = "next"
        let next = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: next))
        #expect(ChatReducer.reduce(&state, .delta(run, .text("old"))).isEmpty && state.conversation.messages.last?.parts.isEmpty == true)
    }

    @Test("A failure marks the answer failed and keeps the error; the next send is allowed; regenerate replaces the last answer")
    func failureAndRegenerate() {
        var state = Self.chat()
        let run = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: run))
        _ = ChatReducer.reduce(&state, .failed(run, .notRunning))
        #expect(state.run == .failed(.notRunning) && state.conversation.messages[1].state == .failed)
        let (reply, again) = (MessageID(), RunID())
        let effects = ChatReducer.reduce(&state, .regenerate(reply: reply, run: again))
        #expect(state.conversation.messages.count == 2 && state.conversation.messages[1].id == reply && state.run == .streaming(again))
        #expect(Self.started(effects, again)?.messages.count == 1)
        #expect(ChatReducer.reduce(&state, .regenerate(reply: MessageID(), run: RunID())).isEmpty)
    }

    @Test("Engine: start spawns, the handshake writes the record and probes, ready follows; stop kills that epoch and ignores its late exit")
    func engineLifecycle() {
        var state = EngineState()
        #expect(EngineReducer.reduce(&state, .start) == [.spawn(epoch: 1)] && state.phase == .spawning)
        #expect(EngineReducer.reduce(&state, .start).isEmpty)
        #expect(EngineReducer.reduce(&state, .handshake(epoch: 1, port: 5000)) == [.writeRecord(port: 5000), .probe(epoch: 1, port: 5000)])
        #expect(EngineReducer.reduce(&state, .ready(epoch: 1)).isEmpty && state.phase == .ready(port: 5000) && state.port == 5000)
        #expect(EngineReducer.reduce(&state, .stop) == [.kill(epoch: 1), .removeRecord] && state.phase == .stopped && state.port == nil)
        #expect(EngineReducer.reduce(&state, .exited(epoch: 1, at: .seconds(5), log: [])).isEmpty && state.phase == .stopped)
        #expect(EngineReducer.reduce(&state, .stop).isEmpty)
    }

    @Test("Engine: a crash backs off 1 s then 2 s and respawns under a new epoch; the third crash within a minute fails with the log; old epochs are ignored")
    func engineCrashes() {
        var state = EngineState()
        _ = EngineReducer.reduce(&state, .start)
        #expect(EngineReducer.reduce(&state, .exited(epoch: 1, at: .seconds(10), log: ["a"])) == [.removeRecord, .schedule(epoch: 1, after: .seconds(1))])
        #expect(state.phase == .backingOff(attempt: 1))
        #expect(EngineReducer.reduce(&state, .retry(epoch: 1)) == [.spawn(epoch: 2)])
        #expect(EngineReducer.reduce(&state, .handshake(epoch: 1, port: 1)).isEmpty && EngineReducer.reduce(&state, .retry(epoch: 1)).isEmpty)
        #expect(EngineReducer.reduce(&state, .exited(epoch: 2, at: .seconds(20), log: ["b"])) == [.removeRecord, .schedule(epoch: 2, after: .seconds(2))])
        _ = EngineReducer.reduce(&state, .retry(epoch: 2))
        #expect(EngineReducer.reduce(&state, .exited(epoch: 3, at: .seconds(30), log: ["boom"])) == [.removeRecord] && state.phase == .failed(log: ["boom"]))
        #expect(EngineReducer.reduce(&state, .start) == [.spawn(epoch: 4)])
        #expect(EngineReducer.reduce(&state, .exited(epoch: 4, at: .seconds(200), log: [])) == [.removeRecord, .schedule(epoch: 4, after: .seconds(1))])
        #expect(EngineReducer.backoff(attempt: 6) == .seconds(30) && EngineReducer.backoff(attempt: 40) == .seconds(30))
    }
}
