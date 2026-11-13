import Testing
@testable import BasaltAppCore
@testable import BasaltCore

@Suite("Chat scroll-pin reducer and unread streaming delta counter (#475)")
struct ScrollPinReducerTests {
    @Test("Scrolling away detaches the transcript, counts streaming content deltas, and resets on jumpToBottom or send")
    func detachCountAndReattach() {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/model"))
        state.draft = "Explain scroll pockets"
        #expect(state.scrollPin == .pinned && state.scrollPin.isPinned && !state.scrollPin.showsJumpControl)

        let run = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: run))
        #expect(state.scrollPin == .pinned)

        _ = ChatReducer.reduce(&state, .delta(run, .text("First ")))
        #expect(state.scrollPin == .pinned && state.scrollPin.unreadDeltas == 0)

        _ = ChatReducer.reduce(&state, .scrolled(atBottom: false))
        #expect(state.scrollPin == .detached(unreadDeltas: 0) && state.scrollPin.showsJumpControl)

        _ = ChatReducer.reduce(&state, .delta(run, .text("second ")))
        _ = ChatReducer.reduce(&state, .delta(run, .reasoning("step")))
        _ = ChatReducer.reduce(&state, .delta(run, .usage(prompt: 10, completion: 4)))
        _ = ChatReducer.reduce(&state, .delta(run, .finish("stop")))
        #expect(state.scrollPin == .detached(unreadDeltas: 2))

        _ = ChatReducer.reduce(&state, .scrolled(atBottom: false))
        #expect(state.scrollPin == .detached(unreadDeltas: 2))

        _ = ChatReducer.reduce(&state, .jumpToBottom)
        #expect(state.scrollPin == .pinned && !state.scrollPin.showsJumpControl && state.scrollPin.unreadDeltas == 0)

        _ = ChatReducer.reduce(&state, .scrolled(atBottom: false))
        _ = ChatReducer.reduce(&state, .delta(run, .text("third")))
        #expect(state.scrollPin == .detached(unreadDeltas: 1))

        _ = ChatReducer.reduce(&state, .scrolled(atBottom: true))
        #expect(state.scrollPin == .pinned)
    }

    @Test("Sending a new prompt or regenerating re-pins a detached scroll state")
    func sendAndRegenerateResetPin() {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/model"))
        state.draft = "First prompt"
        let firstRun = RunID()
        _ = ChatReducer.reduce(&state, .send(message: MessageID(), reply: MessageID(), run: firstRun))
        _ = ChatReducer.reduce(&state, .scrolled(atBottom: false))
        _ = ChatReducer.reduce(&state, .delta(firstRun, .text("Reply")))
        _ = ChatReducer.reduce(&state, .finished(firstRun, RunTiming(timeToFirstToken: .milliseconds(50), streaming: .milliseconds(200))))
        #expect(state.scrollPin == .detached(unreadDeltas: 1))

        let regenRun = RunID()
        _ = ChatReducer.reduce(&state, .regenerate(reply: MessageID(), run: regenRun))
        #expect(state.scrollPin == .pinned)
    }
}
