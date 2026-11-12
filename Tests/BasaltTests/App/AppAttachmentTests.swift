import BasaltCore
import Foundation
import Testing
@testable import BasaltAppCore

@Suite("Mac app attachments: images, PDFs and audio go through /v1/pipeline, the perceived text stays in the conversation for later turns (#434 phase 3b)", .timeLimit(.minutes(1)))
struct AppAttachmentTests {
    static func audio() throws -> ChatAttachment { try #require(ChatAttachment(name: "memo.wav", data: AudioFixture.sine(seconds: 0.2))) }

    @Test("Only known extensions within the size limit are accepted, by kind")
    func accepts() throws {
        #expect(ChatAttachment(name: "a.PDF", data: Data([1]))?.kind == "pdf" && ChatAttachment(name: "b.heic", data: Data([1]))?.kind == "image")
        #expect(ChatAttachment(name: "c.m4a", data: Data([1]))?.format == "m4a" && ChatAttachment.accepts("d.wav") && !ChatAttachment.accepts("e.exe"))
        #expect(ChatAttachment(name: "f.txt", data: Data([1])) == nil && ChatAttachment(name: "g.pdf", data: Data()) == nil)
        #expect(ChatAttachment(name: "h.pdf", data: Data(count: ChatAttachment.byteLimit + 1)) == nil)
    }

    @Test("Sending with attachments asks the pipeline; the perceived text and the answer fill the exchange, a stale run is ignored")
    func reducer() throws {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID(), model: "org/model"))
        let attachment = try Self.audio()
        _ = ChatReducer.reduce(&state, .attach(attachment))
        _ = ChatReducer.reduce(&state, .attach(attachment))
        #expect(state.attachments.count == 1 && state.canSend)
        let (message, reply, run) = (MessageID(), MessageID(), RunID())
        let effects = ChatReducer.reduce(&state, .send(message: message, reply: reply, run: run))
        #expect(effects.contains(.perceive(run, PipelineCall(attachments: [attachment], task: "Summarize the attached files."))))
        #expect(!effects.contains { if case .start = $0 { true } else { false } } && state.attachments.isEmpty && state.conversation.title == "memo.wav")
        #expect(state.conversation.messages.first?.attachments == [AttachmentPart(name: "memo.wav", kind: "audio")])
        var receipt = Receipt(model: "org/model")
        receipt.stages = [StageReceipt(stage: "perceive", model: "org/model", ms: 12)]
        let result = PipelineResult(perceived: ["pay it today"], answer: "Pay today.", receipt: receipt)
        #expect(ChatReducer.reduce(&state, .perceived(RunID(), result)).isEmpty)
        #expect(ChatReducer.reduce(&state, .perceived(run, result)) == [.persist(state.conversation), .answered(state.conversation)])
        #expect(state.conversation.messages[0].attachments.first?.perceived == "pay it today" && state.conversation.messages[1].text == "Pay today.")
        #expect(state.conversation.messages[1].state == .complete && state.conversation.messages[1].receipt?.stages?.count == 1 && state.run == .idle)
        #expect(ChatRequest(state.conversation).messages.first?.content == "Attached audio memo.wav:\npay it today\n\n")
        _ = ChatReducer.reduce(&state, .attach(try Self.audio()))
        _ = ChatReducer.reduce(&state, .detach(state.attachments[0].id))
        #expect(state.attachments.isEmpty && !state.canSend)
    }

    @Test("The pipeline response maps pages to their attachment, the transcript to the audio, and stages to the receipt")
    func result() throws {
        let json = #"{"perceived":{"pages":[{"attachment":0,"page":1,"markdown":"Invoice"},{"attachment":0,"page":2,"markdown":"Total 42"}],"transcript":"pay it today"},"answers":{},"system2":{"model":"org/chat","text":"Pay 42 today."},"receipt":{"stages":[{"stage":"perceive","model":"apple-vision","ms":80,"tokens":0,"escalated":false},{"stage":"answer","model":"org/chat","ms":900,"tokens":7,"escalated":false}],"answered_by":{},"escalated":false,"system2_calls":1,"ms":990}}"#
        let response = try JSONDecoder().decode(PipelineResponse.self, from: Data(json.utf8))
        let result = PipelineResult(response, count: 2)
        #expect(result.perceived == ["Invoice\n\nTotal 42", "pay it today"] && result.answer == "Pay 42 today.")
        #expect(result.receipt.model == "org/chat" && result.receipt.completionTokens == 7 && result.receipt.stages?.map(\.stage) == ["perceive", "answer"])
        let body = try JSONSerialization.jsonObject(with: PipelineCall(attachments: [try Self.audio()], task: "Sum it up.").body) as? [String: Any]
        #expect(body?["task"] as? String == "Sum it up." && (body?["attachments"] as? [[String: String]])?.first?["kind"] == "audio")
    }

    @MainActor
    @Test("Against the live REST server an audio attachment is transcribed and answered through /v1/pipeline")
    func live() async throws {
        let engine = MockInferenceEngine()
        await engine.setMockTranscription(Transcription(text: "pay it today"))
        await engine.setResponder { _ in "Pay it today." }
        let attachment = try Self.audio()
        try await AppEngineClientTests.serve(engine) { endpoint in
            let store = await MainActor.run {
                Store(ChatState(conversation: ConversationSnapshot(id: ConversationID())), reduce: ChatReducer.reduce,
                      handle: ChatEffects.handler(ChatEnvironment(engine: { EngineClient.http(endpoint, session: .isolated) }, persist: { _ in })))
            }
            await MainActor.run {
                store.send(.attach(attachment))
                store.send(.draftChanged("When do I pay?"))
                store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
            }
            await store.settle()
            let messages = await MainActor.run { store.state.conversation.messages }
            #expect(messages.first?.attachments.first?.perceived == "pay it today")
            #expect(messages.last?.text == "Pay it today." && messages.last?.receipt?.stages?.isEmpty == false)
        }
    }
}
