import BasaltCore
import Foundation
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app phase 4: what Shortcuts, Services and the Dock run — ask, summarize, transcribe, switch model, start a conversation (#434 phase 4a)", .timeLimit(.minutes(1)))
struct AppIntentActionsTests {
    @Test("Ask, summarize and transcribe run against the live REST server; transcribe refuses a file that is not audio")
    func live() async throws {
        let engine = MockInferenceEngine()
        await engine.setMockTranscription(Transcription(text: "pay it today"))
        await engine.setResponder { _ in "Pay it today." }
        let audio = try #require(ChatAttachment(name: "memo.wav", data: AudioFixture.sine(seconds: 0.2)))
        try await AppEngineClientTests.serve(engine) { endpoint in
            let client = EngineClient.http(endpoint)
            let answer = try await IntentActions.ask("When do I pay?", model: nil, engine: client)
            let summary = try await IntentActions.summarize(audio, engine: client)
            let transcript = try await IntentActions.transcribe(audio, engine: client)
            #expect(answer == "Pay it today." && summary == "Pay it today." && transcript == "pay it today")
            await #expect(throws: EngineError.self) { _ = try await IntentActions.transcribe(try #require(ChatAttachment(name: "a.pdf", data: Data([1]))), engine: client) }
        }
    }

    @Test("Switch model refuses a bad id and a model that does not fit, and loads one that fits")
    func switchModel() async throws {
        let loaded = Mutex<[String]>([])
        let library = LibraryClient(
            models: { [] },
            fit: { id in ModelFit(fits: id != "org/big", level: "", requiredBytes: 0, availableBytes: 0, warning: id == "org/big" ? "needs 40 GB" : nil) },
            load: { id in loaded.withLock { $0.append(id) } }, download: { _, _ in }, delete: { _ in })
        await #expect(throws: EngineError.self) { _ = try await IntentActions.switchModel("../x", library: library) }
        let refusal = await #expect(throws: EngineError.self) { _ = try await IntentActions.switchModel("org/big", library: library) }
        #expect("\(String(describing: refusal))".contains("needs 40 GB"))
        #expect(try await IntentActions.switchModel("org/small", library: library) == "org/small" && loaded.withLock { $0 } == ["org/small"])
    }

    @MainActor
    @Test("A service or Dock drop starts a selected conversation with its text and files; without an engine the ready endpoint times out")
    func startConversation() async throws {
        let model = try AppModel(launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []), model: "org/model", storeURL: nil,
                                 drafts: DraftStore(load: { _ in "" }, save: { _, _ in }))
        let file = try #require(ChatAttachment(name: "bill.pdf", data: Data([1])))
        model.startConversation(text: "What is the total?", attachments: [file])
        let id = try #require(model.selection)
        let store = try #require(model.chats[id])
        await store.settle()
        #expect(store.state.conversation.messages.first?.text == "What is the total?" && store.state.conversation.messages.first?.attachments.first?.name == "bill.pdf")
        #expect(await model.readyEndpoint(within: .milliseconds(300)) == nil)
    }
}
