import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltUI
@testable import BasaltCLI

@Suite("Switches that list every case of basalt's own enums (#284)")
struct ExhaustiveSwitchTests {
    @Test("A model that fails to load for a reason other than a bad configuration leaves serve running in stub mode")
    func serveFallsBackToStubMode() async throws {
        let engine = MockInferenceEngine()
        await engine.setShouldFailOnLoad(true)
        let orchestrator = ModelOrchestratorActor(engine: engine)
        try await ServeCommand.parse([]).load(orchestrator, from: "missing")
        #expect(await orchestrator.currentModelId.isEmpty)
    }

    @Test("A research call that is not a lab call is refused by the lab dispatcher")
    func nonLabCallRefused() async {
        await #expect(throws: BasaltError.self) { try await ResearchCall.describe.run(on: MLXInferenceEngine() as any ModelLab) }
    }

    @Test("A clipboard tool without a copy or paste side has no command for it")
    func clipboardCommands() {
        #expect(ClipboardTool.pbpaste.copyCommand == nil && ClipboardTool.none.copyCommand == nil)
        #expect(ClipboardTool.pbcopy.pasteCommand == nil && ClipboardTool.none.pasteCommand == nil)
        #expect(ClipboardTool.xclip.copyCommand != nil && ClipboardTool.xsel.pasteCommand != nil)
    }
}
