import AppKit
import BasaltCore
import Foundation
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app phase 4b: the engine runs a model or a compose system, a running helper is adopted only with the same arguments, compose state and cloud spend in the Monitor, the cloud mark on receipts (#434 phase 4b)", .timeLimit(.minutes(1)))
struct AppEngineSystemsTests {
    typealias Fixture = EngineSupervisorTests.Fixture

    @Test("The engine's arguments name the model, or the compose file when one is chosen")
    func arguments() {
        #expect(EngineChoice.arguments(model: "org/m", compose: nil) == ["serve", "--model", "org/m"])
        #expect(EngineChoice.arguments(model: "org/m", compose: " ") == ["serve", "--model", "org/m"])
        #expect(EngineChoice.arguments(model: "org/m", compose: "/srv/desk.yaml") == ["serve", "--compose", "/srv/desk.yaml"])
    }

    @Test("A basalt://run deep link resolves verified Hugging Face model roles and rejects invalid repo IDs or flags")
    func launchURL() throws {
        func resolve(_ raw: String, fallback: String = EngineChoice.defaultChatModel) throws -> (model: String, arguments: [String])? {
            let url = try #require(URL(string: raw))
            return EngineChoice.launchURL(url, fallbackChatModel: fallback)
        }
        let chat = try #require(try resolve("basalt://run?model=goodmai/gemma-4-12B-it-qat-4bit"))
        #expect(chat.model == "goodmai/gemma-4-12B-it-qat-4bit" && chat.arguments == ["serve", "--model", "goodmai/gemma-4-12B-it-qat-4bit"])
        let mtp = try #require(try resolve("basalt://run?model=goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp"))
        #expect(mtp.arguments == ["serve", "--model", "goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed", "--mtp"])
        let stream = try #require(try resolve("basalt://run?model=goodmai/Qwen3.6-35B-A3B-8bit&role=moe-stream"))
        #expect(stream.arguments == ["serve", "--model", "goodmai/Qwen3.6-35B-A3B-8bit", "--experimental-expert-streaming"])
        let ocr = try #require(try resolve("basalt://run?model=goodmai/DeepSeek-OCR-2-8bit&role=ocr", fallback: "goodmai/Qwen3-0.6B-4bit"))
        #expect(ocr.model == "goodmai/Qwen3-0.6B-4bit" && ocr.arguments == ["serve", "--model", "goodmai/Qwen3-0.6B-4bit", "--pipeline-ocr-model", "goodmai/DeepSeek-OCR-2-8bit"])
        for (role, flag) in [("rerank", "--rerank-model"), ("embed", "--embedding-model"), ("asr", "--pipeline-audio-model"), ("decide", "--decision-model")] {
            let sidecar = try #require(try resolve("basalt://run?model=goodmai/laya&role=\(role)", fallback: "bad"))
            #expect(sidecar.model == EngineChoice.defaultChatModel && sidecar.arguments == ["serve", "--model", EngineChoice.defaultChatModel, flag, "goodmai/laya"])
        }
        for bad in [
            "https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit",
            "basalt://exec?model=goodmai/gemma-4-12B-it-qat-4bit",
            "basalt://run?model=../etc/passwd",
            "basalt://run?model=goodmai/gemma%20--unsafe",
            "basalt://run?model=goodmai/gemma-4-12B-it-qat-4bit&role=--unsafe",
        ] {
            #expect(try resolve(bad) == nil)
        }
    }

    @MainActor
    @Test("Cold launch with matching basalt:// arguments still starts the engine, and switching models updates configuredModel immediately")
    func coldLaunchAndSwitchModel() async throws {
        let fixture = try Fixture()
        let launch = try fixture.script(EngineSupervisorTests.ready)
        let app = try AppModel(launch: launch, model: "org/m", storeURL: nil, environment: fixture.environment(), drafts: DraftStore(load: { _ in "" }, save: { _, _ in }))
        await app.start(arguments: launch.arguments)
        #expect(await app.readyEndpoint(within: .seconds(5))?.base.port == 4242 && app.model == "org/m")
        await app.switchEngine(["serve", "--model", "goodmai/Qwen3.6-35B-A3B-8bit", "--experimental-expert-streaming"])
        #expect(app.configuredModel == "goodmai/Qwen3.6-35B-A3B-8bit" && app.model == "goodmai/Qwen3.6-35B-A3B-8bit")
        #expect(await app.readyEndpoint(within: .seconds(5))?.base.port == 4242)
        await app.stop()
    }

    @Test("A live helper started with other arguments is not adopted; relaunch stops the helper and starts it with the new arguments")
    func relaunch() async throws {
        let fixture = try Fixture()
        EngineRecords.file(fixture.recordURL).save(EngineRecord(pid: ProcessInfo.processInfo.processIdentifier, port: 5151, token: "kept", arguments: ["serve", "--model", "org/other"]))
        let supervisor = EngineSupervisor(launch: try fixture.script(EngineSupervisorTests.ready), environment: fixture.environment())
        await supervisor.start()
        #expect(await EngineSupervisorTests.until(supervisor) { $0 == .ready(port: 4242) } == .ready(port: 4242))
        let first = try #require(EngineRecords.file(fixture.recordURL).load())
        #expect(first.token != "kept" && first.arguments == ["serve", "--model", "org/m"])
        let current = await supervisor.launch
        await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: ["serve", "--compose", "/srv/desk.yaml"]))
        #expect(await EngineSupervisorTests.until(supervisor) { $0 == .ready(port: 4242) } == .ready(port: 4242))
        #expect(fixture.file("args")?.contains("serve --compose /srv/desk.yaml") == true && kill(first.pid, 0) != 0)
        await supervisor.shutdown()
    }

    @Test("Shutting down an adopted helper stops that process and removes its record")
    func stopsAdopted() async throws {
        let fixture = try Fixture()
        let helper = Process()
        (helper.executableURL, helper.arguments) = (URL(fileURLWithPath: "/bin/sleep"), ["30"])
        try helper.run()
        defer { helper.terminate() }
        EngineRecords.file(fixture.recordURL).save(EngineRecord(pid: helper.processIdentifier, port: 5151, token: "kept", arguments: ["serve", "--model", "org/m"]))
        let supervisor = EngineSupervisor(launch: try fixture.script("exit 1"), environment: fixture.environment())
        await supervisor.start()
        #expect(await supervisor.state.phase == .ready(port: 5151))
        await supervisor.shutdown()
        helper.waitUntilExit()
        let phase = await supervisor.state.phase
        #expect(!helper.isRunning && EngineRecords.file(fixture.recordURL).load() == nil && phase == .stopped)
    }

    @MainActor
    @Test("The compose state decodes, polls into the Monitor, renders, and a server without compose answers nil")
    func compose() async throws {
        let json = #"{"name":"desk","pipelines":["default","support"],"spent_today":0.25,"models":[{"name":"chat","source":"org/chat","keep":"resident","state":"ready"},{"name":"haiku","source":"cloud:openrouter/anthropic/claude-haiku-5.5","keep":"cloud","state":"over-budget","spent_today":0.25,"limit":0.25,"refusal":"daily budget spent"}]}"#
        let info = try JSONDecoder().decode(ComposeInfo.self, from: Data(json.utf8))
        #expect(info.models.map(\.isCloud) == [false, true] && info.spentToday == 0.25)
        let store = Store(MonitorState(), reduce: MonitorReducer.reduce, handle: MonitorReducer.handler(
            stats: { ServerTelemetrySnapshot() }, every: .seconds(1), sleep: { _ in throw CancellationError() }, compose: { info }))
        store.send(.appeared)
        await store.settle()
        #expect(store.state.compose == info)
        #expect(ComposeSection.detail(info.models[1]).contains("$0.2500 of $0.25") && ComposeSection.detail(info.models[1]).contains("daily budget spent"))
        let host = NSHostingView(rootView: MonitorView(store: store).frame(width: 400, height: 600))
        host.layoutSubtreeIfNeeded()
        let settings = NSHostingView(rootView: EngineSettings { _ in }.frame(width: 480, height: 300))
        settings.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width == 400 && settings.fittingSize.width == 480)
        let none = try await AppEngineClientTests.serve(MockInferenceEngine()) { endpoint in try await EngineClient.http(endpoint, session: .isolated).compose() }
        #expect(none == nil)
    }

    @Test("A stage with a cost is a cloud stage")
    func cloudMark() {
        #expect(StageReceipt(stage: "answer", model: "anthropic/claude-haiku-5.5", ms: 1, cost: 0.001).isCloud && !StageReceipt(stage: "answer", model: "org/chat", ms: 1).isCloud)
    }
}
