import Foundation
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore
@testable import BasaltCore
@testable import BasaltFoundationModels
@testable import BasaltSpeech

@Suite("AppCore, FoundationModels, and Speech defect regressions (A-A2, A-A4, A-A5, A-A6, A-A8, A-A9, A-A12)")
struct AppCoreFMSpeechDefectRegressionTests {
    @Test("A-A2: BasaltFoundationRequest.openAIJSONBody emits reasoning_effort=none when enableThinking is false")
    func foundationRequestDisablesThinkingViaReasoningEffort() throws {
        var req = BasaltFoundationRequest(messages: [ChatMessageInput(role: .user, text: "hi")])
        req.enableThinking = false
        let obj = try #require(JSONSerialization.jsonObject(with: try req.openAIJSONBody()) as? [String: Any])
        #expect(obj["enable_thinking"] as? Bool == false)
        #expect(obj["reasoning_effort"] as? String == "none")

        req.enableThinking = true
        let enabledObj = try #require(JSONSerialization.jsonObject(with: try req.openAIJSONBody()) as? [String: Any])
        #expect(enabledObj["reasoning_effort"] == nil)
    }

    @Test("A-A8: BasaltFoundationTransport throws transport error on truncated SSE stream missing [DONE]")
    func foundationTransportRejectsTruncatedSSEWithoutDone() async throws {
        let truncated = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "text/event-stream"], Data(truncated.utf8)) }
        let url = try #require(URL(string: "http://\(host)"))
        let endpoint = EngineEndpoint(base: url, token: nil)
        let session = URLSession(configuration: StubURLProtocol.configuration)
        let transport = BasaltFoundationTransport.rest(endpoint: endpoint, session: session)
        let req = BasaltFoundationRequest(messages: [ChatMessageInput(role: .user, text: "hi")])
        await #expect(throws: EngineError.self) {
            _ = try await transport.collect(req)
        }
    }

    @Test("A-A9: ChatDeltaDecoder preserves tool_calls[].index across interleaved deltas and ChatReducer merges by id")
    func interleavedToolCallDeltasMergeByIndexAndId() throws {
        var map: [Int: String] = [:]
        let events = [
            SSEEvent(data: #"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"first","arguments":"{\"x\":"}}]}}]}"#),
            SSEEvent(data: #"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"second","arguments":"{\"y\":"}}]}}]}"#),
            SSEEvent(data: #"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#),
            SSEEvent(data: #"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}"#),
        ]
        var msg = MessageSnapshot(id: MessageID(), role: .assistant, parts: [])
        for event in events {
            if case .deltas(let deltas) = try ChatDeltaDecoder.decode(event, toolCallIDs: &map) {
                for delta in deltas { ChatReducer.apply(delta, to: &msg) }
            }
        }
        #expect(msg.parts == [
            .toolCall(ToolCallOutput(id: "call_a", name: "first", arguments: "{\"x\":1}")),
            .toolCall(ToolCallOutput(id: "call_b", name: "second", arguments: "{\"y\":2}")),
        ])
    }

    @Test("A-A5: VoiceReducer switching read-aloud emits only .synthesize without racing .stopPlayback")
    func voiceReducerSwitchReadAloudDoesNotRaceStopPlayback() {
        let msgA = MessageID()
        let msgB = MessageID()
        var state = VoiceSessionState()
        state.speakingMessage = msgA
        let effects = VoiceReducer.reduce(&state, .readAloud(msgB, "Second message"))
        #expect(effects == [.synthesize(msgB, text: "Second message", voice: state.voice, speed: state.speed)])
        #expect(state.synthesizingMessage == msgB && state.speakingMessage == nil)
    }

    @Test("A-A6: LibraryReducer retries fit on next listing after transient fitFailed")
    func libraryReducerRetriesFitAfterTransientFailure() {
        var state = LibraryState()
        let m = LibraryModel(id: "org/m", sizeBytes: 100, isLoaded: false)
        let first = LibraryReducer.reduce(&state, .listed([m]))
        #expect(first == [.fit("org/m")])
        #expect(state.fitAttempted.contains("org/m"))

        _ = LibraryReducer.reduce(&state, .fitFailed("org/m", "temporary error"))
        #expect(!state.fitAttempted.contains("org/m"))
        #expect(!state.canLoad("org/m"))

        let retry = LibraryReducer.reduce(&state, .listed([m]))
        #expect(retry == [.fit("org/m")])
    }

    private actor SleepHoldGate {
        private var entered = 0
        private var enterWaiter: CheckedContinuation<Void, Never>?
        private var resumeWaiter: CheckedContinuation<Void, Never>?
        private var released = false

        var count: Int { entered }

        func hold() async {
            entered += 1
            if let waiter = enterWaiter {
                enterWaiter = nil
                waiter.resume()
            }
            guard !released else { return }
            await withCheckedContinuation { resumeWaiter = $0 }
        }

        func waitForEntry() async {
            guard entered == 0 else { return }
            await withCheckedContinuation { enterWaiter = $0 }
        }

        func release() {
            released = true
            let waiter = resumeWaiter
            resumeWaiter = nil
            waiter?.resume()
        }
    }

    @Test("A-A4: EngineSupervisor concurrent relaunch during stopAdopted finishes cleanup before spawning and never leaves engine idle", .timeLimit(.minutes(1)))
    func engineSupervisorConcurrentStopAdoptedAndRelaunch() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sup-adopt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = EngineRecords.file(dir.appendingPathComponent("engine.json"))
        store.save(EngineRecord(pid: 999_999, port: 18_999, token: "tok", arguments: ["serve"]))

        let alive = Mutex(true)
        let sleepGate = SleepHoldGate()
        let joined = AsyncStream.makeStream(of: Void.self)
        let initial = HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/true"), arguments: ["serve"])
        let next = HelperLaunch(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "exec sleep 30"])
        let env = EngineEnvironment(
            now: { .seconds(0) },
            sleep: { _ in
                await sleepGate.hold()
                alive.withLock { $0 = false }
            },
            probe: { _ in true },
            records: store,
            isAlive: { _ in alive.withLock { $0 } }
        )
        let supervisor = EngineSupervisor(launch: initial, environment: env, token: "tok")
        await supervisor.setOnAdoptedStopWait { joined.continuation.yield() }
        await supervisor.start()
        #expect(await supervisor.state.port == 18_999)

        let stopTask = Task { await supervisor.shutdown() }
        await sleepGate.waitForEntry()
        let relaunchTask = Task { await supervisor.relaunch(next) }
        var joinedIter = joined.stream.makeAsyncIterator()
        _ = await joinedIter.next()
        await sleepGate.release()
        _ = await (stopTask.value, relaunchTask.value)

        #expect(await supervisor.launch == next)
        #expect(await supervisor.state.phase == .spawning)
        #expect(await sleepGate.count == 1)
        await supervisor.shutdown()
    }

    @Test("A-A12: AudioCaptureActor.encodeMono16kWAV returns empty Data for zero samples and AppleSpeechSynthesizer rejects 0 Hz / overflow")
    func emptyRecordingAndInvalidSampleRateGuards() {
        #expect(AudioCaptureActor.encodeMono16kWAV(samples: [], sourceRate: 48_000).isEmpty)
        #expect(throws: BasaltError.self) {
            _ = try AppleSpeechSynthesizer.validatedConversionCapacity(frameLength: 1024, sourceRate: 0, targetRate: 24_000)
        }
        #expect(throws: BasaltError.self) {
            _ = try AppleSpeechSynthesizer.validatedConversionCapacity(frameLength: 1024, sourceRate: .nan, targetRate: 24_000)
        }
        #expect(throws: BasaltError.self) {
            _ = try AppleSpeechSynthesizer.validatedConversionCapacity(frameLength: 1024, sourceRate: 1e-12, targetRate: 24_000)
        }
    }
}
