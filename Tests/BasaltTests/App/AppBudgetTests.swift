import AppKit
import BasaltCore
import Foundation
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app budgets from #434: streaming into a 2,000-message transcript at 60 fps, quick ask to the first token in under 150 ms over the engine (set BASALT_APP_BUDGETS=1)",
       .enabled(if: ProcessInfo.processInfo.environment["BASALT_APP_BUDGETS"] == "1", "timing budgets: run on a quiet Mac with BASALT_APP_BUDGETS=1"), .serialized)
@MainActor
struct AppBudgetTests {
    static func transcript(_ count: Int) -> ConversationSnapshot {
        var conversation = ConversationSnapshot(id: ConversationID())
        let paragraph = String(repeating: "Basalt runs local models on Apple silicon and keeps the receipts. ", count: 4)
        conversation.messages = (0 ..< count).map { MessageSnapshot(id: MessageID(), role: $0.isMultiple(of: 2) ? .user : .assistant, parts: [.text(paragraph)]) }
        return conversation
    }

    static func host(_ view: some View, in window: NSWindow) -> NSView {
        let host = NSHostingView(rootView: view)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFrontRegardless()
        Self.draw(host)
        return host
    }

    static func draw(_ view: NSView) {
        RunLoop.main.run(until: Date())
        view.layoutSubtreeIfNeeded()
        view.display()
    }

    static func percentile(_ values: [Duration], _ fraction: Double) -> Duration {
        values.sorted()[min(values.count - 1, Int(Double(values.count) * fraction))]
    }

    static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
        return status == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    @Test("Each of 300 streamed tokens into a 2,000-message transcript is applied and drawn within one 60 fps frame at the 95th percentile; the open transcript adds under 100 MB")
    func streaming() throws {
        let before = Self.footprint()
        let store = ChatStore(ChatState(conversation: Self.transcript(2000)), reduce: ChatReducer.reduce, handle: { _ in .none })
        let run = RunID()
        store.send(.draftChanged("One more?"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: run))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 700), styleMask: [.titled], backing: .buffered, defer: false)
        let host = Self.host(ChatView(store: store), in: window)
        var frames: [Duration] = []
        for index in 0 ..< 300 {
            let start = ContinuousClock.now
            store.send(.delta(run, .text("token\(index) ")))
            Self.draw(host)
            frames.append(ContinuousClock.now - start)
        }
        let added = Double(Self.footprint() - before) / 1_048_576
        window.close()
        let (median, p95) = (Self.percentile(frames, 0.5), Self.percentile(frames, 0.95))
        print("BUDGET streaming median=\(median) p95=\(p95) max=\(frames.max() ?? .zero) transcript_mb=\(String(format: "%.1f", added))")
        #expect(store.state.conversation.messages.last?.text.hasSuffix("token299 ") == true)
        #expect(p95 < .microseconds(16_700), "p95 \(p95)")
        #expect(added < 100, "the open transcript added \(added) MB")
    }

    @Test("Opening the quick ask panel and getting the first token from the live server adds under 150 ms to the engine's own time once the app is warm")
    func quickAsk() async throws {
        let engine = MockInferenceEngine()
        await engine.setResponder { _ in "Basalt is volcanic." }
        let (cold, timing) = try await AppEngineClientTests.serve(engine) { endpoint in
            let session = URLSession.isolated
            return (try await Self.ask(endpoint, session), try await Self.ask(endpoint, session))
        }
        print("BUDGET quick-ask cold panel=\(cold.panel) first-token=\(cold.answer); warm panel=\(timing.panel) first-token=\(timing.answer)")
        #expect(timing.answer < .milliseconds(150), "panel \(timing.panel), first token \(timing.answer)")
    }

    static func ask(_ endpoint: EngineEndpoint, _ session: URLSession) async throws -> (panel: Duration, answer: Duration) {
        let start = ContinuousClock.now
        let store = ChatStore(ChatState(conversation: ConversationSnapshot(id: ConversationID())), reduce: ChatReducer.reduce,
                              handle: ChatEffects.handler(ChatEnvironment(engine: { EngineClient.http(endpoint, session: session) }, persist: { _ in })))
        let window = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 620, height: 420), styleMask: [.nonactivatingPanel, .titled], backing: .buffered, defer: true)
        let host = Self.host(QuickAskView(store: store) {}, in: window)
        let panel = ContinuousClock.now - start
        store.send(.draftChanged("What is basalt?"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        Self.draw(host)
        let answer = ContinuousClock.now - start
        window.close()
        #expect(store.state.conversation.messages.last?.text == "Basalt is volcanic.", "\(store.state.run)")
        return (panel, answer)
    }
}
