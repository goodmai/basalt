import Testing
import Foundation
import Synchronization
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Priority admission and cancellable prefill (#140)", .serialized)
struct PriorityAndPrefillCancellationTests {
    actor Order {
        var admitted: [String] = []
        func add(_ name: String) { admitted.append(name) }
    }

    @Test("Waiters are admitted by priority, first come first served within a level")
    func priorityOrder() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 8)
        let holder = try await gate.admit()
        let order = Order()
        var tasks: [Task<Void, Error>] = []
        for (name, priority) in [("std1", RequestPriority.standard), ("bg", .background), ("std2", .standard),
                                 ("tool", .toolValidation), ("ui", .interactive)] {
            tasks.append(Task {
                let ticket = try await gate.admit(priority: priority)
                await order.add(name)
                await gate.release(ticket)
            })
            #expect(await GenerationGateTests.eventually { await gate.queued == tasks.count })
        }
        await gate.release(holder)
        for task in tasks { try await task.value }
        #expect(await order.admitted == ["ui", "tool", "std1", "std2", "bg"])
    }

    @Test("A full queue lets a higher-priority request in by turning away the lowest-priority waiter")
    func preemptsTheLowestWaiter() async throws {
        let gate = GenerationGate(maxConcurrent: 1, maxQueue: 2)
        let holder = try await gate.admit()
        let background = Task { try await gate.admit(priority: .background) }
        #expect(await GenerationGateTests.eventually { await gate.queued == 1 })
        let standard = Task { try await gate.admit(priority: .standard) }
        #expect(await GenerationGateTests.eventually { await gate.queued == 2 })
        #expect(await gate.refusal(for: .background) != nil)
        #expect(await gate.refusal(for: .interactive) == nil)
        let interactive = Task { try await gate.admit(priority: .interactive) }
        do {
            _ = try await background.value
            Issue.record("the background waiter should be turned away")
        } catch {
            guard case .serverBusy(let retry) = error as? BasaltError else { Issue.record("unexpected \(error)"); return }
            #expect(retry == 1)
        }
        await gate.release(holder)
        await gate.release(try await interactive.value)
        await gate.release(try await standard.value)
        #expect(await gate.inFlight == 0)
        await #expect(throws: BasaltError.self) {
            let full = GenerationGate(maxConcurrent: 1, maxQueue: 0)
            _ = try await full.admit()
            _ = try await full.admit(priority: .interactive)
        }
    }

    @Test("Requests get their priority from where they come from")
    func priorityFromOrigin() {
        func priority(_ source: String, service: Bool = false) -> RequestPriority {
            GenerationRequest(prompt: "p", isServiceRequest: service)
                .withOrigin(RequestOrigin(source: source, endpoint: "", stream: false)).priority
        }
        #expect(priority("cli") == .interactive)
        #expect(priority("harness") == .toolValidation)
        #expect(priority("anthropic", service: true) == .toolValidation)
        #expect(priority("benchmark") == .background)
        #expect(priority("openai-chat") == .standard)
        #expect(GenerationRequest(prompt: "p").priority == .standard)
    }

    static let longPrompt = String(repeating: "abcdefgh", count: 2_048)

    static func tinyEngine(cacheBlendRatio: Float? = nil) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(promptCacheBytes: cacheBlendRatio == nil ? 0 : 1 << 26, cacheBlendRatio: cacheBlendRatio)
        await engine.attach(try TinyModel.container())
        return engine
    }

    final class PrefillChunks: Sendable {
        private let reports = Mutex<[(processed: Int, total: Int)]>([])

        func record(_ processed: Int, _ total: Int) -> Int {
            reports.withLock {
                $0.append((processed, total))
                return $0.count
            }
        }

        var all: [(processed: Int, total: Int)] { reports.withLock { $0 } }
    }

    @Test("Cancelling a request mid-prefill of a 16k-token prompt stops it before the next prefill chunk, not at the end of the prompt",
          arguments: [nil, 0.5] as [Float?])
    func prefillStopsOnCancel(cacheBlendRatio: Float?) async throws {
        let engine = try await Self.tinyEngine(cacheBlendRatio: cacheBlendRatio)
        let request = GenerationRequest(prompt: Self.longPrompt, maxTokens: 4, temperature: 0, cacheKey: "cancelled")
        let chunks = PrefillChunks()
        let streamed = await MLXInferenceEngine.$prefillProgress.withValue({ processed, total in
            if chunks.record(processed, total) == 2 { withUnsafeCurrentTask { $0?.cancel() } }
        }) {
            await Task {
                var streamed = 0
                do {
                    for await _ in try await engine.generateStream(request: request) { streamed += 1 }
                } catch {}
                return streamed
            }.value
        }
        let seen = chunks.all
        let last = try #require(seen.last)
        #expect(streamed == 0)
        #expect(seen.count == 2, "\(seen.count) prefill chunks ran, 2 before the cancel")
        #expect(last.processed * 4 < last.total, "\(last.processed) of \(last.total) tokens prefilled")
    }

    @Test("Shutting the server down during a stream's prefill aborts the prefill instead of waiting for it", .timeLimit(.minutes(5)))
    func drainAbortsStreamPrefill() async throws {
        let engine = try await Self.tinyEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 4, telemetry: TelemetryTracker())
        let request = GenerationRequest(prompt: Self.longPrompt, maxTokens: 4, temperature: 0)
        let chunks = PrefillChunks()
        let (midPrefill, reachMidPrefill) = AsyncStream<Void>.makeStream()
        let streaming = MLXInferenceEngine.$prefillProgress.withValue({ processed, total in
            guard chunks.record(processed, total) == 2 else { return }
            reachMidPrefill.finish()
            while !Task.isCancelled { Thread.sleep(forTimeInterval: 0.001) }
        }) {
            Task { () -> Bool in
                do {
                    for await _ in try await orchestrator.generateStream(request: request) {}
                    return true
                } catch {
                    return false
                }
            }
        }
        for await _ in midPrefill {}
        await orchestrator.shutdown(deadline: .zero)
        #expect(await streaming.value == false)
        #expect(chunks.all.count == 2)
    }
}
