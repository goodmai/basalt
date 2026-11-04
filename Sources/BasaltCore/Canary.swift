import Foundation

public struct CanarySchedule: Sendable {
    public enum Step: Equatable, Sendable {
        case run
        case wait(until: ContinuousClock.Instant)
    }

    public let interval: Duration
    public let retry: Duration
    public private(set) var due: ContinuousClock.Instant

    public init(interval: Duration, retry: Duration = .seconds(10), start: ContinuousClock.Instant) {
        self.interval = interval
        self.retry = retry
        due = start + interval
    }

    public mutating func step(at now: ContinuousClock.Instant, idle: Bool) -> Step {
        guard now >= due else { return .wait(until: due) }
        guard idle else { return .wait(until: now + retry) }
        due = now + interval
        return .run
    }
}

public enum Canary {
    public static let request: GenerationRequest = {
        var request = GenerationRequest(prompt: "What is 17 + 25? Reply with the number only.", maxTokens: 32, temperature: 0)
        request.enableThinking = false
        request.isServiceRequest = true
        return request.withOrigin(RequestOrigin(source: "canary", endpoint: "canary", stream: false))
    }()

    public static func passes(_ response: GenerationResponse) -> Bool {
        response.finishReason != .error && response.generatedText.contains("42")
    }

    public static func run(every interval: Duration, on orchestrator: ModelOrchestratorActor) async {
        let clock = ContinuousClock()
        var schedule = CanarySchedule(interval: interval, start: clock.now)
        while !Task.isCancelled {
            let idle = await orchestrator.gate.inFlight + orchestrator.gate.queued == 0
            switch schedule.step(at: clock.now, idle: idle) {
            case .wait(let until):
                guard (try? await clock.sleep(until: until)) != nil else { return }
            case .run:
                if let passed = await probe(orchestrator) { await orchestrator.telemetry.recordCanary(passed: passed) }
            }
        }
    }

    static func probe(_ orchestrator: ModelOrchestratorActor) async -> Bool? {
        let logger = BasaltLogger(module: "Canary")
        do throws(BasaltError) {
            let response = try await orchestrator.generate(request: request)
            guard passes(response) else {
                logger.warn("Canary failed: asked 17 + 25, got \(response.generatedText.prefix(80).debugDescription)")
                return false
            }
            return true
        } catch .serverBusy, .memoryPressure {
            return nil
        } catch {
            logger.warn("Canary failed: \(error.localizedDescription)")
            return false
        }
    }
}
