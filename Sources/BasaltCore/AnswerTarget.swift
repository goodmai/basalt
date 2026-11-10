import Foundation

public struct AnswerReply: Sendable, Equatable {
    public let text: String
    public let model: String
    public let tokens: Int
    public let cost: Double?

    public init(text: String, model: String, tokens: Int, cost: Double? = nil) {
        (self.text, self.model, self.tokens, self.cost) = (text, model, tokens, cost)
    }
}

public protocol AnswerTarget: Sendable {
    var isCloud: Bool { get }
    func unavailable() async -> String?
    func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply
}

extension AnswerTarget {
    public var isCloud: Bool { false }
    public func unavailable() async -> String? { nil }
}

public struct LocalAnswer: AnswerTarget {
    public let orchestrator: ModelOrchestratorActor

    public init(_ orchestrator: ModelOrchestratorActor) { self.orchestrator = orchestrator }

    public func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        let response = try await orchestrator.generate(request: request)
        return AnswerReply(text: response.generatedText, model: await orchestrator.modelInfo, tokens: response.promptTokens + response.completionTokens)
    }
}

public struct OnDemandAnswer: AnswerTarget {
    public let held: OnDemandModel

    public init(_ held: OnDemandModel) { self.held = held }

    public func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        _ = try await held.acquire()
        do throws(BasaltError) {
            let reply = try await LocalAnswer(held.model).answer(request)
            await held.release()
            return reply
        } catch {
            await held.release()
            throw error
        }
    }
}

public struct AnswerRouter: Sendable {
    public struct Routed: Sendable, Equatable {
        public let reply: AnswerReply
        public let rule: Int?
        public let skipped: [String]
    }

    public let routes: [AnswerRoute]
    public let targets: [String: any AnswerTarget]
    public let fallback: any AnswerTarget

    public init(routes: [AnswerRoute], targets: [String: any AnswerTarget], fallback: any AnswerTarget) {
        (self.routes, self.targets, self.fallback) = (routes, targets, fallback)
    }

    public func costLeft() async -> Set<String> {
        var left: Set<String> = []
        for (name, target) in targets where await target.unavailable() == nil { left.insert(name) }
        return left
    }

    public func answer(_ request: GenerationRequest, situation: AnswerSituation) async throws(BasaltError) -> Routed {
        let choice = AnswerRouting.choose(routes, situation)
        var skipped: [String] = []
        for name in choice?.chain ?? [] {
            guard let target = targets[name] else { continue }
            if target.isCloud, situation.perceived {
                skipped.append("\(name): documents and transcripts stay local")
                continue
            }
            if let reason = await target.unavailable() {
                skipped.append("\(name): \(reason)")
                continue
            }
            do {
                return Routed(reply: try await target.answer(request), rule: choice?.rule, skipped: skipped)
            } catch {
                if Task.isCancelled { throw .cancelled }
                skipped.append("\(name): \(error.localizedDescription)")
            }
        }
        return Routed(reply: try await fallback.answer(request), rule: choice?.rule, skipped: skipped)
    }
}
