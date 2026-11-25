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
    public let orchestrator: any ModelOrchestrating

    public init(_ orchestrator: any ModelOrchestrating) { self.orchestrator = orchestrator }

    public func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        let response = try await orchestrator.generate(request: request)
        return AnswerReply(text: response.generatedText, model: await orchestrator.modelInfo, tokens: response.promptTokens + response.completionTokens)
    }
}

public struct OnDemandAnswer: AnswerTarget {
    public let held: OnDemandModel

    public init(_ held: OnDemandModel) { self.held = held }

    public func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        let epoch = try await held.acquireGrant().epoch
        do throws(BasaltError) {
            let reply = try await LocalAnswer(held.model).answer(request)
            await held.release(epoch: epoch)
            return reply
        } catch {
            await held.release(epoch: epoch)
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
    public let defaultAdapter: String?

    public init(routes: [AnswerRoute], targets: [String: any AnswerTarget], fallback: any AnswerTarget, defaultAdapter: String? = nil) {
        (self.routes, self.targets, self.fallback, self.defaultAdapter) = (routes, targets, fallback, defaultAdapter)
    }

    public func costLeft() async -> Set<String> {
        var left: Set<String> = []
        for (name, target) in targets where await target.unavailable() == nil { left.insert(name) }
        return left
    }

    public func answer(_ request: GenerationRequest, situation: AnswerSituation) async throws(BasaltError) -> Routed {
        let choice = AnswerRouting.choose(routes, situation)
        var skipped: [String] = []
        for step in choice?.steps ?? [] {
            let name = step.model
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
                let adapted = step.adapter != nil ? request.withAdapter(step.adapter) : request
                return Routed(reply: try await target.answer(adapted), rule: choice?.rule, skipped: skipped)
            } catch {
                if Task.isCancelled { throw .cancelled }
                skipped.append("\(name): \(error.localizedDescription)")
            }
        }
        let fallbackRequest = defaultAdapter != nil && request.adapter == nil ? request.withAdapter(defaultAdapter) : request
        return Routed(reply: try await fallback.answer(fallbackRequest), rule: choice?.rule, skipped: skipped)
    }
}
