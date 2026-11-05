import Foundation

public struct IntrospectRequest: Sendable, Codable, Equatable {
    public static let steps = 1...32
    public static let budgets = 200...20_000

    public var goal: String
    public var maxSteps = 8
    public var allowInterventions = false
    public var resultBudget = 2_000

    public init(goal: String, maxSteps: Int = 8, allowInterventions: Bool = false) {
        self.goal = goal
        self.maxSteps = maxSteps
        self.allowInterventions = allowInterventions
    }
}

public struct IntrospectStep: Sendable, Codable, Equatable {
    public var text: String
    public var tool: String?
    public var arguments: String?
    public var result: String?

    public init(text: String, tool: String? = nil, arguments: String? = nil, result: String? = nil) {
        self.text = text
        self.tool = tool
        self.arguments = arguments
        self.result = result
    }
}

public struct Hypothesis: Sendable, Codable, Equatable {
    public var statement: String
    public var test = ""
    public var outcome = ""

    public init(statement: String, test: String = "", outcome: String = "") {
        self.statement = statement
        self.test = test
        self.outcome = outcome
    }
}

public struct IntrospectReport: Sendable, Codable, Equatable {
    public enum Stop: String, Sendable, Codable {
        case answered, maxSteps, repeatedCall
    }

    public var goal: String
    public var steps: [IntrospectStep] = []
    public var hypotheses: [Hypothesis] = []
    public var answer: String?
    public var stop = Stop.maxSteps
    public var undone: [OverlayID] = []

    public init(goal: String) {
        self.goal = goal
    }
}

public struct Introspector: Sendable {
    public typealias Run = @Sendable (ToolCallOutput, any ModelResearch) async throws(BasaltError) -> String

    static let maxTokens = 1_024
    static let instructions = """
        You are the model being served, and you are studying yourself with X-ray tools that read your own weights, \
        activations, attention and prompt cache. Work towards the user's goal one tool call at a time. \
        Before each call, write the hypothesis it tests in exactly this format, one field per line:
        Hypothesis: <one sentence>
        Test: <the tool call and what result would confirm or refute it>
        Outcome: <confirmed, refuted or open, with the numbers that decide it>
        Restate a hypothesis with its outcome once a result decides it. Never repeat a call you already made. \
        When you can answer the goal, reply without a tool call: list every hypothesis with its outcome, then the answer.
        """

    let orchestrator: ModelOrchestratorActor
    let tools: [ToolDefinition]
    let journal: URL?
    let run: Run

    public init(orchestrator: ModelOrchestratorActor, tools: [ToolDefinition], journal: URL? = OverlayRegistry.defaultJournal, run: @escaping Run) {
        self.orchestrator = orchestrator
        self.tools = tools
        self.journal = journal
        self.run = run
    }

    public func introspect(_ request: IntrospectRequest) async throws(BasaltError) -> IntrospectReport {
        let before = Set(try await orchestrator.research { model in await model.overlays() }.map(\.id))
        let outcome: Result<IntrospectReport, BasaltError>
        do throws(BasaltError) {
            outcome = .success(try await loop(request))
        } catch {
            outcome = .failure(error)
        }
        let undone = await withTaskCancellationShield { await undoOverlays(except: before) }
        var report = try outcome.get()
        report.undone = undone
        return report
    }

    func loop(_ request: IntrospectRequest) async throws(BasaltError) -> IntrospectReport {
        var report = IntrospectReport(goal: request.goal)
        var messages = [ChatMessageInput(role: .system, text: Self.instructions), ChatMessageInput(role: .user, text: request.goal)]
        var made: Set<String> = []
        for _ in 0 ..< request.maxSteps {
            guard !Task.isCancelled else { throw .cancelled }
            let response = try await orchestrator.generate(request: generation(request.goal, messages))
            Hypothesis.merge(Hypothesis.parse(response.generatedText), into: &report.hypotheses)
            guard let call = response.toolCalls?.first else {
                let step = IntrospectStep(text: response.generatedText)
                report.steps.append(step)
                await record(report.steps.count, step)
                report.answer = response.generatedText
                report.stop = .answered
                return report
            }
            var step = IntrospectStep(text: response.generatedText, tool: call.name, arguments: call.arguments)
            let repeated = !made.insert(Self.key(call)).inserted
            if !repeated { step.result = Self.truncated(try await execute(call), to: request.resultBudget) }
            report.steps.append(step)
            await record(report.steps.count, step)
            if repeated {
                report.stop = .repeatedCall
                return report
            }
            messages.append(ChatMessageInput(role: .assistant, text: response.generatedText, toolCalls: [call]))
            messages.append(ChatMessageInput(role: .tool, text: step.result ?? "", toolCallId: call.id, toolName: call.name))
        }
        return report
    }

    func generation(_ goal: String, _ messages: [ChatMessageInput]) -> GenerationRequest {
        var request = GenerationRequest(prompt: goal, maxTokens: Self.maxTokens, temperature: 0, tools: tools, messages: messages)
        request.toolChoice = ToolChoice(parallel: false)
        request.enableThinking = false
        request.cacheKey = "introspect"
        return request.withOrigin(RequestOrigin(source: "introspect", endpoint: "introspect", stream: false))
    }

    func execute(_ call: ToolCallOutput) async throws(BasaltError) -> String {
        guard tools.contains(where: { $0.name == call.name }) else {
            return "error: \(call.name) is not one of the tools offered in this session"
        }
        let run = self.run
        do throws(BasaltError) {
            return try await orchestrator.research { model throws(BasaltError) in try await run(call, model) }
        } catch {
            guard !Task.isCancelled else { throw error }
            return "error: \(error.localizedDescription)"
        }
    }

    func undoOverlays(except kept: Set<OverlayID>) async -> [OverlayID] {
        let undone = try? await orchestrator.research { model in
            var undone: [OverlayID] = []
            for overlay in await model.overlays() where !kept.contains(overlay.id) {
                if (try? await model.undo(overlay.id)) != nil { undone.append(overlay.id) }
            }
            return undone
        }
        return undone ?? []
    }

    func record(_ index: Int, _ step: IntrospectStep) async {
        let now = Date()
        let model = await orchestrator.currentModelId
        let call = step.tool.map { "\($0)(\(step.arguments ?? "")) → \(String((step.result ?? "not run: repeated").prefix(200)))" }
        let description = "\(model) step \(index): \(call ?? "answer: \(step.text.prefix(200))")"
        do {
            try OverlayRegistry(journal: journal).record(
                "introspect", OverlayInfo(id: OverlayID(index), description: description, createdAt: now), at: now, fingerprint: "introspect")
        } catch {
            BasaltLogger(module: "Introspector").warn("The experiment journal was not written: \(error)")
        }
    }

    static func key(_ call: ToolCallOutput) -> String {
        let normalized = (try? JSONSerialization.jsonObject(with: Data(call.arguments.utf8)))
            .flatMap { try? JSONSerialization.data(withJSONObject: $0, options: .sortedKeys) }
            .map { String(decoding: $0, as: UTF8.self) }
        return call.name + (normalized ?? call.arguments.filter { !$0.isWhitespace })
    }

    static func truncated(_ text: String, to budget: Int) -> String {
        guard text.count > budget else { return text }
        return "\(text.prefix(budget))… [truncated \(text.count - budget) characters]"
    }
}

extension Hypothesis {
    static func parse(_ text: String) -> [Hypothesis] {
        var found: [Hypothesis] = []
        for line in text.split(whereSeparator: \.isNewline) {
            guard case let (key, value)? = field(line) else { continue }
            if key.hasPrefix("hypothes") || key.hasPrefix("statement") {
                if !value.isEmpty { found.append(Hypothesis(statement: value)) }
            } else if !found.isEmpty, key.hasPrefix("test") {
                found[found.count - 1].test = value
            } else if !found.isEmpty, key.hasPrefix("outcome") || key.hasPrefix("result") {
                found[found.count - 1].outcome = value
            }
        }
        return found
    }

    static func merge(_ new: [Hypothesis], into all: inout [Hypothesis]) {
        for hypothesis in new {
            if let index = all.firstIndex(where: { $0.statement == hypothesis.statement }) {
                all[index] = hypothesis
            } else {
                all.append(hypothesis)
            }
        }
    }

    private static func field(_ line: Substring) -> (key: String, value: String)? {
        let start = line.drop { " \t-*#>•.)".contains($0) || $0.isNumber }
        guard let colon = start.firstIndex(of: ":") else { return nil }
        let decoration = CharacterSet.whitespaces.union(CharacterSet(charactersIn: "*_`"))
        let key = start[..<colon].trimmingCharacters(in: decoration).lowercased()
        let value = start[start.index(after: colon)...].trimmingCharacters(in: decoration)
        return (key, value)
    }
}
