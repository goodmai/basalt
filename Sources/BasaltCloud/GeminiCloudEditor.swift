import BasaltCore
import Foundation

public struct GeminiCloudEditorConfig: Sendable {
    public static let defaultGeminiBaseURL = URL(string: "https://generativelanguage.googleapis.com/v1beta/openai")!
    public static let defaultOpenRouterBaseURL = URL(string: "https://openrouter.ai/api/v1")!

    public let model: String
    public let apiKey: String
    public let baseURL: URL
    public let allowCloudEditor: Bool
    public let maxSessionUSD: Double
    public let maxTokensPerTurn: Int

    public init(
        model: String,
        apiKey: String,
        baseURL: URL = Self.defaultGeminiBaseURL,
        allowCloudEditor: Bool = false,
        maxSessionUSD: Double = 2.0,
        maxTokensPerTurn: Int = 2048
    ) {
        self.model = model
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.allowCloudEditor = allowCloudEditor
        self.maxSessionUSD = max(0, maxSessionUSD)
        self.maxTokensPerTurn = max(128, maxTokensPerTurn)
    }
}

public actor GeminiCloudEditor: EditSessionEditor {
    public nonisolated let editorLabel: String
    private let config: GeminiCloudEditorConfig
    private let apiClient: CloudAPIClient
    private let cloudGate: CloudGate?
    private let privateStrings: Set<String>
    private let practiceIds: [String]
    private var sessionSpentUSD: Double = 0

    public init(
        config: GeminiCloudEditorConfig,
        cloudGate: CloudGate? = nil,
        privateStrings: Set<String> = [],
        practiceIds: [String] = [],
        configuration: URLSessionConfiguration = .default
    ) throws {
        guard config.allowCloudEditor else {
            throw SelfRewriteSessionError.cloudEditorDisabled
        }
        guard !config.apiKey.isEmpty else {
            throw BasaltError.authenticationFailed(details: "GEMINI_API_KEY or OPENROUTER_API_KEY is required for cloud editor")
        }
        self.editorLabel = "cloud:\(config.model)"
        self.config = config
        self.cloudGate = cloudGate
        self.privateStrings = privateStrings
        self.practiceIds = practiceIds
        let apiConfig = CloudAPIConfig(
            apiKey: config.apiKey,
            baseURL: config.baseURL,
            timeout: 60.0,
            maxRetries: 2
        )
        self.apiClient = CloudAPIClient(config: apiConfig, configuration: configuration)
    }

    public func nextTurn(
        transcript: [EditSessionTranscriptEntry],
        remainingBudget: SelfRewriteBudget
    ) async throws -> EditEditorTurn {
        if let cloudGate, let refusal = await cloudGate.refusal(config.model) {
            throw BasaltError.invalidRequestStructure(details: "cloud gate refused \(config.model): \(refusal.reason)")
        }
        guard sessionSpentUSD < config.maxSessionUSD else {
            throw SelfRewriteBudgetError.cloudCostBudgetExceeded(requested: 0.01, remaining: 0)
        }
        let cleanEntries = try CloudPrivacyFence.sanitizeAndValidate(transcript, privateStrings: privateStrings)
        let messages = Self.buildMessages(entries: cleanEntries, budget: remainingBudget, practiceIds: practiceIds)
        var chatRequest = OpenRouterClient.ChatRequest(
            model: config.model,
            messages: messages,
            temperature: 0.2,
            maxTokens: config.maxTokensPerTurn,
            stream: false
        )
        if config.baseURL.host?.lowercased().contains("openrouter") == true {
            chatRequest.usage = .init(include: true)
        }
        let body = try JSONEncoder().encode(chatRequest)
        let response = try await performChat(body: body)
        let cost = Self.resolvedCostUSD(response.usage)
        let remainingBefore = max(0, config.maxSessionUSD - sessionSpentUSD)
        sessionSpentUSD += cost
        if let cloudGate, cost > 0 {
            await cloudGate.record(config.model, cost: cost)
        }
        if sessionSpentUSD > config.maxSessionUSD + 1e-12 {
            throw SelfRewriteBudgetError.cloudCostBudgetExceeded(requested: cost, remaining: remainingBefore)
        }
        guard let content = response.choices.first?.message.content else {
            throw SelfRewriteSessionError.invalidEditorTurn("cloud model returned empty message content")
        }
        let parsed = try EditEditorTurn.parseJSON(from: content)
        return EditEditorTurn(rationale: parsed.rationale, action: parsed.action, costUSD: cost)
    }

    private func performChat(body: Data) async throws -> OpenRouterClient.ChatResponse {
        do {
            return try await apiClient.performRequest(
                endpoint: "chat/completions",
                method: "POST",
                body: body,
                responseType: OpenRouterClient.ChatResponse.self
            )
        } catch let error as BasaltError {
            if let cloudGate {
                await cloudGate.failed(config.model, error)
            }
            throw error
        }
    }

    public func spentUSD() -> Double { sessionSpentUSD }

    private static func resolvedCostUSD(_ usage: OpenRouterClient.ChatResponse.Usage?) -> Double {
        if let reported = usage?.cost, reported.isFinite, reported >= 0 {
            return reported
        }
        guard let usage else { return 0 }
        let inputUSD = Double(usage.promptTokens) * 1.25 / 1_000_000.0
        let outputUSD = Double(usage.completionTokens) * 5.0 / 1_000_000.0
        return inputUSD + outputUSD
    }

    static func buildMessages(
        entries: [EditSessionTranscriptEntry],
        budget: SelfRewriteBudget,
        practiceIds: [String] = []
    ) -> [OpenRouterClient.ChatRequest.Message] {
        var system = SelfRewritePromptFormatter.systemPrompt(budget: budget)
        if !practiceIds.isEmpty {
            system += "\nAvailable practice problemIds: \(practiceIds.joined(separator: ", "))"
        }
        var messages = [OpenRouterClient.ChatRequest.Message(role: "system", content: system)]
        for entry in SelfRewritePromptFormatter.windowedEntries(entries) {
            let role = entry.role == "editor" ? "assistant" : "user"
            let content = entry.role == "tool"
                ? SelfRewritePromptFormatter.compactObservation(entry.content, limit: 6000)
                : entry.content
            messages.append(OpenRouterClient.ChatRequest.Message(role: role, content: content))
        }
        if entries.isEmpty {
            messages.append(OpenRouterClient.ChatRequest.Message(role: "user", content: "Begin session. Inspect model topology or tensor stats first."))
        }
        return messages
    }
}
