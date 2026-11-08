import Foundation
import BasaltCore

struct TeacherClient: Sendable {
    struct Completion: Sendable, Equatable, Codable {
        var text: String
        var reasoning: String?
        var truncated: Bool
        var inputTokens: Int
        var outputTokens: Int
    }

    struct OpenAIReply: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable {
                var content: String?
                var reasoning: String?
                var reasoningContent: String?

                enum CodingKeys: String, CodingKey {
                    case content, reasoning
                    case reasoningContent = "reasoning_content"
                }
            }
            var message: Message
            var finishReason: String?

            enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }
        struct Usage: Decodable {
            var promptTokens: Int?
            var completionTokens: Int?

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
            }
        }
        var choices: [Choice]
        var usage: Usage?
    }

    struct AnthropicReply: Decodable {
        struct Block: Decodable {
            var type: String
            var text: String?
            var thinking: String?
        }
        struct Usage: Decodable {
            var inputTokens: Int?
            var outputTokens: Int?

            enum CodingKeys: String, CodingKey {
                case inputTokens = "input_tokens"
                case outputTokens = "output_tokens"
            }
        }
        var content: [Block]
        var stopReason: String?
        var usage: Usage?

        enum CodingKeys: String, CodingKey {
            case content, usage
            case stopReason = "stop_reason"
        }
    }

    static let keyVariable = "BASALT_TEACHER_API_KEY"
    static let urlVariable = "BASALT_TEACHER_URL"
    static let retries = 3
    let endpoint: URL
    let model: String
    let provider: TeacherProvider
    let key: String?
    let session: URLSession

    init(
        _ teacher: String, model: String?, provider: TeacherProvider = .openai,
        environment: [String: String] = ProcessInfo.processInfo.environment, session: URLSession = .shared
    ) throws(BasaltError) {
        guard var url = URL(string: teacher), ["http", "https"].contains(url.scheme?.lowercased() ?? ""), url.host != nil else {
            throw .invalidRequestStructure(details: "teacher must be the http(s) URL of an OpenAI-compatible or Anthropic server, e.g. http://127.0.0.1:8080/v1")
        }
        let suffix = provider == .anthropic ? "messages" : "chat/completions"
        if !url.path.hasSuffix(suffix) { url.append(path: suffix) }
        endpoint = url
        self.model = model ?? "default"
        self.provider = provider
        self.session = session
        key = Self.key(for: url, environment: environment)
    }

    static func key(for url: URL, environment: [String: String]) -> String? {
        guard let trusted = environment[urlVariable].flatMap(URL.init(string:)),
              trusted.scheme?.lowercased() == url.scheme?.lowercased(), trusted.host?.lowercased() == url.host?.lowercased(), trusted.port == url.port
        else { return nil }
        return environment[keyVariable]
    }

    func answer(_ prompt: String) async throws(BasaltError) -> String {
        try await complete(prompt, maxTokens: 256, temperature: 0).text
    }

    func complete(_ prompt: String, maxTokens: Int, temperature: Float, reasoningEffort: String? = nil) async throws(BasaltError) -> Completion {
        let request = try request(prompt, maxTokens: maxTokens, temperature: temperature, reasoningEffort: reasoningEffort)
        var attempt = 0
        while true {
            let (status, data) = try await send(request)
            if status == 200, let completion = decode(data) { return completion }
            attempt += 1
            guard status == 429 || status >= 500 || status == 0, attempt <= Self.retries else {
                throw .inferenceError("the teacher at \(endpoint.absoluteString) returned HTTP \(status) without a \(provider.rawValue) completion")
            }
            try? await Task.sleep(for: .milliseconds(250 << attempt))
        }
    }

    func request(_ prompt: String, maxTokens: Int, temperature: Float, reasoningEffort: String? = nil) throws(BasaltError) -> URLRequest {
        var request = URLRequest(url: endpoint, timeoutInterval: 600)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        switch provider {
        case .openai:
            if let key { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        case .anthropic:
            if let key { request.setValue(key, forHTTPHeaderField: "x-api-key") }
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        }
        var body: [String: Any] = [
            "model": model, "messages": [["role": "user", "content": prompt]], "temperature": Double(temperature), "max_tokens": maxTokens,
        ]
        if let reasoningEffort { body[provider == .anthropic ? "output_config" : "reasoning_effort"] = provider == .anthropic ? ["effort": reasoningEffort] : reasoningEffort }
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            throw .invalidRequestStructure(details: "teacher request: \(error.localizedDescription)")
        }
        return request
    }

    func send(_ request: URLRequest) async throws(BasaltError) -> (Int, Data) {
        do {
            let (data, response) = try await session.data(for: request)
            return ((response as? HTTPURLResponse)?.statusCode ?? 0, data)
        } catch is CancellationError {
            throw .inferenceError("the teacher call was cancelled")
        } catch {
            return (0, Data())
        }
    }

    func decode(_ data: Data) -> Completion? {
        switch provider {
        case .openai:
            guard let reply = try? JSONDecoder().decode(OpenAIReply.self, from: data), let choice = reply.choices.first else { return nil }
            return Completion(
                text: choice.message.content ?? "", reasoning: choice.message.reasoningContent ?? choice.message.reasoning,
                truncated: choice.finishReason == "length",
                inputTokens: reply.usage?.promptTokens ?? 0, outputTokens: reply.usage?.completionTokens ?? 0)
        case .anthropic:
            guard let reply = try? JSONDecoder().decode(AnthropicReply.self, from: data) else { return nil }
            let thinking = reply.content.compactMap(\.thinking).joined(separator: "\n")
            return Completion(
                text: reply.content.compactMap(\.text).joined(), reasoning: thinking.isEmpty ? nil : thinking,
                truncated: reply.stopReason == "max_tokens",
                inputTokens: reply.usage?.inputTokens ?? 0, outputTokens: reply.usage?.outputTokens ?? 0)
        }
    }
}
