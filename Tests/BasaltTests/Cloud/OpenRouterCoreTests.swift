import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCloud

@Suite("OpenRouterCoreTests")
struct OpenRouterCoreTests {
    static func config(key: String, url: String) throws -> OpenRouterClient.Config {
        .init(apiKey: key, baseURL: try #require(URL(string: url)), timeout: 5, maxRetries: 1)
    }

    @Test("OpenRouter hosts and subdomains are recognised case-insensitively; look-alikes and hostless URLs are not")
    func openRouterHosts() throws {
        for (url, expected) in [
            ("https://openrouter.ai/api/v1", true), ("https://EU.OpenRouter.ai/v1", true),
            ("https://notopenrouter.ai/v1", false), ("http://127.0.0.1:8080/v1", false), ("file:///tmp", false)
        ] {
            #expect(OpenRouterClient.isOpenRouter(try #require(URL(string: url))) == expected, "\(url)")
        }
    }

    @Test("An empty key is refused only for OpenRouter; the API config carries the key, limits and attribution headers")
    func apiConfig() throws {
        #expect(throws: BasaltError.self) {
            try OpenRouterClient.apiConfig(Self.config(key: "", url: "https://openrouter.ai/api/v1"))
        }
        let local = try OpenRouterClient.apiConfig(Self.config(key: "", url: "http://127.0.0.1:8080/v1"))
        #expect(local.apiKey.isEmpty)
        let api = try OpenRouterClient.apiConfig(Self.config(key: "sk", url: "https://openrouter.ai/api/v1"))
        #expect(api.apiKey == "sk")
        #expect(api.timeout == 5)
        #expect(api.maxRetries == 1)
        #expect(api.extraHeaders == ["HTTP-Referer": "Basalt/\(HealthResponse.version)", "X-Title": "Basalt"])
    }

    @Test("A chat request encodes snake_case keys and omits unset fields")
    func chatRequestEncoding() throws {
        var request = OpenRouterClient.ChatRequest(model: "m", messages: [.init(role: "user", content: "hi")], maxTokens: 8, stream: true)
        request.provider = .init(allowFallbacks: false)
        let json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any])
        #expect(Set(json.keys) == ["model", "messages", "max_tokens", "stream", "provider"])
        #expect(json["max_tokens"] as? Int == 8)
        #expect((json["provider"] as? [String: Any])?["allow_fallbacks"] as? Bool == false)
    }

    @Test("Chat, stream and model-list responses decode their snake_case fields, with usage shared by both chat shapes")
    func responseDecoding() throws {
        let usage = #""usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}"#
        let chat = try JSONDecoder().decode(OpenRouterClient.ChatResponse.self, from: Data(
            #"{"id":"a","model":"m","choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"length"}],\#(usage)}"#.utf8))
        #expect(chat.choices.first?.finishReason == "length")
        #expect(chat.usage?.completionTokens == 2)
        let chunk = try JSONDecoder().decode(OpenRouterClient.ChatStreamResponse.self, from: Data(
            #"{"id":"a","model":"m","choices":[{"delta":{"content":"y"},"finish_reason":null}],\#(usage)}"#.utf8))
        #expect(chunk.choices.first?.delta.content == "y")
        #expect(chunk.choices.first?.finishReason == nil)
        #expect(chunk.usage?.promptTokens == 3)
        let models = try JSONDecoder().decode(OpenRouterClient.ModelsResponse.self, from: Data(
            #"{"data":[{"id":"o/m","name":"M","context_length":4096,"pricing":{"prompt":"0.000001","completion":"0.000002"}}]}"#.utf8))
        #expect(models.data.first?.contextLength == 4_096)
        #expect(models.data.first?.pricing.completion == "0.000002")
    }
}
