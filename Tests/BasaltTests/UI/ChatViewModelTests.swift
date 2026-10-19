import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltApp

/// The SwiftUI chat client end to end: ChatViewModel → RESTClient → a stubbed
/// /api/v1/generate/stream that speaks the server's StreamChunk SSE format.
@Suite("ChatViewModel and RESTClient over a stubbed stream")
@MainActor
struct ChatViewModelTests {

    private func sse(_ chunks: [StreamChunk]) throws -> Data {
        var body = ""
        for chunk in chunks {
            body += "data: \(String(decoding: try JSONEncoder().encode(chunk), as: UTF8.self))\n\n"
        }
        return Data((body + "data: [DONE]\n\n").utf8)
    }

    private func waitUntilIdle(_ vm: ChatViewModel) async throws {
        for _ in 0..<500 where vm.isGenerating { try await Task.sleep(for: .milliseconds(10)) }
        #expect(!vm.isGenerating, "generation did not finish")
    }

    @Test("A sent message streams reasoning and text into one assistant reply and records the stats")
    func streamsReply() async throws {
        let metadata = GenerationResponse(
            generatedText: "", promptTokens: 7, completionTokens: 3, tokensPerSecond: 30, generationTime: 0.1,
            timeToFirstToken: 0.02, memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop)
        let body = try sse([.reasoning("thinking. "), .text("Hel"), .text("lo"), .metadata(metadata)])
        let host = StubURLProtocol.host { _, _ in (200, ["Content-Type": "text/event-stream"], body) }

        let client = RESTClient(baseURL: "http://\(host)", configuration: StubURLProtocol.configuration)
        await client.setToken("k-1")
        let vm = ChatViewModel()
        vm.currentInput = "  hi there \n"
        vm.sendMessage(restClient: client)
        #expect(vm.currentInput.isEmpty)
        #expect(vm.isGenerating)
        try await waitUntilIdle(vm)

        #expect(vm.messages.map(\.role) == [.user, .assistant])
        #expect(vm.messages.first?.text == "hi there")
        #expect(vm.messages.last?.text == "thinking. Hello")
        #expect(vm.errorMessage == nil)
        #expect(vm.statsPromptTokens == 7)
        #expect(vm.statsCompletionTokens == 3)
        #expect(vm.statsTokensPerSecond == 30)

        let (request, sent) = try #require(StubURLProtocol.requests(for: host).first)
        #expect(request.url?.path == "/api/v1/generate/stream")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer k-1")
        let decoded = try JSONDecoder().decode(GenerationRequest.self, from: try #require(sent))
        #expect(decoded.prompt == "hi there")
        #expect(decoded.maxTokens == 65536)
    }

    @Test("A non-200 reply ends generation with an error message and an empty assistant turn")
    func serverError() async throws {
        let host = StubURLProtocol.host { _, _ in (503, [:], Data()) }
        let vm = ChatViewModel()
        vm.currentInput = "hi"
        vm.sendMessage(restClient: RESTClient(baseURL: "http://\(host)", configuration: StubURLProtocol.configuration))
        try await waitUntilIdle(vm)
        #expect(vm.errorMessage != nil)
        #expect(vm.messages.last?.text == "")
    }

    @Test("Blank input sends nothing")
    func blankInput() {
        let vm = ChatViewModel()
        vm.currentInput = " \n\t"
        vm.sendMessage(restClient: RESTClient(baseURL: "http://unused.test", configuration: StubURLProtocol.configuration))
        #expect(vm.messages.isEmpty)
        #expect(!vm.isGenerating)
    }
}
