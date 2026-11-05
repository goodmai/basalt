import Testing
import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
import MLX
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Image input end to end on a real vision model: OpenAI, Anthropic and MCP agree, and the prompt cache follows the image (#345)", .serialized)
struct VisionRealModelTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_VISION_MODEL_PATH"]
    }

    static let question = "What colour is the square and which digit is shown? Answer briefly."

    static func png(square: (CGFloat, CGFloat, CGFloat), digit: String) throws -> Data {
        let (width, height) = (384, 256)
        let sRGB = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: square.0, green: square.1, blue: square.2, alpha: 1))
        context.fill(CGRect(x: 32, y: 48, width: 160, height: 160))
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica-Bold" as CFString, 180, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(red: 0, green: 0, blue: 0, alpha: 1),
        ]
        context.textPosition = CGPoint(x: 236, y: 64)
        CTLineDraw(CTLineCreateWithAttributedString(NSAttributedString(string: digit, attributes: attributes)), context)
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    struct Turn {
        let text: String
        let cachedTokens: Int
    }

    static func post(_ client: some TestClientProtocol, _ uri: String, _ body: [String: Any]) async throws -> [String: Any] {
        let json = String(decoding: try JSONSerialization.data(withJSONObject: body), as: UTF8.self)
        let text = try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: json)) {
            String(buffer: $0.body)
        }
        return try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], "\(text)")
    }

    static func openAI(_ client: some TestClientProtocol, _ messages: [[String: Any]]) async throws -> Turn {
        let reply = try await post(client, "/v1/chat/completions", ["messages": messages, "max_tokens": 64, "temperature": 0])
        let choice = try #require((reply["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any], "\(reply)")
        let usage = reply["usage"] as? [String: Any]
        let cached = (usage?["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int ?? -1
        return Turn(text: choice["content"] as? String ?? "", cachedTokens: cached)
    }

    static func openAIUser(_ image: Data, _ text: String = question) -> [String: Any] {
        ["role": "user", "content": [
            ["type": "text", "text": text],
            ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(image.base64EncodedString())"]],
        ]]
    }

    static func anthropic(_ client: some TestClientProtocol, _ image: Data) async throws -> String {
        let content: [[String: Any]] = [
            ["type": "image", "source": ["type": "base64", "media_type": "image/png", "data": image.base64EncodedString()]],
            ["type": "text", "text": question],
        ]
        let reply = try await post(client, "/v1/messages", ["max_tokens": 64, "temperature": 0, "messages": [["role": "user", "content": content]]])
        let blocks = try #require(reply["content"] as? [[String: Any]], "\(reply)")
        return blocks.filter { $0["type"] as? String == "text" }.compactMap { $0["text"] as? String }.joined()
    }

    static func mcp(_ orchestrator: ModelOrchestratorActor, _ image: Data) async throws -> String {
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }
        let arguments: [String: Any] = [
            "prompt": question, "maxTokens": 64, "temperature": 0,
            "images": [["data": image.base64EncodedString(), "mime": "image/png"]],
        ]
        let call: [String: Any] = ["jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["name": "basalt_generate", "arguments": arguments]]
        await server.dispatch(String(decoding: try JSONSerialization.data(withJSONObject: call), as: UTF8.self))
        let line = try #require(capturer.lines.first)
        let reply = try #require(try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any], "\(line)")
        let content = try #require((reply["result"] as? [String: Any])?["content"] as? [[String: Any]], "\(line)")
        return content.compactMap { $0["text"] as? String }.joined()
    }

    @Test("The red square and the digit 7 are named the same way through OpenAI, Anthropic and MCP; the same image reuses the cache, another image reuses nothing",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_VISION_MODEL_PATH (a vision model, e.g. Gemma 4 26B-A4B)"))
    func imageEndToEnd() async throws {
        let path = try #require(Self.modelPath)
        let log = try await ModelRunLog("vision", modelPath: path)
        defer { log.releaseGPU() }
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: path)).makeRouter())
        let (red, blue) = (try Self.png(square: (1, 0, 0), digit: "7"), try Self.png(square: (0, 0, 1), digit: "3"))
        let (first, anthropic, second, other) = try await app.test(.router) { client in
            let first = try await log.step("openai") { try await Self.openAI(client, [Self.openAIUser(red)]) }
            let anthropic = try await log.step("anthropic") { try await Self.anthropic(client, red) }
            let followUp: [[String: Any]] = [Self.openAIUser(red), ["role": "assistant", "content": first.text], ["role": "user", "content": "And the background colour?"]]
            let second = try await log.step("same-image-turn") { try await Self.openAI(client, followUp) }
            let other = try await log.step("other-image") { try await Self.openAI(client, [Self.openAIUser(blue)]) }
            return (first, anthropic, second, other)
        }
        let mcp = try await log.step("mcp") { try await Self.mcp(orchestrator, red) }
        log.record("answers", "openai \(first.text.debugDescription) anthropic \(anthropic.debugDescription) mcp \(mcp.debugDescription)")
        #expect(first.text.lowercased().contains("red"), "\(first.text)")
        #expect(first.text.contains("7"), "\(first.text)")
        #expect(anthropic == first.text)
        #expect(mcp == first.text)
        log.record("cache", "same image reused \(second.cachedTokens) tokens, another image \(other.cachedTokens) \(log.host())")
        #expect(second.cachedTokens > 0)
        #expect(other.cachedTokens == 0)
        let broken = GenerationRequest(prompt: "", maxTokens: 8, temperature: 0, messages: [
            ChatMessageInput(role: .user, text: Self.question, images: [try ImageInput(data: Data("not an image".utf8), mime: "image/png")]),
        ])
        let error = await #expect(throws: BasaltError.self) { try await orchestrator.generate(request: broken) }
        #expect(error?.localizedDescription.contains("the image in message 0 cannot be decoded") == true, "\(String(describing: error))")
        Memory.clearCache()
    }
}
