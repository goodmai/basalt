import Testing
import Foundation
import CryptoKit
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Image input on REST, MCP, the session log and a text-only engine (#345)")
struct ImageInputTests {
    static let png = Data("png".utf8)
    static let pngURI = "data:image/png;base64,\(png.base64EncodedString())"
    static let remoteRefusal = "remote image URLs are not fetched; send the image as a data: URI"

    private func post(_ uri: String, _ json: String) async throws -> (status: Int, body: String, sent: GenerationRequest?) {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        try await orch.loadModel(path: "mock")
        let app = Application(router: RESTServer(orchestrator: orch, config: try ServerConfig(modelPath: "mock")).makeRouter())
        let (status, body) = try await app.test(.router) { client in
            try await client.execute(
                uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: json)
            ) { (Int($0.status.code), String(buffer: $0.body)) }
        }
        return (status, body, await engine.lastRequest)
    }

    static func openAI(_ parts: [String]) -> String {
        #"{"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, \#(parts.joined(separator: ", "))]}]}"#
    }

    static func imageURL(_ url: String) -> String {
        #"{"type": "image_url", "image_url": {"url": "\#(url)"}}"#
    }

    static func anthropic(source: String) -> String {
        #"{"max_tokens": 16, "messages": [{"role": "user", "content": [{"type": "image", "source": \#(source)}, {"type": "text", "text": "describe"}]}]}"#
    }

    @Test("OpenAI image_url with a data: URI reaches the engine as an ImageInput with its bytes, mime and SHA-256")
    func openAIDataURI() async throws {
        let result = try await post("/v1/chat/completions", Self.openAI([Self.imageURL(Self.pngURI)]))
        #expect(result.status == 200)
        let message = try #require(result.sent?.messages?.first)
        #expect(message.text == "what is this?")
        let image = try #require(message.images.first)
        #expect(image.data == Self.png)
        #expect(image.mime == "image/png")
        #expect(image.digest == SHA256.hash(data: Self.png).map { String(format: "%02x", $0) }.joined())
    }

    @Test("Responses input_image with a string data: URI is an image of that user turn")
    func responsesInputImage() async throws {
        let body = #"{"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}, {"type": "input_image", "image_url": "\#(Self.pngURI)"}]}]}"#
        let result = try await post("/v1/responses", body)
        #expect(result.status == 200)
        #expect(result.sent?.messages?.first?.images.map(\.data) == [Self.png])
        #expect(result.sent?.messages?.first?.text == "hi")
    }

    @Test("Anthropic base64 image source reaches the engine next to the text of its turn")
    func anthropicBase64() async throws {
        let source = #"{"type": "base64", "media_type": "image/png", "data": "\#(Self.png.base64EncodedString())"}"#
        let result = try await post("/v1/messages", Self.anthropic(source: source))
        #expect(result.status == 200)
        let message = try #require(result.sent?.messages?.first)
        #expect(message.text == "describe")
        #expect(message.images.map(\.data) == [Self.png])
    }

    @Test("A remote image URL is a 400 that says it is not fetched, and nothing reaches the engine", arguments: [
        ("/v1/chat/completions", openAI([imageURL("https://example.com/cat.png")])),
        ("/v1/responses", #"{"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "http://example.com/cat.png"}]}]}"#),
        ("/v1/messages", anthropic(source: #"{"type": "url", "url": "https://example.com/cat.png"}"#)),
    ])
    func remoteURLsRefused(uri: String, body: String) async throws {
        let result = try await post(uri, body)
        #expect(result.status == 400)
        #expect(result.body.contains(Self.remoteRefusal), "\(result.body)")
        #expect(result.sent == nil)
    }

    @Test("Malformed base64, an unsupported mime type, a URI without ;base64 and a part without a URL are 400s", arguments: [
        ("/v1/chat/completions", openAI([imageURL("data:image/png;base64,@@@@")]), "not valid base64"),
        ("/v1/chat/completions", openAI([imageURL("data:image/gif;base64,cG5n")]), "is not supported"),
        ("/v1/chat/completions", openAI([imageURL("data:image/png,cG5n")]), "must have the form"),
        ("/v1/chat/completions", openAI([#"{"type": "image_url", "image_url": {"detail": "auto"}}"#]), "needs an image_url"),
        ("/v1/messages", anthropic(source: #"{"type": "base64", "media_type": "text/plain", "data": "cG5n"}"#), "is not supported"),
        ("/v1/messages", anthropic(source: #"{"type": "base64", "media_type": "image/png", "data": "%%%"}"#), "not valid base64"),
    ])
    func malformedRefused(uri: String, body: String, problem: String) async throws {
        let result = try await post(uri, body)
        #expect(result.status == 400)
        #expect(result.body.contains(problem), "\(result.body)")
        #expect(result.sent == nil)
    }

    @Test("Eight images pass; a ninth is a 400 that names the 8-image limit")
    func imageCountLimit() async throws {
        let eight = try await post("/v1/chat/completions", Self.openAI(Array(repeating: Self.imageURL(Self.pngURI), count: 8)))
        #expect(eight.status == 200)
        #expect(eight.sent?.messages?.first?.images.count == 8)
        let nine = try await post("/v1/chat/completions", Self.openAI(Array(repeating: Self.imageURL(Self.pngURI), count: 9)))
        #expect(nine.status == 400)
        #expect(nine.body.contains("at most 8 images; this one has 9"), "\(nine.body)")
    }

    @Test("An image of 20 MB passes the size check; one byte more is a 400 that names the 20 MB limit")
    func imageSizeLimit() async throws {
        #expect(try ImageInput(data: Data(count: ImageInput.maxBytes), mime: "image/jpeg").data.count == ImageInput.maxBytes)
        let oversized = "data:image/jpeg;base64,\(Data(count: ImageInput.maxBytes + 1).base64EncodedString())"
        let result = try await post("/v1/chat/completions", Self.openAI([Self.imageURL(oversized)]))
        #expect(result.status == 400)
        #expect(result.body.contains("over the limit of 20 MB per image"), "\(result.body.prefix(300))")
    }

    @Test("Images that are each within 20 MB but add up to more than 32 MB are refused")
    func imageTotalLimit() throws {
        let image = try ImageInput(data: Data(count: 12 * 1024 * 1024), mime: "image/png")
        #expect(throws: Never.self) { try ImageInput.checkLimits([image, image]) }
        #expect(throws: BasaltError.self) { try ImageInput.checkLimits([image, image, image]) }
    }

    struct ImageTree {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-images-\(UUID().uuidString)")
        var inside: URL { root.appendingPathComponent("inside") }
        var outside: URL { root.appendingPathComponent("outside") }

        init() throws {
            for directory in [inside, outside] { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
            try ImageInputTests.png.write(to: inside.appendingPathComponent("a.png"))
            try ImageInputTests.png.write(to: outside.appendingPathComponent("b.png"))
            try FileManager.default.createSymbolicLink(
                at: inside.appendingPathComponent("escape.png"), withDestinationURL: outside.appendingPathComponent("b.png"))
        }
    }

    private func mcpGenerate(_ image: [String: Any], imageDirectories: [URL]) async throws -> (line: String, sent: GenerationRequest?) {
        let engine = MockInferenceEngine()
        let orch = ModelOrchestratorActor(engine: engine)
        try await orch.loadModel(path: "test-model")
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orch) { capturer.capture($0) }.withImageDirectories(imageDirectories)
        let call: [String: Any] = [
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "basalt_generate", "arguments": ["prompt": "what is this?", "images": [image]]],
        ]
        await server.dispatch(String(decoding: try JSONSerialization.data(withJSONObject: call), as: UTF8.self))
        return (capturer.lines.first ?? "", await engine.lastRequest)
    }

    @Test("MCP generate reads a path inside --image-dir and takes {data, mime} as base64")
    func mcpImagesAccepted() async throws {
        let tree = try ImageTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let byPath = try await mcpGenerate(["path": tree.inside.appendingPathComponent("a.png").path], imageDirectories: [tree.inside])
        #expect(byPath.line.contains("\"result\""), "\(byPath.line)")
        let message = try #require(byPath.sent?.messages?.first)
        #expect(message.text == "what is this?")
        #expect(message.images.map(\.data) == [Self.png])
        #expect(message.images.map(\.mime) == ["image/png"])
        let inline = try await mcpGenerate(["data": Self.png.base64EncodedString(), "mime": "image/png"], imageDirectories: [])
        #expect(inline.sent?.messages?.first?.images.map(\.data) == [Self.png])
    }

    @Test("MCP refuses a path outside --image-dir, a symlink or .. escaping it, and every path when no --image-dir is set",
          arguments: ["outside", "symlink", "dotdot", "noImageDir"])
    func mcpPathRefused(_ escape: String) async throws {
        let tree = try ImageTree()
        defer { try? FileManager.default.removeItem(at: tree.root) }
        let path = switch escape {
        case "outside": tree.outside.appendingPathComponent("b.png").path
        case "symlink": tree.inside.appendingPathComponent("escape.png").path
        case "dotdot": tree.inside.path + "/../outside/b.png"
        default: tree.inside.appendingPathComponent("a.png").path
        }
        let result = try await mcpGenerate(["path": path], imageDirectories: escape == "noImageDir" ? [] : [tree.inside])
        #expect(result.line.contains("-32602"), "\(result.line)")
        #expect(result.line.contains("is not inside a directory given with serve --image-dir"), "\(result.line)")
        #expect(result.sent == nil)
    }

    @Test("An image request leaves the SHA-256 and byte count in the session log, never the base64 bytes")
    func sessionLogKeepsNoImageBytes() async throws {
        let image = try ImageInput(data: Data(repeating: 0xAB, count: 4096), mime: "image/png")
        for redacts in [true, false] {
            let dir = SessionLogTests.tempDirectory()
            defer { try? FileManager.default.removeItem(at: dir) }
            let writer = SessionLogWriter(mode: .full, directory: dir, redacts: redacts, defaultSession: "t")
            let request = GenerationRequest(prompt: "", messages: [ChatMessageInput(role: .user, text: "what is this?", images: [image])])
            await writer.logRequest(request, id: "r1", origin: .unspecified, modelRequested: nil)
            await writer.flush()
            let text = SessionLogTests.rawText(in: dir)
            #expect(!text.contains(String(image.data.base64EncodedString().prefix(64))), "redacts \(redacts)")
            #expect(text.contains(image.digest), "redacts \(redacts)")
            #expect(text.contains("4096"), "redacts \(redacts)")
        }
    }
}

@Suite("Image prompts in the engine: the text-only refusal, the cache namespace and the plain path (#345)")
struct ImagePromptEngineTests {
    @Test("Image digests join the namespace: the same digests in order share it, different ones never do, no images keeps today's key")
    func imageNamespace() throws {
        let request = GenerationRequest(prompt: "p").with { $0.cacheKey = "chat" }
        let (red, blue) = (try ImageInput(data: Data("red".utf8), mime: "image/png"), try ImageInput(data: Data("blue".utf8), mime: "image/png"))
        func namespace(_ media: PromptMedia) -> String? {
            GenerationPlan.cacheNamespace(request, GenerationPlan.Settings(), cacheEnabled: true, media: media, sameEpoch: true)
        }
        #expect(namespace(.text) == "chat")
        #expect(namespace(.images(digests: [red.digest], end: 40)) == namespace(.images(digests: [red.digest], end: 90)))
        #expect(namespace(.images(digests: [red.digest], end: 40)) != namespace(.images(digests: [blue.digest], end: 40)))
        #expect(namespace(.images(digests: [red.digest, blue.digest], end: 40)) != namespace(.images(digests: [blue.digest, red.digest], end: 40)))
        #expect(namespace(.images(digests: [red.digest], end: 40)) != namespace(.text))
        #expect(namespace(.uncacheable) == nil)
        #expect(GenerationPlan.cacheNamespace(
            request.with { $0.cacheKey = nil }, GenerationPlan.Settings(), cacheEnabled: true, media: .images(digests: [red.digest], end: 40),
            sameEpoch: true) == nil)
    }

    @Test("A prompt's media: tokens only is text, images end after the last image token, images the model cannot locate are uncacheable")
    func promptMedia() throws {
        let image = try ImageInput(data: Data("red".utf8), mime: "image/png")
        let messages = [ChatMessageInput(role: .user, text: "what?", images: [image])]
        let tokens = [1, 2, 9, 9, 9, 3, 4]
        #expect(PromptMedia.of(nil, carriesMoreThanTokens: false, tokens: tokens, imageToken: 9) == .text)
        #expect(PromptMedia.of(messages, carriesMoreThanTokens: true, tokens: tokens, imageToken: 9) == .images(digests: [image.digest], end: 5))
        #expect(PromptMedia.of(messages, carriesMoreThanTokens: true, tokens: tokens, imageToken: nil) == .uncacheable)
        #expect(PromptMedia.of(messages, carriesMoreThanTokens: true, tokens: tokens, imageToken: 7) == .uncacheable)
        #expect(PromptMedia.of(nil, carriesMoreThanTokens: true, tokens: tokens, imageToken: 9) == .uncacheable)
    }

    @Test("A prompt that carries media takes the plain path: no lookup, MTP, anchors, blending or batching, even with a namespace")
    func mediaTakesThePlainPath() {
        let request = GenerationRequest(prompt: "p").withSpeculativeLookup(true)
        let facts = GenerationPlan.Facts(
            namespace: "chat#images:a", promptTokens: 100, carriesMoreThanTokens: true, hasMTPDrafter: true, blendAvailable: true,
            modelIsHybrid: false, modelHasRecurrentState: true)
        #expect(GenerationPlan.make(request, GenerationPlan.Settings(), facts) == GenerationPlan())
        var tokensOnly = facts
        tokensOnly.carriesMoreThanTokens = false
        #expect(GenerationPlan.make(request, GenerationPlan.Settings(), tokensOnly).speculativeLookup)
    }

    @Test("A text-only model refuses images with a 400 that names it")
    func textOnlyModelRefusesImages() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = GenerationRequest(
            prompt: "", maxTokens: 4, temperature: 0,
            messages: [ChatMessageInput(role: .user, text: "what is this?", images: [try ImageInput(data: ImageInputTests.png, mime: "image/png")])])
        let error = await #expect(throws: BasaltError.self) { try await engine.generate(request: request) }
        #expect(error?.httpStatus == 400)
        #expect(error?.localizedDescription.contains("tiny-qwen3 does not take images") == true, "\(String(describing: error))")
    }
}

@Suite("Image inputs over the pixel cap are refused from the header, before any decode")
struct ImagePixelCapTests {
    static func crc(_ bytes: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for byte in bytes {
            c ^= UInt32(byte)
            for _ in 0 ..< 8 { c = c & 1 == 1 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        }
        return ~c
    }

    static func bigEndian(_ value: UInt32) -> [UInt8] { [24, 16, 8, 0].map { UInt8(value >> $0 & 0xFF) } }

    static func chunk(_ type: String, _ body: [UInt8]) -> [UInt8] {
        bigEndian(UInt32(body.count)) + Array(type.utf8) + body + bigEndian(crc(Array(type.utf8) + body))
    }

    static func blankPNG(side: UInt32) throws -> Data {
        let rows = Data(count: Int(side) * (Int(side) / 8 + 1))
        let deflated = try (rows as NSData).compressed(using: .zlib) as Data
        let adler = UInt32(rows.count % 65_521) << 16 | 1
        let header = chunk("IHDR", bigEndian(side) + bigEndian(side) + [1, 0, 0, 0, 0])
        return Data([137, 80, 78, 71, 13, 10, 26, 10] + header + chunk("IDAT", [0x78, 0x9C] + Array(deflated) + bigEndian(adler)) + chunk("IEND", []))
    }

    @Test("A 20000x20000 one-bit PNG of a few dozen KB is refused as over 50 megapixels; a 200-DPI Letter page passes")
    func refusesLargeDimensions() throws {
        let bomb = try Self.blankPNG(side: 20_000)
        #expect(bomb.count < 100_000)
        #expect(throws: BasaltError.self) { try ImageInput(data: bomb, mime: "image/png") }
        #expect(throws: BasaltError.self) { try ImageInput(base64: bomb.base64EncodedString(), mime: "image/png") }
        _ = try ImageInput(data: try OCRFixture.png(width: 1700, height: 2200), mime: "image/png")
    }
}
