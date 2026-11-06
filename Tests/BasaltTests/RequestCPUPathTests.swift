import Testing
import Foundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Request CPU path before inference: byte scans, one shared parse, one image decode (#374)")
struct RequestCPUPathTests {
    static func referenceDepth(of data: Data) -> Int {
        var depth = 0
        var deepest = 0
        var inString = false
        var escaped = false
        for byte in data {
            if inString {
                if escaped { escaped = false } else if byte == UInt8(ascii: "\\") { escaped = true } else if byte == UInt8(ascii: "\"") { inString = false }
            } else if byte == UInt8(ascii: "\"") {
                inString = true
            } else if byte == UInt8(ascii: "[") || byte == UInt8(ascii: "{") {
                depth += 1
                deepest = max(deepest, depth)
            } else if byte == UInt8(ascii: "]") || byte == UInt8(ascii: "}") {
                depth -= 1
            }
        }
        return deepest
    }

    static func nested(_ levels: Int) -> String {
        String(repeating: "[", count: levels) + String(repeating: "]", count: levels)
    }

    static let corpus: [String] = [
        "", "{}", "[]", "\"", "\"\"", "\\", "]]]{{", "}{", "{\"a\":\"b\"}",
        nested(128), nested(129), "{\"x\":" + nested(127) + "}", "{\"x\":" + nested(128) + "}",
        #"{"s": "a \"quoted\" [[[ word"}"#, #"{"s": "\\"}[["#, #"{"s": "\\\"[["}"#, #"{"s": "\\\\\"[[[", "t": [1]}"#,
        #"{"s": "[{[{"}"#, #"{"s": "}}}]]]", "a": [[1]]}"#, #"{"s": "unterminated [[[["#, #"{"a": [[[{"b": "\"#,
        "{\"a\": [1, 2, {\"b\": \"c\u{0}\"}]}", "\u{0}", "[\u{0}[", "{\"s\": \"caf\u{e9} [\"}",
        #"{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}"#,
    ]

    static func fuzzCorpus(count: Int, length: Int) -> [Data] {
        let alphabet = Array("{}[]\"\\a :,".utf8) + [0]
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        return (0..<count).map { _ in
            Data((0..<length).map { _ in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return alphabet[Int(state >> 33) % alphabet.count]
            })
        }
    }

    @Test("The memchr scanner gives the same depth and zero-byte verdict as the byte-by-byte scan on every corpus body and its truncations")
    func scannerMatchesReference() {
        let bodies = Self.corpus.map { Data($0.utf8) } + Self.fuzzCorpus(count: 2000, length: 48)
        for body in bodies {
            for end in 0...body.count {
                let prefix = body.prefix(end)
                #expect(JSONNesting.depth(of: prefix) == Self.referenceDepth(of: prefix), "\(String(decoding: prefix, as: UTF8.self).debugDescription)")
                #expect(JSONNesting.containsZero(prefix) == prefix.contains(0))
            }
        }
        #expect(JSONNesting.depth(of: Data(Self.nested(128).utf8)) == 128)
        #expect(JSONNesting.depth(of: Data(Self.nested(129).utf8)) == 129)
        #expect(JSONNesting.depth(of: Data(#"{"s": "a \"[[[\" b"}"#.utf8)) == 1)
    }

    @Test("Folding DRY and XTC into the request decode gives the same values and the same failure text as a separate decode", arguments: [
        #"{"messages": [{"role": "user", "content": "hi"}]}"#,
        #"{"messages": [{"role": "user", "content": "hi"}], "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "dry_sequence_breakers": ["\n", ":"], "xtc_threshold": 0.1, "xtc_probability": 0.5}"#,
        #"{"messages": [{"role": "user", "content": "hi"}], "dry_allowed_length": "two"}"#,
        #"{"messages": [{"role": "user", "content": "hi"}], "xtc_probability": [1]}"#,
    ])
    func sharedDecodeMatchesSeparateDecode(json: String) throws {
        let data = Data(json.utf8)
        let shared = try JSONDecoder().decode(ChatPipeline.Body<OpenAIController.ChatCompletionRequest>.self, from: data)
        let separate = Result { try JSONDecoder().decode(ChatPipeline.AntiRepetition.self, from: data) }
        switch (shared.antiRepetition, separate) {
        case (.success(let folded), .success(let alone)): #expect(String(describing: folded) == String(describing: alone))
        case (.failure(let folded), .failure(let alone)): #expect(folded.localizedDescription == alone.localizedDescription)
        default: Issue.record("the folded and separate decodes disagree on \(json)")
        }
        #expect(shared.dto.messages.count == 1)
    }

    @Test("Tools and response_format read from one shared parse equal what each would read from its own parse")
    func sharedObjectFields() throws {
        let data = Data(#"""
            {"messages": [], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "b": 1, "a": 2}}}],
             "tool_choice": {"type": "function", "function": {"name": "f"}},
             "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}}}}}
            """#.utf8)
        let shared = ChatPipeline.object(in: data)
        let tools = try OpenAIController.requestedTools(in: shared)
        let alone = try OpenAIController.requestedTools(in: try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(tools.tools == alone.tools)
        #expect(tools.choice == alone.choice)
        #expect(tools.choice == ToolChoice(forcedTool: "f"))
        let format = try OpenAIController.responseFormat(in: shared, data: data)
        #expect(format == .jsonSchema(#"{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"integer"}}}"#, strict: false))
        #expect(OpenAIController.requestedTools(in: nil).tools == nil)
        #expect(try OpenAIController.responseFormat(in: ChatPipeline.object(in: Data("[1]".utf8)), data: data) == nil)
    }

    static func encoded(_ type: UTType, width: Int = 64, height: Int = 48) -> Data {
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(red: 0.8, green: 0.1, blue: 0.2, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width / 2, height: height))
        let out = NSMutableData()
        let destination = CGImageDestinationCreateWithData(out, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        CGImageDestinationFinalize(destination)
        return out as Data
    }

    static var imageCorpus: [Data] {
        let png = encoded(.png)
        let jpeg = encoded(.jpeg)
        var corruptBody = png
        corruptBody.replaceSubrange(40..<min(80, png.count), with: Data(repeating: 0xFF, count: min(80, png.count) - 40))
        return [png, jpeg, encoded(.png, width: 1, height: 1), png.prefix(8), png.prefix(33), png.prefix(png.count / 2),
                jpeg.prefix(2), jpeg.prefix(jpeg.count / 2), corruptBody, Data("not an image".utf8), Data(), Data(repeating: 0, count: 64)]
    }

    @Test("The header check plus one decode refuses exactly the images a full CIImage decode refuses, with the same message")
    func imageRefusalParity() throws {
        for data in Self.imageCorpus {
            let image = try ImageInput(data: data, mime: "image/png")
            let messages = [ChatMessageInput(role: .system, text: "s"), ChatMessageInput(role: .user, text: "q", images: [image])]
            let accepted = CIImage(data: data) != nil
            do {
                let decoded = try MLXInferenceEngine.decodedImages(messages)
                #expect(accepted, "header check let through \(data.count) bytes CIImage refuses")
                #expect(decoded.count == 1)
            } catch {
                #expect(!accepted, "header check refused \(data.count) bytes CIImage accepts")
                #expect(error.localizedDescription.contains("the image in message 1 cannot be decoded"))
            }
        }
    }

    @Test("Each distinct image is decoded once per request and the chat reuses the decoded image")
    func imagesDecodedOnce() throws {
        let image = try ImageInput(data: Self.encoded(.png), mime: "image/png")
        let other = try ImageInput(data: Self.encoded(.jpeg), mime: "image/jpeg")
        let messages = [ChatMessageInput(role: .user, text: "a", images: [image, other]), ChatMessageInput(role: .user, text: "b", images: [image])]
        let decoded = try MLXInferenceEngine.decodedImages(messages)
        #expect(decoded.count == 2)
        let chat = MLXInferenceEngine.chat(messages, lastText: "b", images: decoded)
        #expect(chat.map(\.images.count) == [2, 1])
        guard case .ciImage(let reused) = chat[1].images[0] else { Issue.record("expected a CIImage"); return }
        #expect(reused === decoded[image.digest])
    }
}
