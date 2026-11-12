import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltCloud
@testable import BasaltServer
@testable import BasaltCLI

@Suite("Gemini Decision 1 as a cloud decision stage: the Decisions API's wire format, multimodal media, compose rules and the pipeline's opt-in (#gemini-decision-1)", .timeLimit(.minutes(1)))
struct GeminiDecisionTests {
    static let questions: [String: SystemOneQuestion] = [
        "cracked": SystemOneQuestion(type: .noul, instructions: "Is the screen cracked?"),
        "team": SystemOneQuestion(type: .choice, instructions: "Which team?", criteria: .named(["billing": "Payments.", "tech": "Devices."])),
        "severity": SystemOneQuestion(type: .score, instructions: "How severe?", criteria: .levels(["Cosmetic.", "Workaround.", "Blocking."])),
    ]
    static let reply = #"{"answers":{"cracked":{"type":"noul","noul":0.97,"confidence":0.94},"team":{"type":"choice","choice":"tech","probabilities":{"tech":0.9,"billing":0.1},"confidence":0.9},"severity":{"type":"score","score":1.4,"legend":{"0":"Cosmetic."},"probabilities":{"0":0.1,"1":0.4,"2":0.5},"confidence":0.7}},"usage":{"total_input_tokens":2000,"total_output_tokens":0,"total_tokens":2000}}"#

    @Test("cloud:gemini/<model> picks the Gemini provider, its endpoint and its key variable")
    func provider() throws {
        #expect(CloudDecisionProvider.parse("gemini/gemini-decision-1") == (.gemini, "gemini-decision-1"))
        #expect(CloudDecisionProvider.parse("~typesafe/jev-latest") == (.openRouter, "~typesafe/jev-latest"))
        #expect(CloudDecisionProvider.gemini.baseURL.host == "generativelanguage.googleapis.com" && CloudDecisionProvider.gemini.keyEnvironment == "GEMINI_API_KEY")
        let gemini = try ServeCommand.parse(["--decision-model", "cloud:gemini/gemini-decision-1", "--decision-escalate", "self"])
        #expect(gemini.decision.keyEnvironment == "GEMINI_API_KEY" && gemini.decision.cloud?.model == "gemini-decision-1")
        let custom = try ServeCommand.parse(["--decision-model", "cloud:gemini/gemini-decision-1", "--decision-escalate", "self", "--decision-api-key-env", "MY_KEY"])
        #expect(custom.decision.keyEnvironment == "MY_KEY")
        #expect(try ServeCommand.parse(["--decision-model", "cloud:~typesafe/jev-latest", "--decision-escalate", "self"]).decision.keyEnvironment == "OPENROUTER_API_KEY")
    }

    @Test("The Gemini body sends input as text, or as blocks with the media after the redacted state; the answer decodes without a model and is priced by input tokens")
    func wire() throws {
        let image = try #require(DecisionMedia(kind: "image", mimeType: "image/png", data: Data([1, 2, 3])))
        var request = SystemOneRequest(state: "Ticket 1042, key sk-or-v1-0123456789abcdef0123456789abcdef", questions: Self.questions)
        let text = try JSONSerialization.jsonObject(with: RemoteDecisionCore.body(request, model: "gemini-decision-1", provider: .gemini)) as? [String: Any]
        #expect(text?["input"] is String && text?["state"] == nil && text?["calibrated"] == nil && !(text?["input"] as? String ?? "").contains("0123456789abcdef"))
        request.media = [image]
        let blocks = (try JSONSerialization.jsonObject(with: RemoteDecisionCore.body(request, model: "gemini-decision-1", provider: .gemini)) as? [String: Any])?["input"] as? [[String: String]]
        #expect(blocks?.map { $0["type"] } == ["text", "image"] && blocks?[1]["mime_type"] == "image/png" && blocks?[1]["data"] == Data([1, 2, 3]).base64EncodedString())
        let open = try JSONSerialization.jsonObject(with: RemoteDecisionCore.body(request, model: "jev", provider: .openRouter)) as? [String: Any]
        #expect(open?["state"] is String && open?["input"] == nil)
        let response = try #require(RemoteDecisionCore.response(Data(Self.reply.utf8), for: request, model: "gemini-decision-1", provider: .gemini))
        #expect(response.model == "gemini-decision-1" && response.answers["cracked"]?.noul == 0.97 && response.answers["team"]?.choice == "tech")
        #expect(response.answers["severity"]?.score == 1.4 && abs((response.usage.cost ?? 0) - 0.0002) < 1e-12)
    }

    @Test("The engine posts to /decisions with x-goog-api-key and no bearer token, and says it reads media")
    func engine() async throws {
        let seen = Mutex<[(String, String?, String?)]>([])
        let router = Router()
        router.post("v1beta/decisions") { request, _ in
            seen.withLock { $0.append((request.uri.path, request.headers[HTTPField.Name("x-goog-api-key")!], request.headers[.authorization])) }
            return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: ByteBuffer(string: Self.reply)))
        }
        try await Application(router: router).test(.live) { client in
            let port = try #require(client.port)
            let base = try #require(URL(string: "http://localhost:\(port)/v1beta"))
            let engine = try RemoteDecisionEngine(model: "gemini-decision-1", apiKey: "gm-test-key", provider: .gemini, baseURL: base)
            let response = try await engine.decide(SystemOneRequest(state: "A cracked phone.", questions: Self.questions))
            #expect(response.answers.count == 3 && engine.acceptsMedia)
        }
        #expect(seen.withLock { $0.first?.0 } == "/v1beta/decisions" && seen.withLock { $0.first?.1 } == "gm-test-key" && seen.withLock { $0.first?.2 } == nil)
        #expect(try !RemoteDecisionEngine(model: "jev", apiKey: "k").acceptsMedia)
        #expect(throws: BasaltError.self) { _ = try RemoteDecisionEngine(model: "gemini-decision-1", apiKey: "", provider: .gemini) }
    }

    @Test("/v1/systemone takes media, refuses it when no stage reads media, and refuses malformed media")
    func systemOneMedia() async throws {
        let png = Data([137, 80, 78, 71]).base64EncodedString()
        let body = #"{"state": "Photo of ticket 1042.", "questions": {"cracked": {"type": "noul", "instructions": "Is the screen cracked?"}}, "media": [{"kind": "image", "mime_type": "image/png", "data": "\#(png)"}]}"#
        #expect(SystemOneController.request(from: Data(body.utf8))?.media?.first?.kind == "image")
        #expect(SystemOneController.request(from: Data(body.replacingOccurrences(of: "\"image\"", with: "\"hologram\"").utf8)) == nil)
        let (_, chat) = try await PipelineSwapEdgeTests.chat()
        let server = RESTServer(orchestrator: chat, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(StubDecisionEngine())
        try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { response in
                #expect(response.status == .badRequest && String(buffer: response.body).contains("gemini-decision-1"))
            }
        }
    }

    @Test("Compose allows cloud:gemini only in decide cascades, data: media only for it, and passes it to serve as cloud:gemini/<model>")
    func compose() throws {
        let gemini = ComposeOperationsTests.base.replacingOccurrences(of: "cloud:openrouter/~typesafe/jev-latest", with: "cloud:gemini/gemini-decision-1")
        let spec = try ComposeFile.parse(gemini.replacingOccurrences(of: "api_key_env: HOME\n", with: "api_key_env: HOME\n    data: media\n"), file: "g.yaml")
        #expect(spec.serveArguments.contains("cloud:gemini/gemini-decision-1"))
        for (yaml, reason) in [
            (ComposeOperationsTests.base.replacingOccurrences(of: "api_key_env: HOME\n", with: "api_key_env: HOME\n    data: media\n"), "only for cloud:gemini"),
            (ComposeOperationsTests.with("  look:\n    source: cloud:gemini/gemini-decision-1\n    api_key_env: HOME\n", pipelines: "pipelines:\n  vision:\n    answer:\n      routes:\n        - model: look\n          fallback: chat\n        - model: chat\n"), "decisions only"),
        ] {
            do {
                _ = try ComposeFile.parse(yaml, file: "g.yaml")
                Issue.record("accepted a file that should be refused for \(reason)")
            } catch {
                #expect(error.description.contains(reason), "\(error.description)")
            }
        }
    }

    @Test("Pipeline attachments become media: a PDF as a document, an image with its type, audio as 16 kHz mono WAV")
    func pipelineMedia() throws {
        let pdf = try PipelineAttachment(kind: "pdf", data: Data("%PDF-1.4\n%%EOF".utf8), format: "pdf")
        #expect(pdf.media?.kind == "document" && pdf.media?.mimeType == "application/pdf")
        let audio = try PipelineAttachment(kind: "audio", data: AudioFixture.sine(seconds: 0.1), format: "wav")
        let wav = try #require(audio.media)
        #expect(wav.kind == "audio" && wav.mimeType == "audio/wav" && wav.data.prefix(4) == Data("RIFF".utf8) && wav.data[8 ..< 12] == Data("WAVE".utf8))
        #expect(DecisionMedia(kind: "pdf", mimeType: "application/pdf", data: Data(count: DecisionMedia.byteLimit + 1)) == nil)
    }

    static let liveKey = ProcessInfo.processInfo.environment["BASALT_E2E"] == "1" ? ProcessInfo.processInfo.environment["GEMINI_API_KEY"] : nil

    @Test("Live: gemini-decision-1 answers noul, choice and score over text, and over an image", .enabled(if: liveKey != nil, "set BASALT_E2E=1 and GEMINI_API_KEY"))
    func live() async throws {
        let engine = try RemoteDecisionEngine(model: "gemini-decision-1", apiKey: try #require(Self.liveKey), provider: .gemini)
        let text = try await engine.decide(SystemOneRequest(state: "Customer: my phone screen shattered after a drop and the touch no longer works.", questions: Self.questions))
        #expect((text.answers["cracked"]?.noul ?? 0) > 0.5 && text.answers["team"]?.choice == "tech" && text.answers["severity"]?.score != nil)
        var withImage = SystemOneRequest(state: "The attached picture.", questions: ["red": SystemOneQuestion(type: .noul, instructions: "Is the picture mostly red?")])
        withImage.media = [try #require(DecisionMedia(kind: "image", mimeType: "image/png", data: try OCRFixture.png(width: 64, height: 64)))]
        let image = try await engine.decide(withImage)
        #expect(image.answers["red"]?.noul != nil && (image.usage.cost ?? 0) > 0)
    }
}
