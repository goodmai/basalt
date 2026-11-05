import Testing
import Foundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

final class EmbeddingCalls: Sendable {
    let inputs = Mutex<[[EmbeddingInput]]>([])
    let events = Mutex<[String]>([])
}

struct StubEmbeddingEngine: EmbeddingEngine {
    var models: [EmbeddingModelInfo] = [
        EmbeddingModelInfo(id: "org/stub-embed", task: .embedding),
        EmbeddingModelInfo(id: "org/stub-rerank", task: .rerank),
    ]
    var calls = EmbeddingCalls()
    var betweenBatches: (paused: AsyncStream<Void>.Continuation, resume: AsyncStream<Void>)?

    func pauseBetweenBatches() async {
        guard let betweenBatches else { return }
        calls.events.withLock { $0.append("micro-batch 1") }
        betweenBatches.paused.yield()
        for await _ in betweenBatches.resume { break }
        calls.events.withLock { $0.append("micro-batch 2") }
    }

    func embed(_ inputs: [EmbeddingInput], dimensions: Int?, yieldingTo gate: GenerationGate?) async throws(BasaltError) -> Embeddings {
        calls.inputs.withLock { $0.append(inputs) }
        await pauseBetweenBatches()
        let vectors = inputs.indices.map { row -> [Float] in
            let raw = (0 ..< (dimensions ?? 8)).map { Float($0 + row + 1) }
            let norm = raw.reduce(0) { $0 + $1 * $1 }.squareRoot()
            return raw.map { $0 / norm }
        }
        return Embeddings(vectors: vectors, promptTokens: inputs.count * 3)
    }

    func rerank(query: String, documents: [String], yieldingTo gate: GenerationGate?) async throws(BasaltError) -> [Double] {
        await pauseBetweenBatches()
        return documents.map { Double($0.count) }
    }
}

@Suite("/v1/embeddings and /v1/rerank on a stub engine (#178)")
struct EmbeddingsAPITests {

    private func app(_ engine: StubEmbeddingEngine? = StubEmbeddingEngine(), orchestrator: ModelOrchestratorActor? = nil) async throws
        -> Application<RouterResponder<BasicRequestContext>> {
        let orch = orchestrator ?? ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
        let config = try ServerConfig(modelPath: "org/mock-model")
        return Application(router: RESTServer(orchestrator: orch, config: config, embeddings: engine).makeRouter())
    }

    private func post(_ client: some TestClientProtocol, _ uri: String, _ body: String) async throws -> (HTTPResponse.Status, [String: Any]) {
        try await client.execute(uri: uri, method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { res in
            (res.status, (try? JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any]) ?? [:])
        }
    }

    private func floats(_ base64: String) throws -> [Float] {
        let bytes = [UInt8](try #require(Data(base64Encoded: base64)))
        return stride(from: 0, to: bytes.count, by: 4).map { offset in
            Float(bitPattern: (0 ..< 4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) })
        }
    }

    @Test("Float embeddings come back in the OpenAI list shape: one L2-normalized vector per input, in input order, with usage")
    func floatEmbeddings() async throws {
        try await app().test(.router) { (client) async throws in
            let (status, body) = try await post(client, "/v1/embeddings", #"{"model": "x", "input": ["a", "b", "c"]}"#)
            #expect(status == .ok)
            #expect(body["object"] as? String == "list")
            #expect(body["model"] as? String == "org/stub-embed")
            let data = try #require(body["data"] as? [[String: Any]])
            #expect(data.map { $0["index"] as? Int } == [0, 1, 2])
            #expect(data.allSatisfy { $0["object"] as? String == "embedding" })
            for item in data {
                let vector = try #require(item["embedding"] as? [Double])
                #expect(vector.count == 8)
                #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 1e-5)
            }
            let usage = try #require(body["usage"] as? [String: Any])
            #expect(usage["prompt_tokens"] as? Int == 9)
            #expect(usage["total_tokens"] as? Int == 9)
        }
    }

    @Test("encoding_format base64 carries the same float32 vectors, little-endian")
    func base64Embeddings() async throws {
        try await app().test(.router) { (client) async throws in
            let (_, plain) = try await post(client, "/v1/embeddings", #"{"input": ["a", "b"]}"#)
            let (status, encoded) = try await post(client, "/v1/embeddings", #"{"input": ["a", "b"], "encoding_format": "base64"}"#)
            #expect(status == .ok)
            let expected = try #require(plain["data"] as? [[String: Any]]).map { ($0["embedding"] as? [Double])?.map(Float.init) }
            let items = try #require(encoded["data"] as? [[String: Any]])
            let decoded = try items.map { try floats(try #require($0["embedding"] as? String)) }
            #expect(decoded.count == 2)
            #expect(decoded.map { Optional($0) } == expected)
            let (bad, _) = try await post(client, "/v1/embeddings", #"{"input": "a", "encoding_format": "int8"}"#)
            #expect(bad == .badRequest)
        }
    }

    @Test("A string, a token array and an array of token arrays reach the engine as texts and token inputs")
    func inputShapes() async throws {
        let engine = StubEmbeddingEngine()
        try await app(engine).test(.router) { (client) async throws in
            for body in [#"{"input": "hi"}"#, #"{"input": [1, 2, 3]}"#, #"{"input": [[1, 2], [3]]}"#] {
                #expect(try await post(client, "/v1/embeddings", body).0 == .ok, "\(body)")
            }
            for body in [#"{"input": []}"#, #"{"input": [""]}"#, #"{"input": [[]]}"#, #"{"input": 5}"#, #"{}"#] {
                #expect(try await post(client, "/v1/embeddings", body).0 == .badRequest, "\(body)")
            }
        }
        #expect(engine.calls.inputs.withLock { $0 } == [[.text("hi")], [.tokens([1, 2, 3])], [.tokens([1, 2]), .tokens([3])]])
    }

    @Test("dimensions is refused with 400 on a model without Matryoshka training and truncates on one with it")
    func dimensions() async throws {
        try await app().test(.router) { (client) async throws in
            let (status, body) = try await post(client, "/v1/embeddings", #"{"input": "a", "dimensions": 4}"#)
            #expect(status == .badRequest)
            #expect(((body["error"] as? [String: Any])?["message"] as? String)?.contains("Matryoshka") == true)
        }
        var matryoshka = StubEmbeddingEngine()
        matryoshka.models = [EmbeddingModelInfo(id: "org/mrl", task: .embedding, truncatesDimensions: true)]
        try await app(matryoshka).test(.router) { (client) async throws in
            let (status, body) = try await post(client, "/v1/embeddings", #"{"input": "a", "dimensions": 4}"#)
            #expect(status == .ok)
            #expect(((body["data"] as? [[String: Any]])?.first?["embedding"] as? [Double])?.count == 4)
            #expect(try await post(client, "/v1/embeddings", #"{"input": "a", "dimensions": 0}"#).0 == .badRequest)
        }
    }

    @Test("More inputs or documents than the batch cap get 413 before the engine runs")
    func batchCap() async throws {
        let engine = StubEmbeddingEngine()
        let inputs = Array(repeating: "\"x\"", count: EmbeddingLimits.maxInputs + 1).joined(separator: ",")
        let documents = Array(repeating: "\"x\"", count: EmbeddingLimits.maxDocuments + 1).joined(separator: ",")
        try await app(engine).test(.router) { (client) async throws in
            #expect(try await post(client, "/v1/embeddings", "{\"input\": [\(inputs)]}").0 == .contentTooLarge)
            #expect(try await post(client, "/v1/rerank", "{\"query\": \"q\", \"documents\": [\(documents)]}").0 == .contentTooLarge)
        }
        #expect(engine.calls.inputs.withLock { $0 }.isEmpty)
    }

    @Test("Rerank returns results sorted by score, index pointing into the input order, top_n and return_documents honored")
    func rerank() async throws {
        try await app().test(.router) { (client) async throws in
            let (status, body) = try await post(client, "/v1/rerank", #"{"model": "r", "query": "q", "documents": ["ccc", "a", "eeeee", "bb"]}"#)
            #expect(status == .ok)
            #expect(body["model"] as? String == "org/stub-rerank")
            let results = try #require(body["results"] as? [[String: Any]])
            #expect(results.map { $0["index"] as? Int } == [2, 0, 3, 1])
            #expect(results.map { $0["relevance_score"] as? Double } == [5, 3, 2, 1])
            #expect(results.allSatisfy { $0["document"] == nil })

            let (_, top) = try await post(client, "/v1/rerank",
                #"{"query": "q", "documents": [{"text": "ccc"}, {"text": "a"}, {"text": "eeeee"}], "top_n": 2, "return_documents": true}"#)
            let topResults = try #require(top["results"] as? [[String: Any]])
            #expect(topResults.map { $0["index"] as? Int } == [2, 0])
            #expect(topResults.map { ($0["document"] as? [String: Any])?["text"] as? String } == ["eeeee", "ccc"])

            for bad in [#"{"query": "", "documents": ["a"]}"#, #"{"query": "q", "documents": []}"#, #"{"query": "q", "documents": ["a"], "top_n": 0}"#, #"{"documents": ["a"]}"#] {
                #expect(try await post(client, "/v1/rerank", bad).0 == .badRequest, "\(bad)")
            }
        }
    }

    @Test("Without --embedding-model or --rerank-model both routes answer 404 with the flag to pass")
    func notLoaded() async throws {
        try await app(nil).test(.router) { (client) async throws in
            let (embedStatus, embedBody) = try await post(client, "/v1/embeddings", #"{"input": "a"}"#)
            #expect(embedStatus == .notFound)
            #expect(((embedBody["error"] as? [String: Any])?["message"] as? String)?.contains("--embedding-model") == true)
            #expect(try await post(client, "/v1/rerank", #"{"query": "q", "documents": ["a"]}"#).0 == .notFound)
        }
    }

    @Test("/v1/models lists the embedder and the reranker with their task")
    func modelsList() async throws {
        try await app().test(.router) { (client) async throws in
            let body = try await client.execute(uri: "/v1/models", method: .get) { res in
                try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: res.body).utf8)) as? [String: Any])
            }
            let data = try #require(body["data"] as? [[String: Any]])
            let tasks = Dictionary(uniqueKeysWithValues: data.compactMap { entry in
                (entry["task"] as? String).map { (entry["id"] as? String ?? "", $0) }
            })
            #expect(tasks == ["org/stub-embed": "embedding", "org/stub-rerank": "rerank"])
        }
    }

    @Test("A chat checkpoint passed as --embedding-model is refused at load with the reason, before any weight is read")
    func chatCheckpointRefused() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        do {
            _ = try await EmbeddingService(embeddingModel: (id: "mlx-community/Qwen3-4B-4bit", directory: dir), rerankModel: nil)
            Issue.record("a chat checkpoint loaded as an embedder")
        } catch {
            guard case .modelConfigurationInvalid(_, let reason) = error else { Issue.record("\(error)"); return }
            #expect(reason.contains("chat checkpoint"))
        }
    }

    @Test("Embedder validation accepts encoders and embedding checkpoints and refuses chat, reranker and ColBERT ones",
          arguments: [
            (#"{"model_type": "bert", "architectures": ["BertModel"]}"#, false, "BAAI/bge-small-en-v1.5", true),
            (#"{"model_type": "nomic_bert"}"#, false, "nomic-ai/nomic-embed-text-v1.5", true),
            (#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#, false, "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", true),
            (#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#, true, "/models/snapshot", true),
            (#"{"model_type": "lfm2", "mlx": {"head": "embedding"}}"#, false, "mlx-community/LFM2.5-Embedding-350M-4bit", true),
            (#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#, false, "mlx-community/Qwen3-4B-4bit", false),
            (#"{"model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"]}"#, true, "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit", false),
            (#"{"model_type": "gemma3_text"}"#, false, "mlx-community/gemma-3-1b-it-qat-4bit", false),
            (#"{"model_type": "lfm2", "architectures": ["Lfm2ForCausalLM"]}"#, false, "LiquidAI/LFM2-1.2B", false),
            (#"{"model_type": "lfm2", "mlx": {"head": "colbert"}}"#, false, "mlx-community/LFM2.5-ColBERT-350M-4bit", false),
            (#"{"model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"]}"#, false, "BAAI/bge-reranker-v2-m3", false),
          ])
    func embedderValidation(config: String, layout: Bool, name: String, accepted: Bool) throws {
        let json = try #require(try JSONSerialization.jsonObject(with: Data(config.utf8)) as? [String: Any])
        let refusal = EmbeddingService.embedderRefusal(config: json, sentenceTransformersLayout: layout, name: name)
        #expect((refusal == nil) == accepted, "\(name): \(refusal ?? "accepted")")
        if let refusal { #expect(refusal.contains(name)) }
    }

    @Test("Micro-batches group inputs by length under the token and count caps and cover every input once")
    func microBatches() {
        let lengths = [500, 10, 9000, 20, 30, 4000, 4000]
        let batches = EmbeddingService.microBatches(lengths, maxTokens: 8192, maxCount: 2)
        #expect(batches.flatMap { $0 }.sorted() == Array(lengths.indices))
        #expect(batches.allSatisfy { $0.count <= 2 && ($0.count == 1 || ($0.map { lengths[$0] }.max() ?? 0) * $0.count <= 8192) })
        #expect(batches.first == [1, 3])
        #expect(batches.last == [2])
    }
}

extension EmbeddingsAPITests {
    @Test("A generation runs to completion between two micro-batches of one long embeddings request (#233)", .timeLimit(.minutes(1)))
    func generationRunsBetweenMicroBatches() async throws {
        try await generationRunsBetweenBatches(of: "/v1/embeddings", #"{"input": ["a", "b"]}"#)
    }

    @Test("A chat request completes between two /v1/rerank chunks instead of waiting for the whole document list (#355)", .timeLimit(.minutes(1)))
    func generationRunsBetweenRerankChunks() async throws {
        try await generationRunsBetweenBatches(of: "/v1/rerank", #"{"query": "q", "documents": ["a", "bb"]}"#)
    }

    private func generationRunsBetweenBatches(of uri: String, _ body: String) async throws {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
        try await orchestrator.loadModel(path: "mock")
        let (paused, pause) = AsyncStream<Void>.makeStream()
        let (resumed, resume) = AsyncStream<Void>.makeStream()
        var engine = StubEmbeddingEngine()
        engine.betweenBatches = (pause, resumed)
        let calls = engine.calls
        try await app(engine, orchestrator: orchestrator).test(.router) { (client) async throws in
            async let embedding = post(client, uri, body)
            for await _ in paused { break }
            let response = try await orchestrator.generate(request: GenerationRequest(prompt: "hi", maxTokens: 4))
            #expect(response.completionTokens > 0)
            calls.events.withLock { $0.append("generation") }
            resume.yield()
            #expect((try await embedding).0 == .ok)
        }
        #expect(calls.events.withLock { $0 } == ["micro-batch 1", "generation", "micro-batch 2"])
        #expect(await orchestrator.gate.checkInvariants(idle: true).isEmpty)
    }
}

@Suite("Embedding micro-batches wait for generation to go idle (#233)")
struct GenerationIdleTests {
    @Test("idle(within:) is true at once on an idle gate, false at the limit while a request runs, and true again once it is released")
    func idleWithin() async throws {
        let clock = Mutex(Duration.zero)
        let gate = GenerationGate(now: { clock.withLock { $0 } }, sleep: { step in clock.withLock { $0 += step } })
        #expect(await gate.idle(within: .seconds(1)))
        let id = try await gate.admit()
        #expect(await !gate.idle(within: .seconds(1)))
        #expect(clock.withLock { $0 } >= .seconds(1))
        await gate.release(id)
        #expect(await gate.idle(within: .seconds(1)))
    }

    @Test("Rerank asks the gate before each micro-batch only when a chat model is loaded, and splits only while generation stays busy (#355)")
    func rerankOptions() async throws {
        #expect(EmbeddingService.rerankOptions(yieldingTo: nil).beforeMicroBatch == nil)
        let clock = Mutex(Duration.zero)
        let gate = GenerationGate(now: { clock.withLock { $0 } }, sleep: { step in clock.withLock { $0 += step } })
        let split = try #require(EmbeddingService.rerankOptions(yieldingTo: gate).beforeMicroBatch)
        #expect(await !split())
        let id = try await gate.admit()
        #expect(await split())
        await gate.release(id)
        #expect(await !split())
    }

    @Test("A cancelled wait returns false instead of spinning until the limit")
    func cancelledWait() async throws {
        let gate = GenerationGate(now: { .zero }, sleep: { _ in throw CancellationError() })
        _ = try await gate.admit()
        #expect(await !gate.idle(within: .seconds(1)))
    }
}
