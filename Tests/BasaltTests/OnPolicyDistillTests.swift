import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
import Synchronization
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("on-policy distill: the student samples, a basalt teacher scores its tokens on /v1/score, per-token reverse KL under LoRA (#394)", .serialized)
struct OnPolicyDistillTests {
    static let logprobTolerance: Float = 1e-4
    static let messages = [["role": "user", "content": "Name a colour."]]

    static func teacherHost(_ family: TinyModel.Family = .qwen3Small, tokenizer: String? = nil) throws -> String {
        let teacher = Mutex(try TinyModel.make(family))
        return StubURLProtocol.host { request, body in
            guard request.url?.path == "/v1/score", let body, let score = try? JSONDecoder().decode(ScoreRequest.self, from: body),
                  var report = try? teacher.withLock({ try MLXInferenceEngine.score(score, model: $0, tokenizer: ByteTokenizer()) })
            else { return (500, [:], Data()) }
            report.tokenizer = tokenizer ?? report.tokenizer
            return (200, [:], (try? JSONEncoder().encode(report)) ?? Data())
        }
    }

    static func onPolicy(_ host: String, prompts: Int, in directory: URL) throws -> DistillRequest {
        var request = DistillTests.request(host, prompts: try DistillTests.prompts(prompts, in: directory), output: directory.appending(path: "adapter"))
        (request.onPolicy, request.teacher.maxTokens, request.teacher.temperature) = (true, 16, 1)
        return request
    }

    static func expectedTop(_ row: [Float], k: Int) -> (ids: [Int], logprobs: [Float]) {
        let peak = Double(row.max() ?? 0)
        let normalizer = peak + log(row.reduce(0.0) { $0 + exp(Double($1) - peak) })
        let ranked = row.indices.sorted { row[$0] > row[$1] }.prefix(k)
        return (Array(ranked), ranked.map { Float(Double(row[$0]) - normalizer) })
    }

    @Test("Reverse KL over the teacher's top-k plus one tail bucket matches a hand computation, ignores masked positions, and is zero when the student equals the teacher")
    func reverseKLByHand() {
        let studentLogits: [Float] = [1, 0, -1, 0.5]
        let logits = MLXArray(studentLogits + [9, -9, 3, 0]).reshaped([1, 2, 4])
        let ids = MLXArray([Int32(0), 2, 1, 3]).reshaped([1, 2, 2])
        let teacher = MLXArray([Float(log(0.5)), Float(log(0.3)), 0, 0]).reshaped([1, 2, 2])
        let mask = MLXArray([Float(1), 0]).reshaped([1, 2])
        let total = studentLogits.reduce(0.0) { $0 + exp(Double($1)) }
        let p = studentLogits.map { exp(Double($0)) / total }
        let tail = 1 - p[0] - p[2]
        let expected = p[0] * log(p[0] / 0.5) + p[2] * log(p[2] / 0.3) + tail * log(tail / 0.2)
        let kl = MLXInferenceEngine.reverseKL(logits, teacherIds: ids, teacherLogprobs: teacher, mask: mask).item(Float.self)
        #expect(abs(Double(kl) - expected) < 1e-5, "\(kl) vs \(expected)")
        let own = MLXArray([Float(log(p[0])), Float(log(p[2])), 0, 0]).reshaped([1, 2, 2])
        #expect(abs(MLXInferenceEngine.reverseKL(logits, teacherIds: ids, teacherLogprobs: own, mask: mask).item(Float.self)) < 1e-6)
    }

    @Test("POST /v1/score on a tiny engine returns, per completion token, the same top-k ids as a direct forward pass and logprobs within 1e-4, with the tokenizer fingerprint; bad requests are 400")
    func scoreEndpoint() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "tiny-qwen3")).makeRouter())
        let completion = [10, 20, 30, 40, 50]
        let prompt = try ByteTokenizer().applyChatTemplate(messages: Self.messages, tools: nil, additionalContext: nil)
        let model = try TinyModel.make(.qwen3)
        let logits = model(MLXArray((prompt + completion).map(Int32.init))[.newAxis], cache: nil).asType(.float32)
        let rows = (0 ..< completion.count).map { logits[0, prompt.count - 1 + $0].asArray(Float.self) }
        func post(_ body: String) async throws -> (HTTPResponse.Status, Data) {
            try await app.test(.router) { client in
                try await client.execute(uri: "/v1/score", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) {
                    ($0.status, Data(String(buffer: $0.body).utf8))
                }
            }
        }
        let (status, data) = try await post(#"{"messages": [{"role": "user", "content": "Name a colour."}], "completion": [10, 20, 30, 40, 50], "top_logprobs": 5}"#)
        #expect(status == .ok, "\(String(decoding: data, as: UTF8.self))")
        let report = try JSONDecoder().decode(ScoreReport.self, from: data)
        #expect(report.tokenizer == MLXInferenceEngine.tokenizerFingerprint(ByteTokenizer()) && report.promptTokens == prompt.count)
        #expect(report.positions.count == completion.count)
        for (position, row) in zip(report.positions, rows) {
            let expected = Self.expectedTop(row, k: 5)
            #expect(position.ids == expected.ids)
            #expect(zip(position.logprobs, expected.logprobs).allSatisfy { abs($0 - $1) < Self.logprobTolerance }, "\(position.logprobs) vs \(expected.logprobs)")
        }
        for bad in [
            #"{"messages": [{"role": "user", "content": "x"}], "completion": [1], "top_logprobs": 21}"#,
            #"{"messages": [{"role": "user", "content": "x"}], "completion": [999], "top_logprobs": 5}"#,
            #"{"messages": [], "completion": [1], "top_logprobs": 5}"#, #"{"completion": [1]}"#,
        ] {
            #expect(try await post(bad).0 == .badRequest, "\(bad)")
        }
        #expect(await orchestrator.gate.inFlight == 0)
    }

    @Test("A 16-prompt greedy on-policy run on a tiny model lowers the held-out reverse KL to a different tiny teacher, sends the student's token ids to /v1/score, saves a loadable adapter and leaves the base weights bit-identical")
    func endToEnd() async throws {
        let directory = try DistillTests.scratch()
        let host = try Self.teacherHost()
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        var request = try Self.onPolicy(host, prompts: 16, in: directory)
        (request.teacher.temperature, request.iterations, request.learningRate) = (0, 60, 1e-2)
        let report = try await engine.distill(request, session: URLSession(configuration: StubURLProtocol.configuration), home: directory, environment: [:])
        #expect(report.kept == 12 && report.heldOut == 4 && report.cached == 0)
        let klBefore = try #require(report.reverseKLBefore), klAfter = try #require(report.reverseKLAfter)
        #expect(klAfter < klBefore, "\(klBefore) -> \(klAfter), losses \(report.losses)")
        #expect(!report.losses.isEmpty && report.losses.allSatisfy(\.isFinite))
        let calls = StubURLProtocol.requests(for: host)
        #expect(report.teacherCalls == request.iterations * request.batchSize + 2 * 4 && calls.count == report.teacherCalls)
        let sent = try calls.map { try JSONDecoder().decode(ScoreRequest.self, from: try #require($0.1)) }
        #expect(sent.allSatisfy { $0.topLogprobs == MLXInferenceEngine.onPolicyTopK && !$0.completion.isEmpty && $0.completion.count <= 16 })
        _ = try LoRAContainer.from(directory: directory.appending(path: "adapter"))
        if let overlay = report.verdict?.overlay { try await engine.undo(overlay.id) }
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("A teacher with another tokenizer is refused before any training step, with an error that says so; nothing is written and the weights are unchanged")
    func tokenizerMismatch() async throws {
        let directory = try DistillTests.scratch()
        let host = try Self.teacherHost(tokenizer: "0123456789abcdef")
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let request = try Self.onPolicy(host, prompts: 4, in: directory)
        let error = await #expect(throws: BasaltError.self) {
            try await engine.distill(request, session: URLSession(configuration: StubURLProtocol.configuration), home: directory, environment: [:])
        }
        #expect(String(describing: error).contains("tokenizer"))
        #expect(!FileManager.default.fileExists(atPath: directory.appending(path: "adapter").path))
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("Scores with the wrong position count, k, or ids outside the student's vocabulary are refused; an anthropic teacher is refused for onPolicy")
    func malformedScores() async throws {
        let plan = OnPolicyPlan(train: [], heldOut: [], tokenizer: "t", vocabulary: 256, tooLong: 0)
        let good = TopLogprobs(ids: Array(0 ..< 20), logprobs: Array(repeating: -3, count: 20))
        #expect(throws: Never.self) { try MLXInferenceEngine.checked(ScoreReport(tokenizer: "t", promptTokens: 3, positions: [good]), completion: 1, plan: plan) }
        let outside = TopLogprobs(ids: Array(240 ..< 260), logprobs: good.logprobs)
        let short = TopLogprobs(ids: [1], logprobs: [-1])
        for positions in [[outside], [short], [good, good]] {
            #expect(throws: BasaltError.self) { try MLXInferenceEngine.checked(ScoreReport(tokenizer: "t", promptTokens: 3, positions: positions), completion: 1, plan: plan) }
        }
        let directory = try DistillTests.scratch()
        var request = try Self.onPolicy("unused.test", prompts: 4, in: directory)
        request.teacher.provider = .anthropic
        let engine = try await TinyModel.engine(.qwen3)
        await #expect(throws: BasaltError.self) { try await engine.distill(request, session: URLSession(configuration: StubURLProtocol.configuration), home: directory, environment: [:]) }
    }
}
