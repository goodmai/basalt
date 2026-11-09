import Foundation
import Testing
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

@Suite("Core AI decisions (Laya) on the Neural Engine against the PyTorch reference (#234)", .serialized)
struct CoreAIDecisionRealModelTests {
    struct Case: Decodable {
        struct Expected: Decodable {
            let probabilities: [String: Double]
            let ids: [Int]
            let markers: [Int]
        }

        let state: String
        let questions: [String: SystemOneQuestion]
        let answers: [String: Expected]
    }

    static let env = ProcessInfo.processInfo.environment
    static let bundle = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_COREAI_DECISION_BUNDLE"] : nil
    static let reference = env["BASALT_E2E_COREAI_DECISION_REFERENCE"]
    static let tolerance = 1e-2

    static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
    }

    @Test("Token ids and markers equal the reference exactly; calibrated probabilities stay within the fp16 tolerance of fp32 PyTorch; the worst gap is logged",
          .enabled(if: bundle != nil && reference != nil, "set BASALT_E2E=1, BASALT_E2E_COREAI_DECISION_BUNDLE (export_laya.py) and BASALT_E2E_COREAI_DECISION_REFERENCE (reference_laya.py)"))
    func matchesReference() async throws {
        let engine = try await CoreAIDecisionEngine(directory: URL(fileURLWithPath: try #require(Self.bundle)))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: try #require(Self.reference))))
        var worst = 0.0
        for item in cases {
            for (name, expected) in item.answers {
                let tokenize = { (text: String) in engine.tokenizer.encode(text: text, addSpecialTokens: false) }
                let encoded = try engine.bundle.encode(try #require(item.questions[name]), state: tokenize(CoreAIDecisionBundle.clean(item.state)), tokenize: tokenize)
                #expect(encoded.ids == expected.ids && encoded.markers == expected.markers, "question \(name) tokenizes differently")
            }
            let response = try await engine.decide(SystemOneRequest(state: item.state, questions: item.questions))
            for (name, expected) in item.answers {
                let got = try #require(response.answers[name])
                let probabilities = got.probabilities ?? got.noul.map { ["false": 1 - $0, "true": $0] } ?? [:]
                let gap = expected.probabilities.map { abs((probabilities[$0.key] ?? -1) - $0.value) }.max() ?? 0
                print("RESULT laya-ane question \(name) (\(expected.ids.count) tokens): gap \(String(format: "%.2e", gap))")
                worst = max(worst, gap)
            }
        }
        print("RESULT laya-ane worst calibrated probability gap to the fp32 reference \(String(format: "%.2e", worst)) over \(cases.count) requests")
        #expect(worst < Self.tolerance, "worst probability gap \(worst)")
    }

    @Test("Latency per function on the Neural Engine: p50 and p95 over 50 calls after warm-up, logged",
          .enabled(if: bundle != nil, "set BASALT_E2E=1 and BASALT_E2E_COREAI_DECISION_BUNDLE"))
    func latency() async throws {
        let engine = try await CoreAIDecisionEngine(directory: URL(fileURLWithPath: try #require(Self.bundle)))
        for function in engine.bundle.functions {
            let rows = Array(repeating: Array(repeating: 1000, count: function.length), count: function.batch)
            let kinds = Array(repeating: SystemOneKind.choice, count: function.batch)
            for _ in 0 ..< 3 { _ = try await engine.run(function, rows: rows, kinds: kinds) }
            var times: [Double] = []
            for _ in 0 ..< 50 {
                let start = ContinuousClock.now
                _ = try await engine.run(function, rows: rows, kinds: kinds)
                times.append(Self.milliseconds(ContinuousClock.now - start))
            }
            times.sort()
            print("RESULT laya-ane \(function.name): p50 \(String(format: "%.2f", times[25])) ms, p95 \(String(format: "%.2f", times[47])) ms, \(String(format: "%.2f", times[25] / Double(function.batch))) ms per question")
        }
    }

    @Test("/v1/systemone answers through the REST router with this engine, and the compute plan places it on the Neural Engine as an encoder beside decode",
          .enabled(if: bundle != nil, "set BASALT_E2E=1 and BASALT_E2E_COREAI_DECISION_BUNDLE"))
    func contract() async throws {
        let path = try #require(Self.bundle)
        let engine = try await CoreAIDecisionEngine(directory: URL(fileURLWithPath: path))
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), maxTokens: 65536, telemetry: TelemetryTracker())
        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).withDecisions(engine).makeRouter())
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/systemone", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: SystemOneAPITests.body)) { response in
                #expect(response.status == .ok)
                let json = try #require(try JSONSerialization.jsonObject(with: Data(String(buffer: response.body).utf8)) as? [String: Any])
                let answers = try #require(json["answers"] as? [String: [String: Any]])
                #expect(answers["color"]?["choice"] as? String == "red")
                #expect((answers["rain"]?["noul"] as? Double).map { (0 ... 1).contains($0) } == true)
                #expect(json["model"] as? String == engine.modelID)
                #expect((json["usage"] as? [String: Any])?["output_decoding_tokens"] as? Int == 0)
            }
        }
        let workloads = try await ServeCommand.parse(["--decision-model", path]).sidecarWorkloads(embeddings: nil, decisions: engine)
        #expect(workloads.map(\.unit) == [.ane] && workloads.map(\.kind) == [.encoder])
        #expect((workloads.first?.residentBytes ?? 0) > 500_000_000)
        #expect(ComputePlan.admit(workloads, budgetBytes: 0).placements.first?.concurrentWithDecode == true && engine.runsBesideDecode)
    }
}
