import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import Synchronization
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

private final class MockCorpusTransport: CorpusHTTPTransport {
    private let handler: @Sendable (URLRequest) throws -> (Int, Data)
    let requests = Mutex<[URLRequest]>([])

    init(_ handler: @escaping @Sendable (URLRequest) throws -> (Int, Data)) {
        self.handler = handler
    }

    func fetch(_ request: URLRequest) async throws -> (Int, Data) {
        requests.withLock { $0.append(request) }
        return try handler(request)
    }
}

private final class AdapterProbeTarget: AnswerTarget {
    let label: String
    let seenAdapters = Mutex<[String?]>([])

    init(label: String) {
        self.label = label
    }

    func answer(_ request: GenerationRequest) async throws(BasaltError) -> AnswerReply {
        seenAdapters.withLock { $0.append(request.adapter) }
        return AnswerReply(text: "\(label):\(request.adapter ?? "base")", model: label, tokens: 8)
    }
}

@Suite("Nemotron presets, template-leakage sanitization, and multi-schema row normalization (#510)")
struct NemotronCorpusMapperTests {
    @Test("All 6 Nemotron presets resolve canonical nvidia Hugging Face repositories, splits, and CorpusLocation parsing",
          arguments: NemotronPreset.allCases)
    func presetResolution(preset: NemotronPreset) throws {
        #expect(preset.repoID.hasPrefix("nvidia/"))
        #expect(!preset.defaultSplit.isEmpty)
        #expect(!preset.licenseSummary.isEmpty)
        let spec = CorpusSourceSpec(source: preset.rawValue)
        #expect(spec.location == .preset(preset))
        #expect(spec.location.datasetRepo == preset.repoID)
        #expect(spec.resolvedSplit == preset.defaultSplit)
        try spec.validate()
    }

    @Test("Foreign Llama and ChatML special tokens are stripped and <think> spans split cleanly across Nemotron schemas")
    func sanitizationAndSchemaNormalization() throws {
        let leaked = "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n<think>\nStep 1: check parity.\n</think>\nThe answer is 42.<|eot_id|>"
        let split = NemotronRowMapper.splitThinking(leaked)
        #expect(split.reasoning == "Step 1: check parity.")
        #expect(split.answer == "The answer is 42.")

        let qwenMarkers = ThinkingMarkers(open: ["<think>"], close: ["</think>"])
        let withThought = NemotronRowMapper.formatTarget(reasoning: split.reasoning, answer: split.answer, includeReasoning: true, markers: qwenMarkers)
        let withoutThought = NemotronRowMapper.formatTarget(reasoning: split.reasoning, answer: split.answer, includeReasoning: false, markers: qwenMarkers)
        #expect(withThought == "<think>\nStep 1: check parity.\n</think>\n\nThe answer is 42.")
        #expect(withoutThought == "The answer is 42.")

        let postV2Row: [String: Any] = [
            "messages": [
                ["role": "system", "content": "Be concise."],
                ["role": "user", "content": "Compute 6 * 7"],
                ["role": "assistant", "content": "<think>6 * 7 = 42</think>\n42", "tool_calls": [["name": "calc", "args": ["x": 42]]]],
            ],
            "pass_rate": 1.0,
        ]
        let postExample = try #require(NemotronRowMapper.normalizeRow(postV2Row))
        #expect(postExample.contextMessages.count == 2)
        #expect(postExample.reasoning == "6 * 7 = 42")
        #expect(postExample.answer.contains("42") && postExample.answer.contains("<tool_call>"))

        let codeRow: [String: Any] = [
            "input": "Write fib(n) in Swift.",
            "output": "<think>Use iterative tuple swap.</think>\nfunc fib(_ n: Int) -> Int { n }",
            "pass_rate": 1.0,
        ]
        let lowCodeRow: [String: Any] = [
            "input": "Broken solution",
            "output": "func bad() {}",
            "pass_rate": 0.25,
        ]
        let mathRow: [String: Any] = [
            "problem": "Solve x + 3 = 10",
            "generated_solution": "<think>x = 10 - 3 = 7</think>\n\\boxed{7}",
            "pass_rate": 1.0,
        ]
        let personaRow: [String: Any] = [
            "persona": "A pragmatic Swift systems engineer in Zurich.",
            "professional_persona": "Writes allocation-free value cores with strict concurrency.",
            "pass_rate": 1.0,
        ]
        var spec = CorpusSourceSpec(source: "nemotron:code", maxRows: 10)
        spec.minPassRate = 0.8
        let prepared = try NemotronRowMapper.prepare(
            rows: [postV2Row, codeRow, lowCodeRow, mathRow, personaRow],
            spec: spec,
            heldOutFraction: 0.25
        )
        #expect(prepared.rejected["low_pass_rate"] == 1)
        #expect(prepared.train.count == 3 && prepared.heldOut.count == 1)
    }
}

@Suite("HFDatasetStreamer pagination, auth headers, gated 403 diagnostics, and 0700 local shard cache (#510)")
struct HFDatasetStreamerTests {
    @Test("Streams /splits + paginated /rows with Bearer token and serves subsequent runs from 0700 offline cache")
    func remoteStreamAndOfflineCacheReplay() async throws {
        let scratchHome = FileManager.default.temporaryDirectory.appending(path: "basalt-hf-stream-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratchHome) }

        let transport = MockCorpusTransport { request in
            let url = request.url?.absoluteString ?? ""
            if url.contains("/splits?") {
                let splitsJSON = #"{"splits":[{"dataset":"nvidia/OpenCodeReasoning","config":"default","split":"split_0"}]}"#
                return (200, Data(splitsJSON.utf8))
            }
            let rowsJSON = """
            {"rows":[
              {"row":{"input":"Task A","output":"<think>tA</think>\\nAnswer A","pass_rate":1.0}},
              {"row":{"input":"Task B","output":"<think>tB</think>\\nAnswer B","pass_rate":1.0}},
              {"row":{"input":"Task C","output":"<think>tC</think>\\nAnswer C","pass_rate":1.0}},
              {"row":{"input":"Task D","output":"<think>tD</think>\\nAnswer D","pass_rate":1.0}}
            ]}
            """
            return (200, Data(rowsJSON.utf8))
        }

        var spec = CorpusSourceSpec(source: "nemotron:code", maxRows: 4)
        spec.subset = nil
        let first = try await HFDatasetStreamer.load(
            spec,
            transport: transport,
            home: scratchHome,
            environment: ["HF_TOKEN": "test_token_value"]
        )
        #expect(first.rowCount == 4)
        #expect(first.repo == "nvidia/OpenCodeReasoning")
        #expect(first.subset == "default" && first.split == "split_0")
        let cacheURL = try #require(first.cacheFile)
        #expect(FileManager.default.fileExists(atPath: cacheURL.path))
        let recorded = transport.requests.withLock { $0 }
        #expect(recorded.count == 2)
        #expect(recorded.allSatisfy { $0.value(forHTTPHeaderField: "Authorization") == "Bearer test_token_value" })

        let countBeforeReplay = transport.requests.withLock { $0.count }
        let second = try await HFDatasetStreamer.load(spec, transport: transport, home: scratchHome, environment: [:])
        #expect(second.rowCount == 4)
        #expect(transport.requests.withLock { $0.count } == countBeforeReplay)

        let gatedTransport = MockCorpusTransport { _ in (403, Data("gated dataset".utf8)) }
        let gatedSpec = CorpusSourceSpec(source: "nemotron:post-v2", subset: "default", split: "chat", maxRows: 4)
        let err = await #expect(throws: BasaltError.self) {
            try await HFDatasetStreamer.load(gatedSpec, transport: gatedTransport, home: scratchHome, environment: [:])
        }
        #expect(err?.localizedDescription.contains("HF_TOKEN") == true)
    }
}

@Suite("Corpus SFT training across LoRA, PHM, and OFT plus Compose v2 multi-adapter training and routing (#510)", .serialized)
struct CorpusTrainAndComposeAdapterTests {
    private static func writeSampleCorpus(to directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(component: "nemotron_sample.jsonl")
        let lines = (0 ..< 8).map { idx in
            #"{"messages":[{"role":"user","content":"Question \#(idx): colour?"},{"role":"assistant","content":"<think>pick blue</think>\nThe colour is blue \#(idx % 2)."}]}"#
        }.joined(separator: "\n")
        try Data(lines.utf8).write(to: file)
        return file
    }

    @Test("trainCorpus trains LoRA, PHM, and OFT adapters on Nemotron JSONL, lowers loss, preserves base weights, and writes provenance",
          arguments: [
            AdapterMethod.lora(rank: 2, scale: 8.0),
            AdapterMethod.phm(hypercomplexDim: 4, factorRank: 1, sharedBasis: true, scale: 8.0),
            AdapterMethod.oft(blockSize: 8, parameterization: .cayley, blockShared: false),
          ])
    func trainCorpusAcrossAdapterMethods(method: AdapterMethod) async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "basalt-corpus-train-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let jsonl = try Self.writeSampleCorpus(to: scratch)
        let outDir = scratch.appending(component: "trained-adapter")

        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let beforeWeights = await DistillTests.baseWeights(container)

        var req = CorpusTrainRequest(corpus: CorpusSourceSpec(source: jsonl.path, maxRows: 8, includeReasoning: true))
        req.method = method
        req.iterations = 6
        req.batchSize = 2
        req.learningRate = 3e-3
        req.maxSeqLen = 256
        req.checkpointEvery = 0
        req.output = outDir.path

        let report = try await engine.trainCorpus(req, lease: nil)
        #expect(report.keptRows >= 4 && report.heldOutRows >= 1)
        let beforeNLL = try #require(report.nllBefore)
        let afterNLL = try #require(report.nllAfter)
        #expect(afterNLL < beforeNLL)
        #expect(await DistillTests.baseWeights(container) == beforeWeights)

        let provURL = outDir.appending(component: "adapter_provenance.json")
        let prov = try JSONDecoder().decode(AdapterProvenance.self, from: Data(contentsOf: provURL))
        #expect(prov.method == method)
        #expect(prov.trainRows == report.keptRows)

        try await engine.loadAdapters(["specialist": outDir])
        #expect(await engine.adapterNames == ["specialist"])
        try await engine.useAdapter("specialist")
        #expect(await engine.activeAdapter == "specialist")
        try await engine.useAdapter(nil)
        #expect(await engine.activeAdapter == nil)
    }

    private static func writeComposeFixture(to scratch: URL) throws -> URL {
        _ = try writeSampleCorpus(to: scratch)
        let yamlURL = scratch.appending(component: "basalt.yaml")
        let yaml = """
        version: 2
        name: nemotron-specialists
        models:
          qwen:
            source: ./tiny-qwen
        adapters:
          code_oft:
            base: qwen
            method: oft
            block_size: 8
            parameterization: cayley
            corpus:
              source: ./nemotron_sample.jsonl
              max_rows: 8
            iterations: 4
            batch_size: 2
            learning_rate: 0.003
            max_seq_len: 256
            checkpoint_every: 0
          math_phm:
            base: qwen
            method: phm
            hypercomplex_dim: 4
            factor_rank: 1
            corpus:
              source: ./nemotron_sample.jsonl
              max_rows: 8
            iterations: 4
            batch_size: 2
            learning_rate: 0.003
            max_seq_len: 256
            checkpoint_every: 0
        pipeline:
          answer:
            model: qwen
            adapter: code_oft
            routes:
              - when:
                  decide:
                    domain: math
                model: qwen
                adapter: math_phm
              - model: qwen
                adapter: code_oft
        """
        try Data(yaml.utf8).write(to: yamlURL)
        return yamlURL
    }

    @Test("Compose v2 parses adapters and route.adapter, emits JSON schema definitions, and catches invalid version or missing adapters")
    func composeAdapterParsingAndSchemaValidation() throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "basalt-compose-parse-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let yamlURL = try Self.writeComposeFixture(to: scratch)

        let spec = try ComposeFile.load(yamlURL.path, context: ComposeContext())
        #expect(spec.adapters.count == 2)
        #expect(spec.answerAdapter == "code_oft")
        #expect(spec.routes.count == 2)
        #expect(spec.routes[0].adapter == "math_phm")
        #expect(spec.issues.isEmpty)

        let schemaJSON = try #require(try JSONSerialization.jsonObject(with: Data(ComposeSchema.jsonSchema.utf8)) as? [String: Any])
        let defs = try #require(schemaJSON["$defs"] as? [String: Any])
        #expect(defs["compose_adapter"] != nil && defs["adapter_corpus"] != nil)

        var badVersion = spec
        badVersion.version = 1
        #expect(badVersion.issues.contains { $0.path == "adapters" })

        var badRoute = spec
        badRoute.routes[0].adapter = "missing_adapter"
        #expect(badRoute.issues.contains { $0.path.contains("routes[0].adapter") })
    }

    @Test("ComposeTrainRunner trains OFT and PHM domain specialists once, skips cached runs, and AnswerRouter switches per-request adapters")
    func composeMultiAdapterTrainAndRoute() async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "basalt-compose-adapters-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let yamlURL = try Self.writeComposeFixture(to: scratch)
        let spec = try ComposeFile.load(yamlURL.path, context: ComposeContext())

        let selected = try ComposeTrainRunner.selectAdapters(in: spec, filter: [])
        let env = ComposeTrainEnvironment(
            force: false,
            resume: false,
            home: scratch,
            makeEngine: { _ in try await TinyModel.engine(.qwen3) }
        )
        let firstOutcomes = try await ComposeTrainRunner.trainSelected(selected, in: spec, env: env)
        #expect(firstOutcomes.count == 2 && firstOutcomes.allSatisfy { !$0.skipped })
        let secondOutcomes = try await ComposeTrainRunner.trainSelected(selected, in: spec, env: env)
        #expect(secondOutcomes.allSatisfy { $0.skipped })

        let liveDirs = spec.adapters(forBase: "qwen", home: scratch)
        #expect(Set(liveDirs.keys) == ["code_oft", "math_phm"])

        let servingEngine = try await TinyModel.engine(.qwen3)
        let orchestrator = ModelOrchestratorActor(engine: servingEngine)
        try await orchestrator.loadAdapters(liveDirs)
        #expect(await orchestrator.adapterNames == ["code_oft", "math_phm"])

        let probeTarget = AdapterProbeTarget(label: "qwen")
        let router = AnswerRouter(routes: spec.routes, targets: ["qwen": probeTarget], fallback: probeTarget, defaultAdapter: spec.answerAdapter)

        let mathSit = AnswerSituation(modalities: [.text], labels: ["domain": "math"], promptTokens: 12, escalated: false, costLeft: ["qwen"])
        let codeSit = AnswerSituation(modalities: [.text], labels: ["domain": "code"], promptTokens: 12, escalated: false, costLeft: ["qwen"])
        let mathReply = try await router.answer(GenerationRequest(prompt: "2+2", maxTokens: 4), situation: mathSit)
        let codeReply = try await router.answer(GenerationRequest(prompt: "fn main()", maxTokens: 4), situation: codeSit)
        #expect(mathReply.reply.text == "qwen:math_phm")
        #expect(codeReply.reply.text == "qwen:code_oft")
        #expect(probeTarget.seenAdapters.withLock { $0 } == ["math_phm", "code_oft"])
    }
}
