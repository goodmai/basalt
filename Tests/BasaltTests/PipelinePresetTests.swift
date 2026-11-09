import Testing
import Foundation
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer
@testable import BasaltCLI

@Suite("Pipeline presets: document-qa, voice-agent and support-triage expand to the serve flags for the working set, explicit flags win, and the plan admits or refuses each set (#402 phase 3)")
struct PipelinePresetTests {
    static let gb: Int64 = 1_000_000_000
    static let budgets: [Int64] = [18 * gb, 24 * gb]

    @Test("Each preset expands to its models: Gemma 4 12B on an 18 GB working set, Qwen3.8-27B 3-bit when it fits with 8 GB to spare")
    func expansion() {
        let small = PipelinePreset.allCases.map { $0.choices(budgetBytes: 18 * Self.gb) }
        #expect(small.map(\.chat) == Array(repeating: PipelinePreset.gemma.repo, count: 3))
        #expect(small.map(\.ocr) == [PipelinePreset.paddleOCR.repo, nil, nil])
        #expect(small.map(\.audio) == [nil, PipelinePreset.nemotronTranscribe.repo, nil])
        #expect(small.map(\.decide) == ["self", nil, "self"])
        #expect(small.map(\.abstainBelow) == [0.75, nil, 0.75] && small.map(\.minMargin) == [0.1, nil, 0.1])
        let large = PipelinePreset.allCases.map { $0.choices(budgetBytes: 24 * Self.gb).chat }
        #expect(large == [PipelinePreset.qwen.repo, PipelinePreset.gemma.repo, PipelinePreset.qwen.repo])
        #expect(PipelinePreset.documentQA.choices(budgetBytes: PipelinePreset.qwen.bytes + PipelinePreset.paddleOCR.bytes + PipelinePreset.kvHeadroomBytes - 1).chat == PipelinePreset.gemma.repo)
        #expect(PipelinePreset.supportTriage.questions == PipelinePreset.triageQuestions && PipelinePreset.documentQA.questions.isEmpty)
        #expect(PipelinePreset.triageQuestions.allSatisfy { (try? SystemOneCore.validate($0.value, name: $0.key)) != nil })
    }

    @Test("Override order: an explicit value replaces the preset's, a missing one keeps it")
    func overrideOrder() {
        let preset = PipelinePreset.documentQA.choices(budgetBytes: 18 * Self.gb)
        let merged = preset.overridden(by: PipelineChoices(ocr: "mlx-community/GLM-OCR-8bit", decide: "/bundles/laya", abstainBelow: 0.9))
        #expect(merged == PipelineChoices(chat: preset.chat, ocr: "mlx-community/GLM-OCR-8bit", decide: "/bundles/laya", abstainBelow: 0.9, minMargin: 0.1))
        #expect(preset.overridden(by: PipelineChoices()) == preset)
    }

    @Test("Plan per preset at 18 and 24 GB: document-qa and support-triage fit, voice-agent is refused because Nemotron Transcribe does not fit beside the chat model", arguments: budgets)
    func plans(budget: Int64) throws {
        let plans = PipelinePreset.allCases.map { $0.plan(budgetBytes: budget) }
        #expect(plans.map(\.fits) == [true, false, true])
        #expect(plans.map { $0.plan.placements.map(\.workload.role) } == [["chat", "perceive"], ["chat", "perceive"], ["chat"]])
        #expect(plans.allSatisfy { $0.plan.residentBytes == $0.plan.placements.reduce(0) { $0 + $1.workload.residentBytes } && $0.plan.budgetBytes == budget })
        let refusal = try #require(plans[1].plan.refusal)
        #expect(refusal.contains("Nemotron-3.5-Transcribe") && refusal.contains(ComputePlan.gigabytes(budget)))
        #expect(plans[1].lines.last == "refused: \(refusal)" && plans[0].lines.first?.hasPrefix("pipeline preset document-qa: chat ") == true)
        #expect(plans[0].plan.placements.last?.note.contains("gated between generations") == true)
    }

    @Test("On battery the plan is power-degraded: no sidecar runs beside decode")
    func degraded() {
        let plan = PipelinePreset.documentQA.plan(budgetBytes: 18 * Self.gb, power: PowerState(onAC: false, charging: false, batteryPercent: 40, adapterWatts: nil))
        #expect(plan.plan.degraded && plan.fits)
        #expect(plan.plan.placements.allSatisfy { !$0.concurrentWithDecode })
        #expect(plan.lines.contains { $0.contains("power-degraded") })
    }

    @Test("serve --pipeline-preset fills only the flags left unset: --model, --model-path, --pipeline-ocr-model and --decision-model given explicitly win")
    func serveExpansion() async throws {
        var plain = try ServeCommand.parse(["--pipeline-preset", "document-qa"])
        await plain.applyPipelinePreset(budgetBytes: 18 * Self.gb)
        #expect(plain.model == PipelinePreset.gemma.repo && plain.pipeline.ocrModel == PipelinePreset.paddleOCR.repo && plain.decisionModel == "self")
        #expect(plain.pipeline.abstainBelow == 0.75 && plain.pipeline.minMargin == 0.1 && plain.pipeline.audioModel == nil)
        var explicit = try ServeCommand.parse([
            "--pipeline-preset", "document-qa", "--model", "org/chat", "--pipeline-ocr-model", "mlx-community/GLM-OCR-8bit", "--decision-model", "/bundles/laya", "--pipeline-min-margin", "0.2",
        ])
        await explicit.applyPipelinePreset(budgetBytes: 18 * Self.gb)
        #expect(explicit.model == "org/chat" && explicit.pipeline.ocrModel == "mlx-community/GLM-OCR-8bit" && explicit.decisionModel == "/bundles/laya")
        #expect(explicit.pipeline.minMargin == 0.2 && explicit.pipeline.abstainBelow == 0.75)
        var local = try ServeCommand.parse(["--pipeline-preset", "voice-agent", "--model-path", "/models/chat"])
        await local.applyPipelinePreset(budgetBytes: 18 * Self.gb)
        #expect(local.model == nil && local.modelPath == "/models/chat" && local.pipeline.audioModel == PipelinePreset.nemotronTranscribe.repo && local.decisionModel == nil)
        var none = try ServeCommand.parse([])
        await none.applyPipelinePreset(budgetBytes: 18 * Self.gb)
        #expect(none.model == nil && none.decisionModel == nil)
        #expect(throws: (any Error).self) { try ServeCommand.parse(["--pipeline-preset", "llama-desk"]) }
    }

    @Test("The config key serve.pipeline.preset selects a preset; config keys and flags beside it win over the preset, and a flag wins over the config's preset")
    func configKey() async throws {
        var configured = try ServeConfigFileTests.serve("defaults: { serve: { pipeline: { preset: document-qa, ocr_model: mlx-community/GLM-OCR-8bit } } }")
        await configured.applyPipelinePreset(budgetBytes: 18 * Self.gb)
        #expect(configured.pipeline.preset == .documentQA && configured.pipeline.ocrModel == "mlx-community/GLM-OCR-8bit" && configured.model == PipelinePreset.gemma.repo)
        let flagged = try ServeConfigFileTests.serve("defaults: { serve: { pipeline: { preset: document-qa } } }", ["--pipeline-preset", "support-triage"])
        #expect(flagged.pipeline.preset == .supportTriage)
    }

    @Test("basalt fit --pipeline-preset parses the name; the JSON plan carries snake_case placements, resident bytes, the degraded state and the refusal")
    func fitPlan() throws {
        #expect(try FitCommand.parse(["--pipeline-preset", "voice-agent"]).pipelinePreset == .voiceAgent)
        #expect(throws: (any Error).self) { try FitCommand.parse(["--pipeline-preset", "nope"]) }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let json = try #require(JSONSerialization.jsonObject(with: encoder.encode(PipelinePreset.voiceAgent.plan(budgetBytes: 24 * Self.gb))) as? [String: Any])
        let plan = try #require(json["plan"] as? [String: Any])
        #expect(json["preset"] as? String == "voice-agent" && json["fits"] as? Bool == false)
        #expect(plan["resident_bytes"] as? Int64 == PipelinePreset.gemma.bytes + PipelinePreset.nemotronTranscribe.bytes && plan["degraded"] as? Bool == false && plan["refusal"] is String)
        #expect((json["choices"] as? [String: Any])?["audio"] as? String == PipelinePreset.nemotronTranscribe.repo)
    }

    static func preview(_ query: String) async throws -> (status: HTTPResponse.Status, json: [String: Any]) {
        let engine = MockInferenceEngine()
        try await engine.load(modelPath: "org/mock-model")
        let server = RESTServer(orchestrator: ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker()), config: try ServerConfig(modelPath: "org/mock-model"))
        return try await Application(router: server.makeRouter()).test(.router) { client in
            try await client.execute(uri: "/v1/models/preview?\(query)", method: .get) {
                ($0.status, (try? JSONSerialization.jsonObject(with: Data(buffer: $0.body)) as? [String: Any]) ?? [:])
            }
        }
    }

    @Test("/v1/models/preview?pipeline_preset= returns the preset's plan on this Mac's working set; an unknown preset is 400 listing the names")
    func restPreview() async throws {
        for preset in PipelinePreset.allCases {
            let response = try await Self.preview("pipeline_preset=\(preset.rawValue)")
            #expect(response.status == .ok)
            let plan = try #require(response.json["plan"] as? [String: Any])
            let placements = try #require(plan["placements"] as? [[String: Any]])
            let resident = try #require(plan["resident_bytes"] as? Int64)
            let budget = try #require(plan["budget_bytes"] as? Int64)
            #expect(response.json["preset"] as? String == preset.rawValue)
            #expect(resident == placements.compactMap { ($0["workload"] as? [String: Any])?["resident_bytes"] as? Int64 }.reduce(0, +))
            #expect(response.json["fits"] as? Bool == (budget == 0 || resident <= budget))
            #expect((plan["refusal"] != nil) == (response.json["fits"] as? Bool == false))
            #expect(plan["degraded"] is Bool)
            #expect(placements.first.flatMap { ($0["workload"] as? [String: Any])?["role"] as? String } == "chat")
        }
        let unknown = try await Self.preview("pipeline_preset=llama-desk")
        #expect(unknown.status == .badRequest)
        #expect(((unknown.json["error"] as? [String: Any])?["message"] as? String)?.contains(PipelinePreset.names) == true)
        #expect(try await Self.preview("model_id=mlx-community/Qwen3.6-27B-4bit").json["weights_bytes"] is Int)
    }
}

@Suite("support-triage preset end to end on the tiny model: serve's preset expansion builds the stages, the preset's questions reach System One, the tiny chat engine answers the escalations (#402, #228)")
struct PipelinePresetTinyModelTests {
    @Test("A ticket without questions is triaged on department, lateness and severity; flat scores abstain under the preset's 0.75 and the real tiny engine answers each", .timeLimit(.minutes(1)))
    func supportTriage() async throws {
        var serve = try ServeCommand.parse(["--pipeline-preset", "support-triage"])
        await serve.applyPipelinePreset(budgetBytes: 18 * PipelinePresetTests.gb)
        #expect(serve.decisionModel == "self" && serve.pipeline.ocrModel == nil)
        let stages = try await serve.loadPipelineStages(ServeCommand.PerceivePaths())
        let ticket = "Ticket: I was charged twice for order #4521.\nIncident: Users affected: 1."
        let late = try #require(PipelinePreset.triageQuestions["late"])
        let call = PipelineCore.System2Call.question(id: "late", prompt: PipelineCore.prompt(late, state: ticket))
        let reference = try await TinyModel.engine().generate(request: PipelineCore.request(for: call, origin: .unspecified)).generatedText
        let chat = ModelOrchestratorActor(engine: try await TinyModel.engine(), telemetry: TelemetryTracker())
        let decisions = GatingDecisionEngine(flat: Set(PipelinePreset.triageQuestions.keys))
        let response = try await PipelineAPITests.post(chat, decisions: decisions, stages: stages, ["text": ticket])
        #expect(response.status == .ok)
        let request = try #require(decisions.calls.requests.withLock { $0.first })
        #expect(request.questions == PipelinePreset.triageQuestions && request.abstainBelow == 0.75 && request.minMargin == 0.1)
        let answers = PipelineAPITests.answers(response.json)
        #expect(Set(answers.keys) == ["department", "late", "severity"])
        #expect(answers.values.allSatisfy { $0["answered_by"] as? String == "answer" })
        #expect(answers["late"]?["text"] as? String == reference)
        #expect((response.json["receipt"] as? [String: Any])?["system2_calls"] as? Int == 3)
        #expect(PipelineAPITests.stages(response.json).map { $0["stage"] as? String } == ["decide", "answer", "answer", "answer"])
    }
}
