import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltCloud

@Suite("basalt compose: the YAML schema, refusals with line numbers, the serve flags it expands to, and the plan for each memory mode (#412)")
struct ComposeFileTests {
    static let cloud = """
        version: 1
        name: support-desk
        models:
          chat:
            source: mlx-community/gemma-4-12B-it-qat-4bit
          ocr:
            source: mlx-community/PaddleOCR-VL-1.5-8bit
            keep: on-demand
            idle_unload: 2m
          jev:
            source: cloud:openrouter/~typesafe/jev-latest
            api_key_env: OPENROUTER_API_KEY
            data: no-perceived
        pipeline:
          perceive:
            documents: [ocr, apple-vision]
          decide:
            cascade: [jev, chat]
            escalate_below: 0.6
          answer: chat
        """

    @Test("A cloud cascade with on-demand OCR expands to serve flags that ServeCommand parses back")
    func cloudFile() throws {
        let spec = try ComposeFile.parse(Self.cloud, file: "c.yaml")
        #expect(spec.serveArguments == [
            "--model", "mlx-community/gemma-4-12B-it-qat-4bit", "--pipeline-ocr-model", "mlx-community/PaddleOCR-VL-1.5-8bit",
            "--pipeline-ocr-idle-unload", "120", "--pipeline-memory", "resident", "--decision-model", "cloud:~typesafe/jev-latest",
            "--decision-escalate", "self", "--decision-api-key-env", "OPENROUTER_API_KEY", "--decision-escalate-below", "0.6",
        ])
        let serve = try ServeCommand.parse(spec.serveArguments)
        #expect(serve.pipeline.ocrIdleUnload == 120 && serve.decision.cloudModel == "~typesafe/jev-latest" && serve.decision.localModel == "self")
        #expect(spec.cloudLines.first?.contains("$OPENROUTER_API_KEY") == true)
    }

    @Test("The bundled files parse, and voice-agent swaps its speech model per request", arguments: ["document-qa", "voice-agent"])
    func bundled(_ name: String) throws {
        let spec = try ComposeFile.load(name)
        #expect(spec.issues.isEmpty)
        let serve = try ServeCommand.parse(spec.serveArguments)
        #expect(serve.model == "mlx-community/gemma-4-12B-it-qat-4bit")
        #expect(name != "voice-agent" || (serve.pipeline.memory == .sequential && serve.pipeline.audioModel != nil && serve.experimental.speechAnalyzer))
        #expect(name != "document-qa" || (serve.pipeline.ocrIdleUnload == 120 && serve.decision.model == "self"))
    }

    static func replacing(_ old: String, _ new: String) -> String {
        precondition(cloud.contains(old))
        return cloud.replacingOccurrences(of: old, with: new)
    }

    @Test("Refusals name the key, its line and the reason", arguments: [
        (replacing("    data: no-perceived", "    data: no-perceived\n    model: x"), "unknown key"),
        (replacing("api_key_env: OPENROUTER_API_KEY", "api_key: sk-or-v1-0123456789abcdef0123456789abcdef"), "secrets never go in a compose file"),
        (replacing("name: support-desk", "name: sk-or-v1-0123456789abcdef0123456789abcdef"), "looks like a secret"),
        (replacing("name: support-desk", "name: ${DESK}"), "environment interpolation"),
        (replacing("cascade: [jev, chat]", "cascade: [jev]"), "a cascade ends with a local model"),
        (replacing("cascade: [jev, chat]", "cascade: [chat, jev]"), "a cascade ends with a local model"),
        (replacing("answer: chat", "answer: { routes: [{ when: { modality: text }, model: chat }] }"), "the last route takes every request left"),
        (replacing("answer: chat", "answer: { routes: [{ model: gpt }] }"), "gpt is not a model declared under models"),
        (replacing("    data: no-perceived", "    data: no-perceived\n    budget: { usd_per_day: -1 }"), "usd_per_day must be positive"),
        (replacing("    keep: on-demand\n", "    keep: on-demand\n    budget: { usd_per_day: 1 }\n"), "applies only to cloud models"),
        (replacing("answer: chat", "answer: { routes: [{ when: { modality: fax }, model: chat }] }"), "must be documents, audio or text"),
        (replacing("cascade: [jev, chat]", "cascade: [jev, laya]"), "laya is not a declared model"),
        (replacing("documents: [ocr, apple-vision]", "documents: [jev]"), "never leave the machine"),
        (replacing("documents: [ocr, apple-vision]", "documents: [ocr, scanner]"), "scanner is neither a declared model nor apple-vision"),
        (replacing("  answer: chat", "  answer: ocr"), "one model serves one role"),
        (replacing("idle_unload: 2m", "idle_unload: soon"), "a duration such as"),
        (replacing("api_key_env: OPENROUTER_API_KEY", "api_key_env: openrouter-key"), "names the environment variable"),
        (replacing("version: 1", "version: 2"), "only version 1"),
    ])
    func refusals(_ yaml: String, _ reason: String) throws {
        do {
            _ = try ComposeFile.parse(yaml, file: "c.yaml")
            Issue.record("parsed a file that should be refused for \(reason)")
        } catch {
            #expect(error.description.contains(reason), "\(error.description)")
            #expect(error.description.contains("c.yaml:"), "no line number in \(error.description)")
        }
    }

    static func sizes(_ bytes: [String: Int64]) -> ComposeSpec.Sizes {
        ComposeSpec.Sizes(bytes: { bytes[$0] }, onNeuralEngine: { _ in false })
    }

    static let gb: Int64 = 1_000_000_000

    @Test("Plan math: on-demand counts as resident, sequential needs only the largest stage plus 2 GB, and one stage too big alone is refused with the numbers")
    func plans() throws {
        let document = try ComposeFile.load("document-qa")
        let fits = document.plan(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 11 * Self.gb, "mlx-community/PaddleOCR-VL-1.5-8bit": 1 * Self.gb]), budgetBytes: 19 * Self.gb)
        #expect(fits.verdict.refusal == nil && fits.verdict.residentBytes == 12 * Self.gb && fits.unknown.isEmpty)
        let tooBig = document.plan(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 11 * Self.gb, "mlx-community/PaddleOCR-VL-1.5-8bit": 9 * Self.gb]), budgetBytes: 19 * Self.gb)
        #expect(tooBig.verdict.refusal?.contains("20.0 GB") == true)
        let voice = try ComposeFile.load("voice-agent")
        let swapped = voice.plan(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 11 * Self.gb, "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX": 15 * Self.gb]), budgetBytes: 19 * Self.gb)
        #expect(swapped.verdict.refusal == nil && swapped.verdict.memory == .sequential && swapped.verdict.keep(ofKind: .decode) == .sequential)
        let alone = voice.plan(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 11 * Self.gb, "divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX": 18 * Self.gb]), budgetBytes: 19 * Self.gb)
        #expect(alone.verdict.refusal?.contains("one stage at a time") == true)
        let unknown = document.plan(Self.sizes([:]), budgetBytes: 19 * Self.gb)
        #expect(unknown.unknown == ["chat", "ocr"])
    }

    @Test("compose plan output names the serve flags, on-demand and cloud models, unknown sizes and the refusal; sizes come from the local snapshot; the source is a file or a bundled name")
    func planCommand() throws {
        let spec = try ComposeFile.parse(Self.cloud, file: "c.yaml")
        let (verdict, unknown) = spec.plan(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 25 * Self.gb]), budgetBytes: 19 * Self.gb)
        let lines = ComposeCommand.lines(spec, verdict: verdict, unknown: unknown)
        #expect(lines.first?.hasPrefix("compose support-desk: basalt serve --model") == true)
        #expect(lines.contains { $0.contains("on demand") } && lines.contains { $0.contains("cloud:openrouter/~typesafe/jev-latest") })
        #expect(lines.contains { $0.contains("ocr: not downloaded") } && lines.last?.hasPrefix("refused:") == true)
        let directory = try OCRAPITests.modelDirectory("compose-sized", modelType: "qwen3")
        try Data(count: 1000).write(to: directory.appendingPathComponent("model.safetensors"))
        #expect(ComposeCommand.sizes.bytes(directory.path) == 1000)
        #expect(ComposeCommand.sizes.bytes("/nonexistent/compose-model") == nil && !ComposeCommand.sizes.onNeuralEngine(directory.path))
        #expect(throws: (any Error).self) { try ComposeCommand.Source.parse([]) }
        #expect(throws: (any Error).self) { try ComposeCommand.Source.parse(["document-qa", "-f", "x.yaml"]) }
        #expect(try ComposeCommand.Source.parse(["document-qa"]).spec().name == "document-qa")
        #expect(throws: (any Error).self) { try ComposeCommand.Source.parse(["-f", "/nonexistent/compose.yaml"]).spec() }
    }

    static let routed = """
        version: 1
        name: mom
        budget: { usd_per_day: 3 }
        models:
          chat:
            source: mlx-community/gemma-4-12B-it-qat-4bit
          big:
            source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly
            keep: on-demand
            idle_unload: 2m
          jev:
            source: cloud:openrouter/~typesafe/jev-latest
            api_key_env: HOME
          jev2:
            source: cloud:openrouter/~typesafe/jev-backup
            api_key_env: HOME
          frontier:
            source: cloud:openrouter/anthropic/claude-sonnet-5.5
            api_key_env: HOME
            budget: { usd_per_day: 2 }
        pipeline:
          decide:
            cascade: [jev, jev2, chat]
            escalate_below: 0.6
          answer:
            routes:
              - when: { decide: { route: code }, cost_left: true }
                model: frontier
                fallback: big
              - when: { prompt_tokens: { over: 8000 } }
                model: big
              - model: chat
        """

    @Test("Routes, budgets and a two-cloud cascade parse; the plan counts the routed local model and prints the budgets")
    func routedFile() throws {
        let spec = try ComposeFile.parse(Self.routed, file: "mom.yaml")
        #expect(spec.answer == "chat" && spec.answerModels == ["chat", "frontier", "big"] && spec.budget == 3)
        #expect(spec.routes.first?.when == AnswerRoute.When(decide: ["route": "code"], costLeft: true) && spec.routes.first?.fallback == "big")
        #expect(spec.routes[1].when.promptTokensOver == 8000 && spec.models["frontier"]?.budget == 2)
        #expect(spec.serveArguments.prefix(2) == ["--model", "mlx-community/gemma-4-12B-it-qat-4bit"] && spec.serveArguments.contains("cloud:~typesafe/jev-latest"))
        let workloads = spec.workloads(Self.sizes(["mlx-community/gemma-4-12B-it-qat-4bit": 11 * Self.gb, "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly": 12 * Self.gb])).workloads
        #expect(workloads.map(\.role) == ["chat", "answer"])
        #expect(spec.cloudLines.first?.contains("$3.00 a day in total") == true && spec.cloudLines.contains { $0.contains("frontier") && $0.contains("$2.00 a day") })
        #expect(spec.onDemandLines.contains { $0.hasPrefix("  big on demand") })
    }

    @Test("serve --compose builds the cascade from every cloud stage before the local one, behind one budget gate")
    func composedRuntime() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "mom-\(UUID().uuidString).yaml")
        defer { try? FileManager.default.removeItem(at: directory) }
        let runtime = Self.routed.replacingOccurrences(of: "  big:\n    source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly\n    keep: on-demand\n    idle_unload: 2m\n", with: "")
            .replacingOccurrences(of: "fallback: big", with: "fallback: chat").replacingOccurrences(of: "    model: big\n", with: "    model: frontier\n")
        try Data(runtime.utf8).write(to: directory)
        let spec = try ComposeFile.load(directory.path)
        let serve = try ServeCommand.parse(spec.serveArguments + ["--compose", directory.path])
        let gate = serve.cloudGate(try serve.composeSpec())
        let cascade = try #require(try serve.composedDecisions(spec, local: StubDecisionEngine(), gate: GenerationGate(), besideDecode: true, cloudGate: gate) as? CascadeDecisionEngine)
        #expect(cascade.stages.map(\.modelID) == ["~typesafe/jev-latest", "~typesafe/jev-backup", "org/stub-jev"])
        #expect(cascade.stages.prefix(2).allSatisfy { $0 is GuardedDecisionEngine })
    }
}
