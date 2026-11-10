import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI

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
        (replacing("cascade: [jev, chat]", "cascade: [jev]"), "needs a local one after it"),
        (replacing("cascade: [jev, chat]", "cascade: [jev, laya]"), "laya is not a declared model"),
        (replacing("documents: [ocr, apple-vision]", "documents: [jev]"), "never leave the machine"),
        (replacing("answer: chat", "answer: { routes: [] }"), "routes come in phase 2"),
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
}
