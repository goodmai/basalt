import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("--rope-yarn: staged config, fingerprint split, refusals that keep stock RoPE (#729 P1)")
struct RopeYarnFlagTests {
    static var qwen: [String: Any] { [
        "model_type": "qwen3_5",
        "text_config": [
            "model_type": "qwen3_5_text", "max_position_embeddings": 262_144, "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25,
            "hidden_size": 64,
        ] as [String: Any],
    ] }

    static func directory(_ config: [String: Any]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "rope-yarn-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appending(component: "config.json"))
        try Data("weights".utf8).write(to: directory.appending(component: "model.safetensors"))
        try Data("{}".utf8).write(to: directory.appending(component: "tokenizer.json"))
        return directory
    }

    @Test("The rewrite sets yarn, the factor and the native context inside text_config, copies rope_theta and partial_rotary_factor into rope_parameters, and scales max_position_embeddings")
    func rewrite() throws {
        let (config, native, effective) = try RopeYarnStage.rewritten(Self.qwen, factor: 2)
        #expect(native == 262_144 && effective == 524_288)
        let text = try #require(config["text_config"] as? [String: Any])
        let rope = try #require(text["rope_parameters"] as? [String: Any])
        #expect(rope["rope_type"] as? String == "yarn")
        #expect(rope["factor"] as? Double == 2)
        #expect(rope["original_max_position_embeddings"] as? Int == 262_144)
        #expect(rope["rope_theta"] as? Double == 10_000_000)
        #expect(rope["partial_rotary_factor"] as? Double == 0.25)
        #expect(text["max_position_embeddings"] as? Int == 524_288)
        #expect(config["model_type"] as? String == "qwen3_5")
    }

    @Test("A config that already scales its RoPE, or names no native context, is refused")
    func rewriteRefusals() throws {
        var yarn = Self.qwen
        var text = try #require(yarn["text_config"] as? [String: Any])
        text["rope_parameters"] = ["rope_type": "yarn", "factor": 4.0]
        yarn["text_config"] = text
        #expect(throws: RopeYarnStage.Refusal.self) { try RopeYarnStage.rewritten(yarn, factor: 2) }
        #expect(throws: RopeYarnStage.Refusal.self) { try RopeYarnStage.rewritten(["model_type": "qwen3_5_text"], factor: 2) }
    }

    @Test("Staging writes the rewritten config beside symlinks to every other file, keyed by the source and factor; an identical staging is reused in place, a stale one is replaced, no temporary directory is left behind, and the fingerprint differs from the source's")
    func stagedDirectory() throws {
        let source = try Self.directory(Self.qwen)
        let root = FileManager.default.temporaryDirectory.appending(path: "rope-yarn-root-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: source); try? FileManager.default.removeItem(at: root) }
        let staged = try RopeYarnStage.stage(directory: source, modelType: "qwen3_5", factor: 2, root: root)
        #expect(staged.directory == RopeYarnStage.stagingDirectory(for: source, factor: 2, root: root))
        #expect(staged.effectiveContext == 524_288)
        let link = try FileManager.default.destinationOfSymbolicLink(atPath: staged.directory.appending(component: "model.safetensors").path)
        #expect(URL(fileURLWithPath: link).resolvingSymlinksInPath() == source.appending(component: "model.safetensors").resolvingSymlinksInPath())
        #expect(PromptCacheDiskStore.fingerprint(modelDirectory: staged.directory) != PromptCacheDiskStore.fingerprint(modelDirectory: source))
        let marker = try FileManager.default.attributesOfItem(atPath: staged.directory.path)[.systemFileNumber] as? Int
        let again = try RopeYarnStage.stage(directory: source, modelType: "qwen3_5", factor: 2, root: root)
        #expect(again.directory == staged.directory)
        #expect(try FileManager.default.attributesOfItem(atPath: again.directory.path)[.systemFileNumber] as? Int == marker)
        try Data("{}".utf8).write(to: staged.directory.appending(component: "config.json"))
        _ = try RopeYarnStage.stage(directory: source, modelType: "qwen3_5", factor: 2, root: root)
        #expect(try Data(contentsOf: staged.directory.appending(component: "config.json")) != Data("{}".utf8))
        #expect(try FileManager.default.contentsOfDirectory(atPath: root.path).count == 1)
        let text = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: staged.directory.appending(component: "config.json"))) as? [String: Any])["text_config"] as? [String: Any]
        #expect((text?["rope_parameters"] as? [String: Any])?["rope_theta"] as? Double == 10_000_000)
    }

    @Test("Gemma 4, an unlisted family and a factor outside (1, 64] are refused with the reason, so the engine loads stock RoPE")
    func stageRefusals() throws {
        let source = try Self.directory(Self.qwen)
        defer { try? FileManager.default.removeItem(at: source) }
        let root = FileManager.default.temporaryDirectory.appending(path: "rope-yarn-root-\(UUID().uuidString)")
        for (type, factor, reason) in [("gemma4", 2.0, "p-RoPE"), ("llama", 2.0, "qwen3_5"), ("qwen3_5", 1.0, "(1, 64]")] {
            #expect {
                _ = try RopeYarnStage.stage(directory: source, modelType: type, factor: factor, root: root)
            } throws: { error in
                (error as? RopeYarnStage.Refusal)?.reason.contains(reason) == true
            }
        }
        #expect(!FileManager.default.fileExists(atPath: root.path))
    }

    @Test("The engine without --rope-yarn loads the source directory; with it, an unlisted family keeps the source and warns")
    func engineSource() async throws {
        let source = try Self.directory(Self.qwen)
        defer { try? FileManager.default.removeItem(at: source) }
        let plain = MLXInferenceEngine(EngineConfiguration())
        #expect(await plain.ropeSource(source, modelType: "qwen3_5", modelPath: source.path) == source)
        var configuration = EngineConfiguration()
        configuration.ropeYarn = 2
        let yarn = MLXInferenceEngine(configuration)
        #expect(await yarn.ropeSource(source, modelType: "gemma4", modelPath: source.path) == source)
    }

    @Test("rope_yarn in a Compose file becomes --rope-yarn and a reload that changes it needs a restart")
    func compose() {
        var spec = ComposeSpec(name: "yarn", models: ["chat": ComposeSpec.Model(source: .local("org/m"))], answer: "chat")
        spec.experimental.ropeYarn = 2
        #expect(spec.optionArguments.contains("--rope-yarn"))
        var next = spec
        next.experimental.ropeYarn = 4
        #expect(spec.experimentalRestartReason(to: next)?.contains("rope_yarn") == true)
        var invalid = spec
        invalid.experimental.ropeYarn = 1
        #expect(invalid.experimentalIssues.contains { $0.path == "experimental.rope_yarn" })
        #expect(!spec.experimentalIssues.contains { $0.path == "experimental.rope_yarn" })
    }
}
