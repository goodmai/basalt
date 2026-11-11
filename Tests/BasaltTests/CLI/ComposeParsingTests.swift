import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI

@Suite("compose v2 parsing: the schema as data, exact lines, duplicate keys, paths next to the file, variables and profiles (#423 phase 4a)")
struct ComposeParsingTests {
    static let base = """
        version: 2
        name: desk
        variables:
          chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
          floor: "0.6"
        profiles:
          small:
            when: { memory_gb: { under: 20 } }
            variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
          big:
            when: { memory_gb: { over: 48 } }
        models:
          chat:
            source: ${chat_repo}
          large:
            source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly
            keep: on-demand
            profiles: [big]
        pipeline:
          answer: chat
        pipelines:
          heavy:
            profiles: [big]
            answer: { routes: [{ when: { prompt_tokens: { over: 8000 } }, model: large }, { model: chat }] }
        """

    static func refusal(_ yaml: String, context: ComposeContext = ComposeContext()) -> String {
        do {
            _ = try ComposeFile.parse(yaml, file: "c.yaml", context: context)
            return ""
        } catch {
            return error.description
        }
    }

    @Test("Variables substitute; a profile chosen by the machine's memory overrides them; --set wins; models and pipelines of inactive profiles drop out")
    func variablesAndProfiles() throws {
        let plain = try ComposeFile.parse(Self.base, file: "c.yaml", context: ComposeContext(memoryGB: 24))
        #expect(plain.local("chat") == "mlx-community/gemma-4-12B-it-qat-4bit" && plain.models["large"] == nil && plain.pipelines.isEmpty)
        let small = try ComposeFile.parse(Self.base, file: "c.yaml", context: ComposeContext(memoryGB: 16))
        #expect(small.local("chat") == "mlx-community/gemma-4-E4B-it-4bit")
        let big = try ComposeFile.parse(Self.base, file: "c.yaml", context: ComposeContext(memoryGB: 64))
        #expect(big.models["large"]?.keep == .onDemand && big.pipelineNames == ["default", "heavy"])
        let forced = try ComposeFile.parse(Self.base, file: "c.yaml", context: ComposeContext(memoryGB: 16, profiles: ["big"], variables: ["chat_repo": "org/override"]))
        #expect(forced.local("chat") == "org/override" && forced.models["large"] != nil)
        #expect(Self.refusal(Self.base, context: ComposeContext(variables: ["typo": "x"])).contains("--set typo is not declared under variables"))
        #expect(Self.refusal(Self.base, context: ComposeContext(profiles: ["huge"])).contains("--profile huge is not declared"))
    }

    @Test("Each parsing fault of #412 has a refusal with the exact path and line", arguments: [
        ("models:\n  chat:\n    source: a\n  chat:\n    source: b\npipeline:\n  answer: chat\n", "duplicated key(s): 'chat'", nil),
        ("models:\n  chat:\n    source: a\n    source: b\npipeline:\n  answer: chat\n", "duplicated key(s): 'source'", nil),
        ("models:\n  chat:\n    source: a\n", "pipeline: a compose file needs a pipeline", nil),
        ("models:\n  chat:\n    source: a\npipeline:\n  answer: chat\npipelines:\n  side:\n    decide:\n      cascade: [chat]\n      escalate_below: high\n    answer: chat\n", "pipelines.side.decide.escalate_below: must be a number", 10),
        ("models:\n  chat:\n    source: a\npipeline:\n  answer: chat\npipelines:\n  side:\n    decide:\n      cascade: [ghost, chat]\n    answer: chat\n", "pipelines.side.decide.cascade", 9),
        ("models:\n  chat:\n    source: ${repo}\npipeline:\n  answer: chat\n", "${repo} is not a variable declared under variables", 3),
        ("models:\n  chat:\n    source: a\n    profiles: [ghost]\npipeline:\n  answer: chat\n", "ghost is not declared under profiles", 4),
        ("models:\n  ? [a, b]\n  : { source: a }\npipeline:\n  answer: chat\n", "keys are plain names", nil),
    ])
    func faults(_ body: String, _ message: String, _ line: Int?) {
        let text = Self.refusal("version: 2\nname: t\n" + body)
        #expect(text.contains(message), "\(text)")
        if let line { #expect(text.contains("c.yaml:\(line + 2):"), "\(text)") }
    }

    @Test("Version 2 keys in a version 1 file name the version they need")
    func versionGate() {
        let text = Self.refusal(Self.base.replacingOccurrences(of: "version: 2", with: "version: 1"))
        #expect(text.contains("variables: needs version: 2") && text.contains("profiles: needs version: 2"))
        let missing = Self.refusal("name: t\nmodels:\n  chat:\n    source: a\npipeline:\n  answer: chat\n")
        #expect(missing.contains("version 1 or 2") && !missing.contains("needs version"), "\(missing)")
    }

    @Test("A relative model path resolves against the compose file's folder, not the server's working directory")
    func relativePaths() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "compose-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appending(path: "desk.yaml")
        try Data("version: 1\nname: d\nmodels:\n  chat:\n    source: ./models/chat\npipeline:\n  answer: chat\n".utf8).write(to: file)
        let spec = try ComposeFile.load(file.path, context: ComposeContext())
        #expect(spec.local("chat") == folder.appending(path: "models/chat").standardizedFileURL.path)
        #expect(spec.serveArguments.prefix(2) == ["--model-path", folder.appending(path: "models/chat").standardizedFileURL.path])
    }

    @Test("The JSON Schema is valid JSON with every section, and docs/compose.md names every key the schema knows")
    func schema() throws {
        let json = try #require(try JSONSerialization.jsonObject(with: Data(ComposeSchema.jsonSchema.utf8)) as? [String: Any])
        let definitions = try #require(json["$defs"] as? [String: Any])
        #expect(Set(definitions.keys) == Set(ComposeSchema.sections.keys))
        let docs = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "../../../docs/compose.md").standardizedFileURL, encoding: .utf8)
        for key in ComposeSchema.allKeys.sorted() { #expect(docs.contains("`\(key)`") || docs.contains("\(key):"), "docs/compose.md does not name \(key)") }
        #expect(ComposeSchema.keys("root", version: 1).isDisjoint(with: ["variables", "profiles"]))
    }

    @Test("Every example file parses on a 16, 24 and 64 GB Mac, and any-mac picks its chat model and long pipeline by memory", arguments: [16.0, 24, 64])
    func examples(memory: Double) throws {
        let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "../../../docs/examples/compose").standardizedFileURL
        let files = try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasSuffix(".yaml") }.sorted()
        #expect(files.count >= 6)
        for file in files { _ = try ComposeFile.load(folder.appending(path: file).path, context: ComposeContext(memoryGB: memory)) }
        let any = try ComposeFile.load(folder.appending(path: "any-mac.yaml").path, context: ComposeContext(memoryGB: memory))
        #expect(any.local("chat") == (memory < 20 ? "ornith-ai/Ornith-1.5-9B-MLX-4bit" : "mlx-community/gemma-4-12B-it-qat-4bit"))
        #expect(any.pipelineNames == (memory > 48 ? ["default", "long"] : ["default"]))
    }

    @Test("--set and --profile reach the context, and compose up passes them on to serve")
    func cliOverrides() throws {
        let source = try ComposeCommand.Source.parse(["-f", "desk.yaml", "--set", "chat_repo=org/x", "--profile", "big"])
        #expect(source.context.variables == ["chat_repo": "org/x"] && source.context.profiles == ["big"])
        #expect(source.serveFlags == ["--compose", "desk.yaml", "--compose-set", "chat_repo=org/x", "--compose-profile", "big"])
        #expect(throws: (any Error).self) { try ComposeCommand.Source.parse(["-f", "desk.yaml", "--set", "novalue"]) }
        let serve = try ServeCommand.parse(["--model", "org/m", "--compose", "desk.yaml", "--compose-set", "a=b", "--compose-profile", "big"])
        #expect(serve.pipeline.composeSet == ["a=b"] && serve.pipeline.composeProfile == ["big"])
    }
}
