import Foundation
import Jinja
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 Mini P0: config, profile, KV geometry and chat template (#650)")
struct Inkling2P0Tests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Inkling2")

    static func config() throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("config.json"))) as? [String: Any])
    }

    @Test("The released config decodes and validates: 24 layers, globals 5/11/17/23, 48 visits with the core 6..<18 three times")
    func configuration() throws {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Inkling2Configuration.self, from: data).validated()
        #expect(config.modelType == "inkling2_loop_text")
        #expect(config.globalLayers == [5, 11, 17, 23])
        #expect(config.loop == Inkling2Configuration.Loop(coreStart: 6, coreEnd: 18, numPasses: 3, gateRank: 64))
        #expect(config.visitOrder.count == 48)
        #expect(Array(config.visitOrder[6 ..< 19]) == Array(6 ..< 18) + [6])
        #expect(config.visitOrder.suffix(6) == Array(18 ..< 24))
        #expect((config.experts, config.expertsPerToken, config.sharedExperts, config.routeScale) == (128, 8, 0, 8))
        #expect(config.unpaddedVocabSize == 200_058 && config.vocabSize == 201_024)
    }

    @Test("Validation refuses a loop core past the last layer and local layers outside the stack")
    func refusals() throws {
        var config = try JSONDecoder().decode(Inkling2Configuration.self, from: Data(contentsOf: Self.fixtures.appendingPathComponent("config.json")))
        config.loop?.coreEnd = 25
        #expect(throws: Inkling2Configuration.Invalid.self) { try config.validated() }
        config.loop?.coreEnd = 18
        config.localLayers.append(24)
        #expect(throws: Inkling2Configuration.Invalid.self) { try config.validated() }
    }

    @Test("KV geometry counts 8 full visits (32 KiB per token at fp16), 40 windowed visits of 1024 tokens (168 MB) and 1.9 MB of short-conv state")
    func geometry() throws {
        let geometry = try #require(KVCacheGeometry(config: try Self.config()))
        #expect(geometry.bytesPerToken(nil) == 32_768)
        #expect(geometry.window == 1024)
        #expect(Int64(geometry.windowedElementsPerToken * KVCacheGeometry.activationBytes) * Int64(geometry.window) == 167_772_160)
        #expect(geometry.fixedStateBytes == 1_916_928)
    }

    @Test("The profile maps the server's reasoning policy onto the template's reasoning_effort", arguments: [
        (ReasoningPolicy.modelDefault, nil as String?), (.disabled, "none"), (.effort("high"), "high"), (.effort("xhigh"), "xhigh"),
        (.effort("0.35"), "0.35"), (.effort("1.5"), nil), (.effort("turbo"), nil),
    ])
    func profile(policy: ReasoningPolicy, expected: String?) {
        let profile = ModelProfileRegistry.profile(forModelType: "inkling2_loop")
        #expect(profile.name == "inkling2")
        let value = profile.templateContext(for: policy)["reasoning_effort"]
        #expect(value.map { "\($0)" } == expected)
    }

    struct GoldenCase: Decodable {
        let expected: String
    }

    static let goldenNames = ["system_user", "tools", "reasoning", "tool_call", "tool_reply_by_name", "tool_reply_by_id", "effort_numeric", "effort_none"]

    @Test("swift-jinja renders the vendor chat template byte for byte like transformers", arguments: goldenNames)
    func template(name: String) throws {
        let source = try String(contentsOf: Self.fixtures.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let goldens = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtures.appendingPathComponent("template_goldens.json"))) as? [String: [String: Any]])
        var golden = try #require(goldens[name])
        let expected = try #require(golden.removeValue(forKey: "expected") as? String)
        var context = try golden.mapValues { try Value(any: $0) }
        context["add_generation_prompt"] = .boolean(true)
        let rendered = try Template(ChatTemplateCompatibility.compatible(source), with: .init(lstripBlocks: true, trimBlocks: true)).render(context)
        #expect(rendered == expected)
    }
}
