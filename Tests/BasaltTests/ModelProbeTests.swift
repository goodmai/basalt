import Foundation
import Testing
import MLXLMCommon
import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

struct ASCIITokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(clamping: $0) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil }
    func convertIdToToken(_ id: Int) -> String? { (0 ..< 128).contains(id) ? String(UnicodeScalar(UInt8(id))) : nil }
    var bosToken: String? { nil }
    var eosToken: String? { "\u{4}" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(text: messages.map { "\($0["role"] as? String ?? ""): \($0["content"] as? String ?? "")\n" }.joined(), addSpecialTokens: false)
    }
}

@Suite("The load-time model probe judges gathered facts: a broken template, tokenizer, stop token or NaN logits fail the load; odd entropy, perplexity or a missed known answer only warn")
struct ModelProbeTests {
    static let healthy = ModelProbe.Facts(
        template: .rendered(tokens: 24), echoed: " " + ModelProbe.roundTripText, stopTokens: [.init(id: 2, name: "</s>")],
        vocabularySize: 32_000, nonFiniteLogits: 0, meanEntropy: 1.2, perplexity: 3.5,
        answers: [.init(expected: "blue cat", got: " Blue cat\n"), .init(expected: "8", got: " 8\n")])

    static func facts(_ change: (inout ModelProbe.Facts) -> Void) -> ModelProbe.Facts {
        var changed = healthy
        change(&changed)
        return changed
    }

    static let cases: [(String, ModelProbe.Facts, [ModelProbe.Check], ModelProbe.Verdict)] = [
        ("healthy", healthy, [], .pass),
        ("no chat template", facts { $0.template = .missing }, [.chatTemplate], .warn),
        ("template throws", facts { $0.template = .failed("unknown filter") }, [.chatTemplate], .fail),
        ("template renders nothing", facts { $0.template = .rendered(tokens: 0) }, [.chatTemplate], .fail),
        ("round trip loses text", facts { $0.echoed = "The quick brown fox" }, [.roundTrip], .fail),
        ("no stop token", facts { $0.stopTokens = [] }, [.stopTokens], .fail),
        ("stop token past the vocabulary", facts { $0.stopTokens = [.init(id: 32_000, name: "<eos>")] }, [.stopTokens], .fail),
        ("stop token the tokenizer cannot name", facts { $0.stopTokens = [.init(id: 2, name: "</s>"), .init(id: 7, name: nil)] }, [.stopTokens], .fail),
        ("NaN logits skip the checks that read them", facts {
            $0.nonFiniteLogits = 3
            $0.meanEntropy = .nan
            $0.perplexity = .nan
            $0.answers = [.init(expected: "8", got: "")]
        }, [.finiteLogits], .fail),
        ("next to uniform entropy", facts { $0.meanEntropy = 10 }, [.entropy], .warn),
        ("collapsed entropy", facts { $0.meanEntropy = 0.001 }, [.entropy], .warn),
        ("NaN entropy", facts { $0.meanEntropy = .nan }, [.entropy], .warn),
        ("perplexity at the ceiling", facts { $0.perplexity = ModelProbe.perplexityCeiling }, [], .pass),
        ("perplexity above the ceiling", facts { $0.perplexity = 41 }, [.perplexity], .warn),
        ("infinite perplexity", facts { $0.perplexity = .infinity }, [.perplexity], .warn),
        ("one known answer missed", facts { $0.answers[1] = .init(expected: "8", got: " 9") }, [.knownAnswer], .warn),
        ("a fail outranks warns", facts {
            $0.perplexity = 500
            $0.echoed = ""
        }, [.roundTrip, .perplexity], .fail),
    ]

    @Test("Each set of facts gives its findings in check order and the worst one is the verdict", arguments: cases)
    func verdicts(name: String, facts: ModelProbe.Facts, checks: [ModelProbe.Check], verdict: ModelProbe.Verdict) {
        let probe = ModelProbe(facts)
        #expect(probe.findings.map(\.check) == checks, "\(name)")
        #expect(probe.verdict == verdict, "\(name)")
        #expect((probe.failure != nil) == (verdict == .fail), "\(name)")
    }

    @Test("A probe result survives the disk cache's JSON round trip")
    func codable() throws {
        let probe = ModelProbe(Self.facts { $0.template = .failed("x") })
        let decoded = try JSONDecoder().decode(ModelProbe.self, from: try JSONEncoder().encode(probe))
        #expect(decoded.findings == probe.findings)
    }

    @Test("A tiny random-weight llama passes the tokenizer and logit checks and only warns: it cannot repeat \"blue cat\" and its perplexity on the fixed text is far above the ceiling")
    func tinyFixtureWarns() async throws {
        let container = ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama"), model: try TinyModel.make(.llama),
            processor: ByteInputProcessor(), tokenizer: ASCIITokenizer()))
        let probe = try await ModelProbe.run(container)
        #expect(probe.failure == nil)
        #expect(probe.verdict == .warn)
        #expect(probe.findings.contains { $0.check == .knownAnswer && $0.detail.contains("blue cat") })
        #expect(probe.findings.contains { $0.check == .perplexity })
        #expect(!probe.findings.contains { [.chatTemplate, .roundTrip, .stopTokens, .finiteLogits].contains($0.check) })
    }

    @Test("A probe failure reaches the caller as itself, not as corrupted weights, and maps to 422")
    func failureSurvivesLoad() async {
        let failure = BasaltError.modelProbeFailed(path: "/m", check: ModelProbe.Check.finiteLogits.rawValue, reason: "NaN")
        guard case .modelProbeFailed(_, let check, _) = await MLXInferenceEngine.loadFailure(failure, modelPath: "/m") else {
            Issue.record("a probe failure was remapped")
            return
        }
        #expect(check == "finite logits")
        #expect(failure.httpStatus == 422)
    }

    @Test("serve probes by default and --no-model-probe skips it; an engine built without the call does not probe")
    func optOut() async throws {
        #expect(try ServeCommand.parse([]).modelProbe)
        #expect(try !ServeCommand.parse(["--no-model-probe"]).modelProbe)
        #expect(await !MLXInferenceEngine().probesModel)
    }
}
