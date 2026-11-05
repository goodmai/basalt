import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MTP drafting deeper than one token on a real hybrid model keeps greedy output identical to plain decoding (#207)", .serialized)
struct MTPBlockParityTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_MTP_MODEL_PATH"]
    }

    static let prompt = """
        Rename the parameter `count` to `limit` everywhere in this function and return only the code:

        func firstWords(_ text: String, count: Int) -> [String] {
            let words = text.split(separator: " ").map(String.init)
            guard count > 0 else { return [] }
            return Array(words.prefix(count))
        }
        """

    static func generate(mtp: Bool, blockSize: Int = 2) async throws -> GenerationResponse {
        let engine = MLXInferenceEngine(mtp: mtp, mtpBlockSize: blockSize)
        try await engine.load(modelPath: try #require(modelPath))
        return try await engine.generate(request: GenerationRequest(
            prompt: prompt, maxTokens: 160, temperature: 0).with { $0.reasoningEffort = "none" })
    }

    @Test("Greedy text with MTP blocks 2, 3 and 4 equals the text without MTP, and each block drafts",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MTP_MODEL_PATH"))
    func deeperBlocksMatchPlainDecoding() async throws {
        let plain = try await Self.generate(mtp: false)
        for blockSize in [2, 3, 4] {
            let drafted = try await Self.generate(mtp: true, blockSize: blockSize)
            print("MTPBlockParity block=\(blockSize) drafted=\(drafted.draftTokens ?? 0) accepted=\(drafted.acceptedDraftTokens ?? 0) calls=\(drafted.modelCalls ?? 0)")
            #expect((drafted.draftTokens ?? 0) > 0)
            #expect(drafted.generatedText == plain.generatedText)
        }
    }

    static func sampled(_ engine: MLXInferenceEngine, workload: String, mtp: Bool) async throws -> [GenerationResponse] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let prompt = try String(contentsOf: root.appending(path: "benchmarks/speculative-lookup/\(workload).txt"), encoding: .utf8)
        var replies: [GenerationResponse] = []
        for run in 1 ... 3 {
            let reply = try await engine.generate(request: GenerationRequest(
                prompt: prompt, maxTokens: 300, temperature: 0.7).with { $0.reasoningEffort = "none" })
            let drafted = reply.draftTokens ?? 0, accepted = reply.acceptedDraftTokens ?? 0
            print("MTPSeeded \(mtp ? "mtp" : "plain") \(workload) run=\(run) tokens=\(reply.completionTokens) tps=\(reply.tokensPerSecond) drafts=\(accepted)/\(drafted) alpha=\(drafted == 0 ? 0 : Double(accepted) / Double(drafted))")
            replies.append(reply)
        }
        return replies
    }

    @Test("At T = 0.7 with seed 7, MTP prints the plain decoder's text on code, prose and reasoning; tok/s and drafts accepted are printed per run (#290)",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MTP_MODEL_PATH"))
    func seededSamplingMatchesPlainDecoding() async throws {
        let workloads = ["code", "prose", "reasoning"]
        var plainText: [String: String] = [:]
        for mtp in [false, true] {
            let engine = MLXInferenceEngine(seed: 7, mtp: mtp)
            try await engine.load(modelPath: try #require(Self.modelPath))
            for workload in workloads {
                let replies = try await Self.sampled(engine, workload: workload, mtp: mtp)
                let reference = plainText[workload] ?? replies[0].generatedText
                plainText[workload] = reference
                for reply in replies { #expect(reply.generatedText == reference, "\(workload) mtp=\(mtp)") }
                if mtp { #expect(replies.allSatisfy { ($0.draftTokens ?? 0) > 0 }, "\(workload)") }
            }
        }
    }
}
