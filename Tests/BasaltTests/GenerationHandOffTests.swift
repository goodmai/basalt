import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltInference

@Suite("Handing a prefilled iterator to the decode worker (#214)")
struct GenerationHandOffTests {
    static func context() throws -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen35"), model: try TinyModel.make(.qwen35),
            processor: ByteInputProcessor(), tokenizer: ByteTokenizer())
    }

    @Test("A launch without a prompt cache hands back its decode task, so cancelling the request stops decoding", .timeLimit(.minutes(1)))
    func plainLaunchStopsOnCancel() async throws {
        let launch = try MLXInferenceEngine.launch(
            input: LMInput(tokens: MLXArray([5, 9, 5, 9, 5].map(Int32.init))), reuse: nil, keepsCache: false, boundary: 4,
            params: GenerateParameters(maxTokens: 100_000, temperature: 0), components: .init(), tools: nil, context: try Self.context())
        let task = try #require(launch.task)
        #expect(launch.prefillSeconds >= 0)
        await MLXInferenceEngine.settle(launch, cancelled: true)
        #expect(task.isCancelled)
    }

    @Test("A launch that keeps its prompt cache reports the prefill it ran before decoding started")
    func cachedLaunchReportsPrefill() async throws {
        let launch = try MLXInferenceEngine.launch(
            input: LMInput(tokens: MLXArray(Array(repeating: [5, 9, 7], count: 64).flatMap { $0 }.map(Int32.init))), reuse: nil,
            keepsCache: true, boundary: 190, params: GenerateParameters(maxTokens: 4, temperature: 0), components: .init(), tools: nil,
            context: try Self.context())
        #expect(launch.prefillSeconds > 0)
        await MLXInferenceEngine.settle(launch, cancelled: false)
    }
}
