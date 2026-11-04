import Foundation
import Testing
import Synchronization
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

    @Test("A launch without a prompt cache hands back its decode task, so cancelling the request stops decoding", .timeLimit(.minutes(10)))
    func plainLaunchStopsOnCancel() async throws {
        let launch = try MLXInferenceEngine.launch(
            input: LMInput(tokens: MLXArray([5, 9, 5, 9, 5].map(Int32.init))), prompt: CachedPrompt(boundary: 4),
            options: LaunchOptions(params: GenerateParameters(maxTokens: 100_000, temperature: 0)), context: try Self.context())
        let task = try #require(launch.task)
        #expect(launch.prefillSeconds >= 0)
        await MLXInferenceEngine.settle(launch.task, cancelled: true)
        #expect(task.isCancelled)
    }

    @Test("A launch that keeps its prompt cache reports the prefill it ran before decoding started")
    func cachedLaunchReportsPrefill() async throws {
        let launch = try MLXInferenceEngine.launch(
            input: LMInput(tokens: MLXArray(Array(repeating: [5, 9, 7], count: 64).flatMap { $0 }.map(Int32.init))),
            prompt: CachedPrompt(boundary: 190), options: LaunchOptions(params: GenerateParameters(maxTokens: 4, temperature: 0), keepsCache: true),
            context: try Self.context())
        #expect(launch.prefillSeconds > 0)
        await MLXInferenceEngine.settle(launch.task, cancelled: false)
    }

    @Test("A launch reports as prefill the whole time it took, including the prompt the iterator prefills when it is built (#277)",
          arguments: [false, true])
    func prefillCoversTheIterator(keepsCache: Bool) async throws {
        let context = try Self.context()
        let chunks = Mutex<[ContinuousClock.Instant]>([])
        let launch = try MLXInferenceEngine.$prefillProgress.withValue({ _, _ in chunks.withLock { $0.append(.now) } }) {
            try MLXInferenceEngine.launch(
                input: LMInput(tokens: MLXArray((0 ..< 3000).map { Int32(1 + $0 % 200) })), prompt: CachedPrompt(boundary: 2999),
                options: LaunchOptions(params: GenerateParameters(maxTokens: 1, temperature: 0), keepsCache: keepsCache), context: context)
        }
        let instants = chunks.withLock { $0 }
        let first = try #require(instants.first)
        let last = try #require(instants.last)
        #expect(instants.count >= 2)
        #expect(launch.prefillSeconds >= (last - first) / .seconds(1))
        await MLXInferenceEngine.settle(launch.task, cancelled: false)
    }
}
