import Testing
@testable import BasaltInference

@Suite("Prompt lookup draft limits follow the thermal state the iterator is given", .serialized)
struct LookupDraftConditionsTests {
    @Test("A hot Mac shortens the draft without changing the tokens, and the test fixture pins nominal conditions")
    func thermalStateShortensDraftsOnly() throws {
        let continuation = (0 ..< 12).map { LookupFixture.unit[(4 + $0) % LookupFixture.unit.count] }
        let hot = DraftConditions(thermalLevel: .critical, isLowPowerModeEnabled: false)
        let cool = try LookupFixture.lookup(
            try TinyModel.make(.qwen3), maxTokens: continuation.count, sampler: ScriptedSampler(continuation))
        let throttled = try LookupFixture.lookup(
            try TinyModel.make(.qwen3), maxTokens: continuation.count, sampler: ScriptedSampler(continuation),
            conditions: hot)
        expectTokenSequence(throttled.tokens, continuation)
        let coolRounds = try #require(cool.iterator.speculativeDecodingTelemetry).roundCount
        let hotRounds = try #require(throttled.iterator.speculativeDecodingTelemetry).roundCount
        #expect(coolRounds == 1)
        #expect(hotRounds > coolRounds)
    }
}
