import Testing
@testable import BasaltInference

@Suite("basalt models types reports the config.json model_type values this build loads")
struct LoadableModelTypesTests {
    @Test("Types from the fork factories and from basalt's own registrations are loadable, in input order; unknown and audio-only types are not")
    func loadableTypes() async {
        let loadable = await MLXInferenceEngine.loadableModelTypes(among: ["whisper", "qwen3", "inkling2_loop", "gemma4", "not_a_model", "lfm3"])
        #expect(loadable == ["qwen3", "inkling2_loop", "gemma4", "lfm3"])
    }
}
