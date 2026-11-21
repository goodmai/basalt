import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("gpt-oss reasoning effort: the template has no off switch")
struct GPTOSSProfileTests {
    @Test("gpt-oss has no way to switch reasoning off, so off and none ask for the lowest level instead of the template's medium")
    func gptOSSEffortMapping() {
        let request = { (effort: String?) in GenerationRequest(prompt: "p").with { $0.reasoningEffort = effort } }
        let profile = ModelProfileRegistry.profile(forModelType: "gpt_oss")
        #expect(profile is GPTOSSProfile)
        for off in ["none", "off", "false", "minimal", "low"] {
            #expect(MLXInferenceEngine.buildTemplateContext(request: request(off), profile: profile)?["reasoning_effort"] as? String == "low")
        }
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("xhigh"), profile: profile)?["reasoning_effort"] as? String == "high")
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("medium"), profile: profile)?["reasoning_effort"] as? String == "medium")
        #expect(MLXInferenceEngine.buildTemplateContext(request: request(nil), profile: profile) == nil)
        #expect(MLXInferenceEngine.buildTemplateContext(request: request("bogus"), profile: profile) == nil)
    }
}
