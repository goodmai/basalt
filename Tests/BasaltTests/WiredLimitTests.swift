import Foundation
import Testing
@testable import BasaltInference
@testable import BasaltCLI

@Suite("Unless --no-wired-limit, the engine wires Metal's recommended working set while a model is loaded, and gives it back on unload (#33)", .serialized)
struct WiredLimitTests {
    @Test("Wiring holds one ticket of the working set and releasing returns it; a second wire replaces the first")
    func wireAndRelease() async {
        let engine = MLXInferenceEngine(EngineConfiguration(wiresWorkingSet: true))
        await engine.wireWorkingSet()
        #expect(await engine.wiredLimitBytes == Int(PromptCacheBudget.gpuWorkingSet().rawValue))
        await engine.wireWorkingSet()
        #expect(await engine.wiredLimitBytes == Int(PromptCacheBudget.gpuWorkingSet().rawValue))
        await engine.releaseWiredMemory()
        #expect(await engine.wiredLimitBytes == nil)
    }

    @Test("serve wires by default and --no-wired-limit opts out; an engine built without the flag leaves the system limit alone")
    func optOut() async throws {
        let engine = MLXInferenceEngine()
        await engine.wireWorkingSet()
        #expect(await engine.wiredLimitBytes == nil)
        #expect(try ServeCommand.parse([]).wiredLimit)
        #expect(try !ServeCommand.parse(["--no-wired-limit"]).wiredLimit)
    }
}
