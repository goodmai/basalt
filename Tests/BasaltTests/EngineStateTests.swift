import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Engine state: model lifecycle transitions and stale load results (#300)")
struct EngineStateTests {
    let container: ModelContainer

    init() throws {
        container = try TinyModel.container(.gemma4)
    }

    func ready(_ epoch: ModelEpoch, _ path: String = "/m") -> EngineState {
        .ready(LoadedModel(container: container, epoch: epoch, facts: ModelFacts(description: path)))
    }

    func finished(_ epoch: ModelEpoch, _ path: String = "/m") -> EngineEvent {
        .loadFinished(epoch, container, ModelFacts(description: path))
    }

    func everyState() -> [EngineState] {
        [.unloaded(3), .loading(epoch: 3, path: "/a"), ready(3), .failed(epoch: 3, error: .modelNotFound(identifier: "/a"))]
    }

    @Test("A load starts at the next epoch from every state")
    func loadStartsAtNextEpoch() {
        for state in everyState() {
            #expect(EngineState.reduce(state, .loadStarted(path: "/b")).summary == "loading 4 /b")
        }
    }

    @Test("A load that finishes at its own epoch becomes ready with its container and facts")
    func loadFinishes() throws {
        let next = EngineState.reduce(.loading(epoch: 2, path: "/m"), finished(2))
        let model = try #require(next.loaded)
        #expect(model.container === container)
        #expect(model.epoch == 2)
        #expect(model.facts.description == "/m")
    }

    @Test("A load that fails at its own epoch records the error")
    func loadFails() {
        let next = EngineState.reduce(.loading(epoch: 2, path: "/m"), .loadFailed(2, .modelNotFound(identifier: "/m")))
        #expect(next.summary == "failed 2")
        #expect(next.loaded == nil)
    }

    @Test("A finish or failure outside its own load changes nothing")
    func strayResultsAreIgnored() {
        for state in everyState() + [.loading(epoch: 4, path: "/b")] {
            let before = state.summary
            #expect(EngineState.reduce(state, finished(2)).summary == before)
            #expect(EngineState.reduce(state, .loadFailed(2, .modelNotFound(identifier: "/m"))).summary == before)
        }
        #expect(EngineState.reduce(ready(3), finished(3)).summary == "ready 3 /m")
    }

    @Test("Unloading bumps the epoch from every state")
    func unloadBumpsEpoch() {
        for state in everyState() {
            #expect(EngineState.reduce(state, .unloaded).summary == "unloaded 4")
        }
    }

    func hybrid(_ epoch: ModelEpoch) -> EngineEvent {
        .learned(epoch) { $0.isHybrid = true }
    }

    @Test("A fact learned for the loaded model is kept; one learned for an older model or while loading is dropped")
    func learnedFacts() {
        #expect(EngineState.reduce(ready(3), hybrid(3)).loaded?.facts.isHybrid == true)
        #expect(EngineState.reduce(ready(3), hybrid(2)).loaded?.facts.isHybrid == nil)
        #expect(EngineState.reduce(.loading(epoch: 3, path: "/a"), hybrid(3)).summary == "loading 3 /a")
        #expect(EngineState.reduce(.unloaded(3), hybrid(3)).summary == "unloaded 3")
    }

    @Test("A load started during another load wins; the first one's late result is dropped")
    func loadDuringLoad() {
        let first = EngineState.reduce(.unloaded(0), .loadStarted(path: "/a"))
        let second = EngineState.reduce(first, .loadStarted(path: "/b"))
        #expect(second.summary == "loading 2 /b")
        let late = EngineState.reduce(second, finished(1, "/a"))
        #expect(late.summary == "loading 2 /b")
        #expect(EngineState.reduce(late, .loadFailed(1, .modelNotFound(identifier: "/a"))).summary == "loading 2 /b")
        #expect(EngineState.reduce(late, finished(2, "/b")).summary == "ready 2 /b")
    }

    @Test("An unload during a load drops the load's result, success or failure")
    func unloadDuringLoad() {
        let unloaded = EngineState.reduce(.loading(epoch: 1, path: "/a"), .unloaded)
        #expect(unloaded.summary == "unloaded 2")
        #expect(EngineState.reduce(unloaded, finished(1, "/a")).summary == "unloaded 2")
        #expect(EngineState.reduce(unloaded, .loadFailed(1, .modelNotFound(identifier: "/a"))).summary == "unloaded 2")
    }
}

private extension EngineState {
    var summary: String {
        switch self {
        case .unloaded(let epoch): "unloaded \(epoch)"
        case .loading(let epoch, let path): "loading \(epoch) \(path)"
        case .ready(let model): "ready \(model.epoch) \(model.facts.description)"
        case .failed(let epoch, _): "failed \(epoch)"
        }
    }
}
