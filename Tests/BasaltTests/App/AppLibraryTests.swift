import BasaltCore
import Foundation
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app model library: fit verdicts gate loading, downloads resume after a relaunch, deletes only unloaded cached models (#434 phase 3)", .timeLimit(.minutes(1)))
struct AppLibraryTests {
    static let fits = ModelFit(fits: true, level: "comfortable", requiredBytes: 8 << 30, availableBytes: 20 << 30)
    static let tooBig = ModelFit(fits: false, level: "insufficient", requiredBytes: 40 << 30, availableBytes: 20 << 30, warning: "needs 40 GB")

    @Test("Only org/name repo ids with safe characters are accepted")
    func repoIDs() {
        #expect(LibraryReducer.isRepoID("mlx-community/gemma-4-12B-it-qat-4bit"))
        for bad in ["../etc", "a/b/c", "a/..", "a b/c", "/abs", "org/", "solo"] { #expect(!LibraryReducer.isRepoID(bad), "\(bad)") }
    }

    @Test("A model the fit check refuses is never loaded; one that fits loads once at a time")
    func loadGate() {
        var state = LibraryState()
        #expect(LibraryReducer.reduce(&state, .listed([LibraryModel(id: "org/big", sizeBytes: 1, isLoaded: false), LibraryModel(id: "org/small", sizeBytes: 1, isLoaded: false)])) == [.fit("org/big"), .fit("org/small")])
        #expect(LibraryReducer.reduce(&state, .load("org/big")).isEmpty && state.problem?.contains("does not fit") == true)
        _ = LibraryReducer.reduce(&state, .fitted("org/big", Self.tooBig))
        _ = LibraryReducer.reduce(&state, .fitted("org/small", Self.fits))
        #expect(LibraryReducer.reduce(&state, .load("org/big")).isEmpty && state.problem?.contains("needs 40 GB") == true && !state.canLoad("org/big"))
        #expect(LibraryReducer.reduce(&state, .load("org/small")) == [.load("org/small")] && state.loading == "org/small" && state.problem == nil)
        #expect(LibraryReducer.reduce(&state, .load("org/small")).isEmpty)
        #expect(LibraryReducer.reduce(&state, .loaded("org/small")) == [.list] && state.loading == nil)
        #expect(LibraryReducer.reduce(&state, .failed("boom")) == [.list] && state.problem == "boom")
    }

    @Test("Downloads persist, resume once on appearing, stop on failure until retried, and can be cancelled")
    func downloads() {
        var state = LibraryState(pending: ["org/a"])
        #expect(LibraryReducer.reduce(&state, .appeared) == [.list, .download("org/a")] && state.downloads["org/a"]?.active == true)
        #expect(LibraryReducer.reduce(&state, .appeared) == [.list])
        #expect(LibraryReducer.reduce(&state, .download("nope")).isEmpty && state.problem?.contains("not a Hugging Face repo id") == true)
        #expect(LibraryReducer.reduce(&state, .download("org/b")) == [.download("org/b"), .fit("org/b"), .savePending(["org/a", "org/b"])])
        #expect(LibraryReducer.reduce(&state, .download("org/b")).isEmpty)
        _ = LibraryReducer.reduce(&state, .progressed("org/b", done: 5, total: 10))
        #expect(state.downloads["org/b"]?.fraction == 0.5)
        #expect(LibraryReducer.reduce(&state, .downloaded("org/b")) == [.savePending(["org/a"]), .list] && state.downloads["org/b"] == nil)
        _ = LibraryReducer.reduce(&state, .downloadFailed("org/a", "offline"))
        #expect(state.downloads["org/a"]?.error == "offline" && LibraryReducer.reduce(&state, .appeared) == [.list])
        #expect(LibraryReducer.reduce(&state, .download("org/a")).first == .download("org/a") && state.downloads["org/a"]?.error == nil)
        #expect(LibraryReducer.reduce(&state, .cancelDownload("org/a")) == [.cancelDownload("org/a"), .savePending([])])
    }

    @Test("A loaded model or one outside the library is not deleted")
    func deleteGate() {
        var state = LibraryState()
        _ = LibraryReducer.reduce(&state, .listed([LibraryModel(id: "org/on", sizeBytes: 1, isLoaded: true), LibraryModel(id: "org/off", sizeBytes: 1, isLoaded: false)]))
        #expect(LibraryReducer.reduce(&state, .delete("org/on")).isEmpty && LibraryReducer.reduce(&state, .delete("org/other")).isEmpty)
        #expect(LibraryReducer.reduce(&state, .delete("org/off")) == [.delete("org/off")])
    }

    @Test("The server's model list and preview decode; delete removes the model's cache folder and refuses a path")
    func client() throws {
        let list = #"{"object":"list","data":[{"id":"claude-local/org--a","display_name":"org/a","size_bytes":123,"is_loaded":true},{"display_name":"/abs/path"}]}"#
        #expect(try LibraryClient.models(Data(list.utf8)) == [LibraryModel(id: "org/a", sizeBytes: 123, isLoaded: true)])
        let preview = #"{"fits":false,"fit_level":"insufficient","required_bytes":40,"available_bytes":20,"warning":"too big"}"#
        #expect(try LibraryClient.fit(Data(preview.utf8)) == ModelFit(fits: false, level: "insufficient", requiredBytes: 40, availableBytes: 20, warning: "too big"))
        #expect(throws: EngineError.self) { _ = try LibraryClient.fit(Data("{}".utf8)) }
        let cache = FileManager.default.temporaryDirectory.appending(path: "basalt-library-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let folder = cache.appending(path: "models--org--gone/snapshots/main")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let client = LibraryClient.live(.loopback(port: 1, token: nil), cache: cache)
        try client.delete("org/gone")
        #expect(!FileManager.default.fileExists(atPath: cache.appending(path: "models--org--gone").path))
        #expect(throws: EngineError.self) { try client.delete("../x") }
    }

    @MainActor
    @Test("The store downloads with ordered progress, reports a failure, and a load tells the app which model it chose")
    func effects() async throws {
        let chosen = Mutex<[String]>([])
        let saved = Mutex<[[String]]>([])
        let stub = LibraryClient(
            models: { [LibraryModel(id: "org/a", sizeBytes: 1, isLoaded: false)] }, fit: { _ in Self.fits },
            load: { _ in },
            download: { id, progress in
                guard id != "org/broken" else { throw EngineError.transport("offline") }
                for done in [Int64(1), 2, 3] { progress(done, 3) }
            },
            delete: { _ in })
        let environment = LibraryEnvironment(client: { stub }, savePending: { ids in saved.withLock { $0.append(ids) } }, chose: { id in chosen.withLock { $0.append(id) } })
        let store = Store(LibraryState(), reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(environment))
        store.send(.appeared)
        await store.settle()
        #expect(store.state.models.map(\.id) == ["org/a"] && store.state.fits["org/a"] == Self.fits)
        store.send(.download("org/new"))
        await store.settle()
        #expect(store.state.downloads.isEmpty && saved.withLock { $0.last } == [])
        store.send(.download("org/broken"))
        await store.settle()
        #expect(store.state.downloads["org/broken"]?.error?.contains("offline") == true)
        store.send(.load("org/a"))
        await store.settle()
        #expect(chosen.withLock { $0 } == ["org/a"] && store.state.loading == nil)
    }

    @Test("Against the live REST server the model list decodes and loading an uncached model is refused")
    func live() async throws {
        try await AppEngineClientTests.serve(MockInferenceEngine()) { endpoint in
            let client = LibraryClient.live(endpoint, session: .isolated)
            _ = try await client.models()
            await #expect(throws: EngineError.self) { try await client.load("org/not-cached-\(UUID().uuidString.prefix(6))") }
        }
    }
}
