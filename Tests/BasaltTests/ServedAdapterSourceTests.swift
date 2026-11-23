import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("--adapter name=org/repo: a repo id is told apart from a directory, and a cached adapter is found by adapter_config.json (#675 P1)")
struct ServedAdapterSourceTests {
    @Test("org/repo is a repository unless such a path exists; absolute, relative and ~ paths stay directories")
    func parse() throws {
        let sources = try ServedAdapterSource.parse(
            ["pi=harness-hub/Qwen3.8-27B-pi-adapted", "local=/tmp/adapter", "rel=./adapters/a", "home=~/a/b", "shadow=adapters/oc"],
            fileExists: { $0 == "adapters/oc" })
        #expect(sources["pi"] == .repository("harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(sources["local"] == .directory(URL(fileURLWithPath: "/tmp/adapter")))
        #expect(sources["rel"] == .directory(URL(fileURLWithPath: "./adapters/a")))
        #expect(sources["home"] == .directory(URL(fileURLWithPath: ("~/a/b" as NSString).expandingTildeInPath)))
        #expect(sources["shadow"] == .directory(URL(fileURLWithPath: "adapters/oc")))
    }

    @Test("A missing name, a missing source or no = sign is refused", arguments: ["=x/y", "pi=", "pi"])
    func invalid(spec: String) {
        #expect(throws: ServedAdapterSource.Invalid.self) { try ServedAdapterSource.parse([spec], fileExists: { _ in false }) }
    }

    @Test("A repeated name is refused; a three-part path, . or .. segments, whitespace and control characters never make a repository id")
    func repeatsAndDepth() throws {
        #expect(throws: ServedAdapterSource.Invalid.self) { try ServedAdapterSource.parse(["a=x/y", "a=z/w"], fileExists: { _ in false }) }
        #expect(try ServedAdapterSource.parse(["a=x/y/z"], fileExists: { _ in false })["a"] == .directory(URL(fileURLWithPath: "x/y/z")))
        for value in ["foo/..", "../foo", "foo/.", "my org/repo", "org/re\tpo"] {
            #expect(try ServedAdapterSource.parse(["a=\(value)"], fileExists: { _ in false })["a"] != .repository(value), "\(value)")
        }
    }

    @Test("A cached adapter needs adapter_config.json and its weights: config.json alone or adapter_config.json alone, as an interrupted download leaves it, is not one")
    func cachedAdapter() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "adapter-source-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(to: directory.appending(component: "config.json"))
        #expect(!ServedAdapterSource.isAdapterDirectory(directory))
        try Data("{}".utf8).write(to: directory.appending(component: "adapter_config.json"))
        #expect(!ServedAdapterSource.isAdapterDirectory(directory))
        try Data("w".utf8).write(to: directory.appending(component: "adapter_model.safetensors"))
        #expect(ServedAdapterSource.isAdapterDirectory(directory))
    }

    @Test("serve resolves a directory as given and a cached org/repo to its snapshot; a malformed --adapter fails validation")
    func serveResolvesSources() async throws {
        let repo = "basalt-test/adapter-\(UUID().uuidString.prefix(8))"
        let snapshot = ModelCache.cacheDir(for: repo)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: snapshot.deletingLastPathComponent().deletingLastPathComponent()) }
        try Data("{}".utf8).write(to: snapshot.appending(component: "adapter_config.json"))
        try Data("w".utf8).write(to: snapshot.appending(component: "adapter_model.safetensors"))
        let local = FileManager.default.temporaryDirectory.appending(path: "served-adapter-\(UUID().uuidString)")
        let command = try ServeCommand.parse(["--adapter", "local=\(local.path)", "--adapter", "pi=\(repo)"])
        let directories = try await command.adapterDirectories()
        #expect(directories["local"] == local)
        #expect(directories["pi"] == snapshot)
        #expect(throws: (any Error).self) { _ = try ServeCommand.adapterSources(["broken"]) }
    }
}
