import Foundation
import Testing
@testable import BasaltCore

@Suite("--adapter name=org/repo: a repo id is told apart from a directory, and a cached adapter is found by adapter_config.json (#675 P1)")
struct AdapterSourceTests {
    @Test("org/repo is a repository unless such a path exists; absolute, relative and ~ paths stay directories")
    func parse() throws {
        let sources = try AdapterSource.parse(
            ["pi=harness-hub/Qwen3.8-27B-pi-adapted", "local=/tmp/adapter", "rel=./adapters/a", "home=~/a/b", "shadow=adapters/oc"],
            fileExists: { $0 == "adapters/oc" })
        #expect(sources["pi"] == .repository("harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(sources["local"] == .directory(URL(fileURLWithPath: "/tmp/adapter")))
        #expect(sources["rel"] == .directory(URL(fileURLWithPath: "./adapters/a")))
        #expect(sources["home"] == .directory(URL(fileURLWithPath: ("~/a/b" as NSString).expandingTildeInPath)))
        #expect(sources["shadow"] == .directory(URL(fileURLWithPath: "adapters/oc")))
    }

    @Test("A missing name, a missing source, a repeated name and a deeper path are refused or kept as directories", arguments: ["=x/y", "pi=", "pi"])
    func invalid(spec: String) {
        #expect(throws: AdapterSource.Invalid.self) { try AdapterSource.parse([spec], fileExists: { _ in false }) }
    }

    @Test("A repeated name is refused, and a three-part path is a directory, not a repository")
    func repeatsAndDepth() throws {
        #expect(throws: AdapterSource.Invalid.self) { try AdapterSource.parse(["a=x/y", "a=z/w"], fileExists: { _ in false }) }
        #expect(try AdapterSource.parse(["a=x/y/z"], fileExists: { _ in false })["a"] == .directory(URL(fileURLWithPath: "x/y/z")))
    }

    @Test("A directory with adapter_config.json is an adapter directory; one with only config.json is not")
    func cachedAdapter() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "adapter-source-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(to: directory.appending(component: "config.json"))
        #expect(!AdapterSource.isAdapterDirectory(directory))
        try Data("{}".utf8).write(to: directory.appending(component: "adapter_config.json"))
        #expect(AdapterSource.isAdapterDirectory(directory))
    }
}
