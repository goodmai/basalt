import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltCLI

@Suite("--adapter name=hf:org/repo: only the hf: prefix names a Hugging Face repo, a missing local path never turns into a download, and a cached adapter needs its weights (#675 P1)")
struct ServedAdapterSourceTests {
    @Test("hf:org/repo is a repository; any other value is a directory, including a two-part relative path that does not exist")
    func parse() throws {
        let sources = try ServedAdapterSource.parse(
            ["pi=hf:harness-hub/Qwen3.8-27B-pi-adapted", "local=/tmp/adapter", "rel=./adapters/a", "home=~/a/b", "missing=adapters/oc"])
        #expect(sources["pi"] == .repository("harness-hub/Qwen3.8-27B-pi-adapted"))
        #expect(sources["local"] == .directory(URL(fileURLWithPath: "/tmp/adapter")))
        #expect(sources["rel"] == .directory(URL(fileURLWithPath: "./adapters/a")))
        #expect(sources["home"] == .directory(URL(fileURLWithPath: ("~/a/b" as NSString).expandingTildeInPath)))
        #expect(sources["missing"] == .directory(URL(fileURLWithPath: "adapters/oc")))
    }

    @Test("A missing name, a missing source, no = sign, a repeated name and a malformed hf: id are refused",
          arguments: [["=x/y"], ["pi="], ["pi"], ["a=x", "a=y"], ["a=hf:org"], ["a=hf:org/repo/sub"], ["a=hf:foo/.."], ["a=hf:../foo"], ["a=hf:my org/repo"], ["a=hf:org/re\tpo"]])
    func invalid(specs: [String]) {
        #expect(throws: ServedAdapterSource.Invalid.self) { try ServedAdapterSource.parse(specs) }
    }

    @Test("A repo listing counts as an adapter only with adapter_config.json beside its weights at the top level")
    func repositoryListing() {
        #expect(ServedAdapterSource.hasAdapterFiles(["adapter_config.json", "adapter_model.safetensors", "README.md"]))
        #expect(ServedAdapterSource.hasAdapterFiles(["adapter_config.json", "adapters.safetensors"]))
        #expect(!ServedAdapterSource.hasAdapterFiles(["config.json", "model.safetensors"]))
        #expect(!ServedAdapterSource.hasAdapterFiles(["adapter_config.json", "checkpoint-500/adapter_model.safetensors"]))
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

    @Test("serve resolves a directory as given and a cached hf:org/repo to its snapshot; a malformed --adapter fails validation")
    func serveResolvesSources() async throws {
        let repo = "basalt-test/adapter-\(UUID().uuidString.prefix(8))"
        let snapshot = ModelCache.cacheDir(for: repo)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: snapshot.deletingLastPathComponent().deletingLastPathComponent()) }
        try Data("{}".utf8).write(to: snapshot.appending(component: "adapter_config.json"))
        try Data("w".utf8).write(to: snapshot.appending(component: "adapter_model.safetensors"))
        let local = FileManager.default.temporaryDirectory.appending(path: "served-adapter-\(UUID().uuidString)")
        let command = try ServeCommand.parse(["--adapter", "local=\(local.path)", "--adapter", "pi=hf:\(repo)"])
        let directories = try await command.adapterDirectories()
        #expect(directories["local"]?.path == local.path)
        #expect(directories["pi"]?.path == snapshot.path)
        #expect(throws: (any Error).self) { _ = try ServeCommand.adapterSources(["broken"]) }
    }
}
