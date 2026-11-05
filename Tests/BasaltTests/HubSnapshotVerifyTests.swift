import CryptoKit
import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltServer

@Suite("Cached snapshots are verified on demand, read-only, and --require-verified refuses what cannot be verified (#367)")
struct HubSnapshotVerifyTests {
    static let revision = "0e415031434cdf5f1b89d584e11be33b82abfc8d"
    static let configSHA1 = "9e26dfeeb6e641a33dae4961196235bdb965b21b"
    static let helloSHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"

    @Test("A flipped byte in a blob is verified false and makes the report not intact; no blob, link or sidecar is written")
    func flippedByte() async throws {
        let cache = try Cache(files: ["config.json": Data("{}".utf8), "model.safetensors": Self.shard()])
        defer { cache.remove() }
        try cache.writeSidecars()
        let (hub, host) = Self.hub { _ in (500, Data()) }

        let clean = await ModelInspector().inspect(modelIdOrPath: cache.snapshot.path, verifyingWith: hub)
        #expect(clean.intact)
        #expect(clean.files == [.init(name: "config.json", verified: .verified), .init(name: "model.safetensors", verified: .verified)])

        let before = try cache.state()
        var blob = try Data(contentsOf: cache.blob("model.safetensors"))
        blob[blob.count - 1] ^= 0xFF
        try blob.write(to: cache.blob("model.safetensors"))
        let flipped = try cache.state()

        let report = await ModelInspector().inspect(modelIdOrPath: cache.snapshot.path, verifyingWith: hub)
        #expect(!report.intact)
        #expect(report.files?.last == .init(name: "model.safetensors", verified: .mismatch))
        #expect(report.errors == ["Digest mismatch: model.safetensors"])
        #expect(try cache.state() == flipped)
        #expect(flipped.links == before.links && flipped.entries == before.entries)
        #expect(StubURLProtocol.requests(for: host).isEmpty, "every file has a sidecar, so the tree API is not asked")
    }

    @Test("A file without a sidecar is checked against the tree API: lfs.oid for an LFS file, the git blob id otherwise")
    func treeAPI() async throws {
        let cache = try Cache(files: ["config.json": Data("{}".utf8), "model.safetensors": Data("hello\n".utf8), "4bit/tokenizer.json": Data("{}".utf8)])
        defer { cache.remove() }
        let tree = """
        [{"type": "file", "path": "config.json", "oid": "\(Self.configSHA1)"},
         {"type": "file", "path": "model.safetensors", "oid": "ce013625030ba8dba906f756967f9e9ca394464a", "lfs": {"oid": "\(Self.helloSHA256)", "size": 6}},
         {"type": "directory", "path": "4bit", "oid": "\(Self.configSHA1)"},
         {"type": "file", "path": "4bit/tokenizer.json", "oid": "ce013625030ba8dba906f756967f9e9ca394464a"}]
        """
        let (hub, host) = Self.hub { _ in (200, Data(tree.utf8)) }

        let verdicts = await HubSnapshot(directory: cache.snapshot).verify(hub: hub)
        #expect(verdicts == [
            .init(name: "4bit/tokenizer.json", verified: .mismatch),
            .init(name: "config.json", verified: .verified),
            .init(name: "model.safetensors", verified: .verified),
        ])
        let request = try #require(StubURLProtocol.requests(for: host).first?.0.url)
        #expect(request.path == "/api/models/org/tiny/tree/\(Self.revision)")
        #expect(request.query == "recursive=true")

        let quant = HubSnapshot(directory: cache.snapshot.appendingPathComponent("4bit"))
        #expect(quant.repoId == "org/tiny")
        #expect(quant.revision == Self.revision)
        #expect(quant.prefix == "4bit/")
        #expect(await quant.verify(hub: hub) == [.init(name: "tokenizer.json", verified: .mismatch)])
    }

    @Test("A tree API failure, or a directory outside the Hugging Face layout, gives unknown and never verified")
    func unknown() async throws {
        let cache = try Cache(files: ["config.json": Data("{}".utf8)])
        defer { cache.remove() }
        let (failing, _) = Self.hub { _ in (503, Data()) }
        #expect(await HubSnapshot(directory: cache.snapshot).verify(hub: failing) == [.init(name: "config.json", verified: .unknown)])

        let plain = try HubDigestTests.directory()
        defer { try? FileManager.default.removeItem(at: plain) }
        try Data("{}".utf8).write(to: plain.appendingPathComponent("config.json"))
        let (hub, host) = Self.hub { _ in (200, Data("[]".utf8)) }
        #expect(await HubSnapshot(directory: plain).verify(hub: hub) == [.init(name: "config.json", verified: .unknown)])
        #expect(StubURLProtocol.requests(for: host).isEmpty)
    }

    @Test("download --require-verified refuses a file the server sends no digest for, after the HEAD, with no GET and no retry")
    func downloadRequiresDigest() async throws {
        let body = Data("hello\n".utf8)
        let host = StubURLProtocol.host { request, _ in
            (200, ["Content-Length": "\(body.count)"], request.httpMethod == "HEAD" ? Data() : body)
        }
        let directory = try HubDigestTests.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = HubFile(
            name: "config.json",
            url: try #require(URL(string: "https://\(host)/org/tiny/resolve/main/config.json")),
            destination: directory.appendingPathComponent("config.json"))

        let strict = HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)", requireVerified: true)
        let error = await #expect(throws: HFError.self) { try await strict.transferTask(file, token: nil) { _, _, _ in }.value }
        #expect(error?.errorDescription == HFError.digestMissing("config.json").errorDescription)
        #expect(StubURLProtocol.requests(for: host).map { $0.0.httpMethod } == ["HEAD"])
        #expect(!FileManager.default.fileExists(atPath: file.destination.path))

        let lenient = HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)")
        try await lenient.downloadFileWithResume(file, token: nil) { _, _, _ in }
        #expect(try Data(contentsOf: file.destination) == body)
    }

    @Test("serve --require-verified refuses a snapshot with a mismatched or unverifiable file, names them and says how to fix it")
    func serveRefuses() async throws {
        let cache = try Cache(files: ["config.json": Data("{}".utf8), "model.safetensors": Data("hello\n".utf8), "tokenizer.json": Data("{}".utf8)])
        defer { cache.remove() }
        try HubSidecar.write(Self.configSHA1, for: cache.snapshot.appendingPathComponent("config.json"))
        try HubSidecar.write(Self.helloSHA256, for: cache.snapshot.appendingPathComponent("tokenizer.json"))
        let (hub, _) = Self.hub { _ in (200, Data("[]".utf8)) }

        let error = await #expect(throws: HFError.self) { try await HubSnapshot(directory: cache.snapshot).requireVerified(hub: hub) }
        guard case .unverifiedSnapshot(let path, let files)? = error else { Issue.record("got \(String(describing: error))"); return }
        #expect(path == cache.snapshot.path)
        #expect(files == ["model.safetensors", "tokenizer.json"])
        #expect(error?.errorDescription?.contains("basalt models inspect \(path) --verify") == true)

        try HubSidecar.write(Self.helloSHA256, for: cache.snapshot.appendingPathComponent("model.safetensors"))
        try HubSidecar.write(Self.configSHA1, for: cache.snapshot.appendingPathComponent("tokenizer.json"))
        try await HubSnapshot(directory: cache.snapshot).requireVerified(hub: hub)
    }

    @Test("The MCP inspect_model tool reports the per-file verdict with verify and leaves it out without")
    func mcpInspect() async throws {
        let directory = try HubDigestTests.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = directory.appendingPathComponent("config.json")
        try Data("{}".utf8).write(to: config)
        try HubSidecar.write(Self.configSHA1, for: config)
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: MockInferenceEngine())) { capturer.capture($0) }

        for verify in [true, false] {
            await server.dispatch(#"{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "inspect_model", "arguments": {"modelId": "\#(directory.path)", "verify": \#(verify)}}}"#)
        }
        let reports = try capturer.lines.map { line in
            let response = try #require(try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
            let content = try #require(((response["result"] as? [String: Any])?["content"] as? [[String: Any]])?.first?["text"] as? String)
            return try #require(try JSONSerialization.jsonObject(with: Data(content.utf8)) as? [String: Any])
        }
        #expect(reports.count == 2)
        #expect((reports.first?["files"] as? [[String: String]]) == [["name": "config.json", "verified": "true"]])
        #expect(reports.last?["files"] == nil)
    }

    static func hub(_ respond: @escaping @Sendable (URLRequest) -> (Int, Data)) -> (HuggingFaceHub, String) {
        let host = StubURLProtocol.host { request, _ in
            let (status, body) = respond(request)
            return (status, [:], body)
        }
        return (HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)"), host)
    }

    static func shard() -> Data {
        let header = Data(#"{"w": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}"#.utf8)
        let length = UInt64(header.count)
        return Data((0 ..< 8).map { UInt8((length >> ($0 * 8)) & 0xFF) }) + header + Data(repeating: 0, count: 8)
    }

    struct State: Equatable {
        let entries: [String]
        let links: [String: String]
        let blobs: [String: Data]
    }

    struct Cache {
        let root: URL
        let snapshot: URL
        let files: [String: Data]

        init(files: [String: Data]) throws {
            root = try HubDigestTests.directory()
            snapshot = root.appendingPathComponent("models--org--tiny/snapshots/\(HubSnapshotVerifyTests.revision)")
            self.files = files
            let blobs = root.appendingPathComponent("models--org--tiny/blobs")
            try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
            for (name, data) in files {
                let digest = HubDigest.hexString(SHA256.hash(data: data))
                try data.write(to: blobs.appendingPathComponent(digest))
                let link = snapshot.appendingPathComponent(name)
                try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
                let depth = name.split(separator: "/").count + 1
                let target = String(repeating: "../", count: depth) + "blobs/\(digest)"
                if !FileManager.default.fileExists(atPath: link.path) {
                    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target)
                }
            }
        }

        func blob(_ name: String) throws -> URL {
            snapshot.appendingPathComponent(name).resolvingSymlinksInPath()
        }

        func writeSidecars() throws {
            for (name, data) in files {
                try HubSidecar.write(HubDigest.hexString(SHA256.hash(data: data)), for: snapshot.appendingPathComponent(name))
            }
        }

        func state() throws -> State {
            let manager = FileManager.default
            let entries = (manager.enumerator(atPath: root.path)?.allObjects as? [String] ?? []).sorted()
            let links = try Dictionary(uniqueKeysWithValues: files.keys.map { ($0, try manager.destinationOfSymbolicLink(atPath: snapshot.appendingPathComponent($0).path)) })
            let blobs = root.appendingPathComponent("models--org--tiny/blobs")
            let contents = try Dictionary(uniqueKeysWithValues: manager.contentsOfDirectory(atPath: blobs.path).map { ($0, try Data(contentsOf: blobs.appendingPathComponent($0))) })
            return State(entries: entries, links: links, blobs: contents)
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }
}

@Suite("models inspect --verify on a real symlinked Hugging Face cache, and on a copy with one flipped byte (#367)", .serialized)
struct HubSnapshotVerifyRealCacheTests {
    static let repo = "mlx-community/bge-small-en-v1.5-bf16"
    static let enabled = ProcessInfo.processInfo.environment["BASALT_E2E"] == "1"

    struct Tree: Equatable {
        let entries: [String]
        let links: [String: String]
        let blobs: [String: [FileAttributeKey: String]]

        init(_ root: URL) throws {
            let manager = FileManager.default
            entries = (manager.enumerator(atPath: root.path)?.allObjects as? [String] ?? []).sorted()
            let paths = entries.map { root.appendingPathComponent($0).path }
            links = Dictionary(uniqueKeysWithValues: paths.compactMap { path in (try? manager.destinationOfSymbolicLink(atPath: path)).map { (path, $0) } })
            blobs = try Dictionary(uniqueKeysWithValues: paths.filter { $0.contains("/blobs/") }.map { path in
                let attributes = try manager.attributesOfItem(atPath: path)
                let kept: [FileAttributeKey: String] = [FileAttributeKey.size, .modificationDate, .systemFileNumber].reduce(into: [:]) { $0[$1] = "\(attributes[$1] ?? "")" }
                return (path, kept)
            })
        }
    }

    @Test("Every cached file verifies; with the sidecars removed from a copy the tree API verifies them; a flipped byte in a copied blob is reported; the real blobs and links are untouched",
          .enabled(if: enabled, "set BASALT_E2E=1 with mlx-community/bge-small-en-v1.5-bf16 in the Hugging Face cache"))
    func realCache() async throws {
        let log = try RealRunLog("hub-verify")
        let snapshot = ModelCache.cacheDir(for: Self.repo)
        let modelRoot = snapshot.deletingLastPathComponent().deletingLastPathComponent()
        log("start", "commit \(ModelRunLog.output("/usr/bin/git", ["-C", log.root.path, "rev-parse", "HEAD"])) snapshot \(snapshot.path)")
        let before = try Tree(modelRoot)

        let real = await ModelInspector().inspect(modelIdOrPath: Self.repo, verifyingWith: HuggingFaceHub())
        log.verdicts("real cache", real)
        #expect(real.intact)
        #expect(real.files?.isEmpty == false && real.files?.allSatisfy { $0.verified == .verified } == true)

        let temp = try HubDigestTests.directory().resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: temp) }
        let copyRoot = temp.appendingPathComponent(modelRoot.lastPathComponent)
        try FileManager.default.copyItem(at: modelRoot, to: copyRoot)
        let copy = copyRoot.appendingPathComponent("snapshots/\(snapshot.lastPathComponent)")
        let blob = copy.appendingPathComponent("model.safetensors").resolvingSymlinksInPath()
        try #require(blob.path.hasPrefix(copyRoot.path + "/blobs/"), "the copied link must point into the copied blobs, not the real cache")
        for name in try FileManager.default.contentsOfDirectory(atPath: copy.path) where name.hasPrefix(".") && name.hasSuffix(".sha256") {
            try FileManager.default.removeItem(at: copy.appendingPathComponent(name))
        }
        log("copy", "cp -R \(modelRoot.path) \(copyRoot.path), sidecars removed from the copy")

        let tree = await ModelInspector().inspect(modelIdOrPath: copy.path, verifyingWith: HuggingFaceHub())
        log.verdicts("copy via tree API", tree)
        #expect(tree.files?.map(\.name) == real.files?.map(\.name))
        #expect(tree.files?.allSatisfy { $0.verified == .verified } == true)

        let handle = try FileHandle(forUpdating: blob)
        let offset = try handle.seekToEnd() / 2
        try handle.seek(toOffset: offset)
        let byte = try #require(try handle.read(upToCount: 1)?.first)
        try handle.seek(toOffset: offset)
        try handle.write(contentsOf: [byte ^ 0xFF])
        try handle.close()
        log("flip", "byte \(offset) of \(blob.path)")

        let flipped = await ModelInspector().inspect(modelIdOrPath: copy.path, verifyingWith: HuggingFaceHub())
        log.verdicts("copy with a flipped byte", flipped)
        #expect(!flipped.intact)
        #expect(flipped.files?.filter { $0.verified != .verified } == [.init(name: "model.safetensors", verified: .mismatch)])

        let after = try Tree(modelRoot)
        log(after == before ? "PASS" : "FAIL", after == before ? "real blobs, links and entries unchanged" : "the real cache changed")
        #expect(after == before)
    }
}

struct RealRunLog {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let url: URL

    init(_ runner: String) throws {
        let directory = root.appendingPathComponent("logs")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("\(runner)-\(Date().ISO8601Format(.iso8601.year().month().day())).log")
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
    }

    func callAsFunction(_ step: String, _ detail: String) {
        let line = "\(Date().ISO8601Format()) \(HubSnapshotVerifyRealCacheTests.repo) \(step) \(detail)\n"
        print(line, terminator: "")
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data(line.utf8))
        try? handle.close()
    }

    func verdicts(_ step: String, _ report: ModelInspector.Report) {
        self(step, "intact \(report.intact), errors \(report.errors)")
        for file in report.files ?? [] { self(step, "\(file.name) verified \(file.verified.rawValue)") }
    }
}
