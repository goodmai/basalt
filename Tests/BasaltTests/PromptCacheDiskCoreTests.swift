import Foundation
import Testing
@testable import BasaltInference

@Suite("Prompt-cache disk value core (#284)")
struct PromptCacheDiskCoreTests {
    typealias Selection = PromptCacheDiskSelection
    typealias Manifest = PromptCacheDiskManifest

    static func entry(_ tokens: [Int], bytes: Int64 = 100, anyPrefix: Bool = true, namespace: String = "ns") -> PromptCacheDiskEntry {
        PromptCacheDiskEntry(
            name: PromptCacheDiskStore.fileName(fingerprint: "fp", namespace: namespace, tokens: tokens), fingerprint: "fp",
            namespace: namespace, tokens: tokens, servesAnyPrefix: anyPrefix, bytes: bytes)
    }

    static func file(_ name: String, bytes: Int64, age: TimeInterval) -> Selection.StoredFile {
        Selection.StoredFile(name: name, bytes: bytes, modified: Date(timeIntervalSince1970: 1_000_000 - age))
    }

    @Test("File names are the SHA-256 of fingerprint, namespace and tokens, unchanged on disk")
    func fileNames() {
        let name = PromptCacheDiskStore.fileName(fingerprint: "fp", namespace: "ns", tokens: [1, 2, 3])
        #expect(name == "f50b842aca32ec0a21699fbfcd23743440ae8dfc3ec295637c3bf1ab809582de.safetensors")
        #expect(PromptCacheDiskStore.isStoreFileName(name))
        #expect(!PromptCacheDiskStore.isStoreFileName(".tmp-x.safetensors"))
        #expect(!PromptCacheDiskStore.isStoreFileName(".lock"))
        #expect(PromptCacheDiskStore.fileName(fingerprint: "fp", namespace: "other", tokens: [1, 2, 3]) != name)
    }

    @Test("The model fingerprint hashes weights by identity and description files by content, in the same lines as before")
    func fingerprint() {
        #expect(PromptCacheFingerprint.Input(name: "model.safetensors", pathExtension: "safetensors") == .weights)
        #expect(PromptCacheFingerprint.Input(name: "config.json", pathExtension: "json") == .description)
        #expect(PromptCacheFingerprint.Input(name: "README.md", pathExtension: "md") == .ignored)

        var fingerprint = PromptCacheFingerprint(version: "9.9")
        fingerprint.addWeights("model.safetensors", identity: "blob abc 10")
        fingerprint.addDescription("config.json", contents: Data("{}".utf8))
        #expect(fingerprint.finalize() == "1ae4fac742b3228a622f45e74a8b86cecb46c21cd53b5e9d66e2f6c4cf04a17c")
    }

    @Test("The manifest written into a file decodes back from its safetensors header")
    func manifestRoundTrip() throws {
        let written = Manifest.encode(fingerprint: "fp", model: "m", namespace: "ns", tokens: [1, 2, 3], servesAnyPrefix: true)
        #expect(written == [
            "basalt_format": "1", "basalt_fingerprint": "fp", "basalt_model": "m", "basalt_namespace": "ns",
            "basalt_tokens": "1,2,3", "basalt_serves_any_prefix": "true",
        ])
        let header = Dictionary(uniqueKeysWithValues: written.map { (Manifest.Key.inHeader($0.key), $0.value) })
        let expected = Self.entry([1, 2, 3], bytes: 42)
        let decoded = try Manifest.decode(header: header, name: expected.name, bytes: 42).get()
        #expect(decoded == expected)
        #expect(Manifest.matches(snapshot: written, expected))
        #expect(!Manifest.matches(snapshot: written, Self.entry([1, 2])))
    }

    @Test("A header without metadata, with missing fields or under another name is unusable; another format is skipped")
    func manifestValidation() throws {
        let name = Self.entry([1, 2, 3]).name
        let full = ["1.basalt_format": "1", "1.basalt_fingerprint": "fp", "1.basalt_namespace": "ns", "1.basalt_tokens": "1,2,3"]
        func problem(_ header: [String: String], _ name: String) -> String? {
            guard case .failure(let error) = Manifest.decode(header: header, name: name, bytes: 1) else { return nil }
            return error.description
        }
        #expect(problem([:], name) == "no basalt prompt-cache metadata")
        #expect(problem(full.filter { $0.key != "1.basalt_tokens" }, name) == "incomplete basalt prompt-cache metadata")
        #expect(problem(full.merging(["1.basalt_tokens": "1,x"]) { $1 }, name) == "incomplete basalt prompt-cache metadata")
        #expect(problem(full, Self.entry([9]).name) == "incomplete basalt prompt-cache metadata")
        #expect(try Manifest.decode(header: full.merging(["1.basalt_format": "0"]) { $1 }, name: name, bytes: 1).get() == nil)
        #expect(Manifest.tokens(from: "4,5") == [4, 5])
        #expect(Manifest.tokens(from: "4,,5") == [4, 5])
        #expect(Manifest.tokens(from: "4,-") == nil)
    }

    @Test("Making room evicts the oldest unprotected files until the new one fits")
    func room() {
        let files = [Self.file("new", bytes: 40, age: 1), Self.file("old", bytes: 40, age: 3), Self.file("mid", bytes: 40, age: 2)]
        let fits = Selection.room(for: 30, in: files, quota: 120, keeping: [])
        #expect(fits.fits && fits.evict.map(\.name) == ["old"])
        let protected = Selection.room(for: 30, in: files, quota: 120, keeping: ["old"])
        #expect(protected.fits && protected.evict.map(\.name) == ["mid"])
        let full = Selection.room(for: 90, in: files, quota: 100, keeping: ["new"])
        #expect(!full.fits && full.evict.map(\.name) == ["old", "mid"])
        #expect(Selection.room(for: 10, in: files, quota: 200, keeping: []).evict.isEmpty)
    }

    @Test("Pruning drops files past the retention window, then the oldest until under the quota")
    func expired() {
        let day: TimeInterval = 86_400
        let files = [Self.file("fresh", bytes: 10, age: 0), Self.file("stale", bytes: 10, age: 8 * day), Self.file("week", bytes: 10, age: 6 * day)]
        let now = Date(timeIntervalSince1970: 1_000_000)
        #expect(Selection.expired(files, now: now, retentionDays: 7, quota: 100).map(\.name) == ["stale"])
        #expect(Selection.expired(files, now: now, retentionDays: 7, quota: 15).map(\.name) == ["stale", "week"])
        #expect(Selection.expired(files, now: now, retentionDays: 30, quota: 100).isEmpty)
    }

    @Test("The longest usable prefix wins, the smaller file breaks a tie, and less than half a file is not worth loading")
    func longestPrefix() {
        let entries = [Self.entry([1, 2, 3, 4], bytes: 200), Self.entry([1, 2, 3, 9], bytes: 100), Self.entry([1, 2, 7, 8, 9, 10, 11], bytes: 50)]
        let best = Selection.longestPrefix(in: entries, fingerprint: "fp", namespace: "ns", of: [1, 2, 3, 5], longerThan: 0, maxBytes: 1_000)
        #expect(best?.entry.bytes == 100 && best?.usable == 3)
        #expect(Selection.longestPrefix(in: entries, fingerprint: "fp", namespace: "ns", of: [1, 2, 3, 5], longerThan: 3, maxBytes: 1_000) == nil)
        #expect(Selection.longestPrefix(in: entries, fingerprint: "fp", namespace: "other", of: [1, 2, 3], longerThan: 0, maxBytes: 1_000) == nil)
        let small = Selection.longestPrefix(in: entries, fingerprint: "fp", namespace: "ns", of: [1, 2, 3, 5], longerThan: 0, maxBytes: 60)
        #expect(small == nil)
    }

    @Test("A longer shareable path supersedes shorter shareable files of the same namespace that it mostly extends")
    func superseded() {
        let long = Array(0 ..< 200)
        let prefix = Self.entry(Array(0 ..< 150))
        let diverged = Self.entry(Array(0 ..< 50) + Array(1_000 ..< 1_100))
        let exact = Self.entry(Array(0 ..< 150), anyPrefix: false)
        let elsewhere = Self.entry(Array(0 ..< 150), namespace: "other")
        let kept = Self.entry(Array(0 ..< 120))
        let found = Selection.superseded(
            by: long, in: [prefix, diverged, exact, elsewhere, kept], fingerprint: "fp", namespace: "ns", keeping: [kept.name])
        #expect(found == [prefix])
        #expect(Selection.supersedes(long, Self.entry(Array(0 ..< 100) + Array(500 ..< 564))))
        #expect(!Selection.supersedes(long, Self.entry(Array(0 ..< 99) + Array(500 ..< 565))))
    }
}
