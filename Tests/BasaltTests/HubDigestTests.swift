import CryptoKit
import Foundation
import Testing
@testable import BasaltCore

@Suite("Hub downloads verify each file's digest (#353)")
struct HubDigestTests {
    static let sha256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    static let body = Data((0 ..< 200_000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
    static let bodySHA256 = HubDigest.hexString(SHA256.hash(data: body))

    @Test("SHA-256 and git blob sha1 of a file match shasum -a 256 and git hash-object", arguments: [
        ("hello\n", "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03", "ce013625030ba8dba906f756967f9e9ca394464a"),
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
        ("basalt", "8369762ae90ec0b2252931df46d51aa11a4c0727669c41c7157e6da8a88430f3", "97c25ab30ef1cb39fc8be32757c0c5aaa5c868e6"),
    ])
    func goldenDigests(text: String, sha256: String, sha1: String) throws {
        let directory = try Self.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("f")
        try Data(text.utf8).write(to: url)
        #expect(try HubDigest.sha256(sha256).computed(fileAt: url) == sha256)
        #expect(try HubDigest.gitBlobSHA1(sha1).computed(fileAt: url) == sha1)
    }

    @Test("Header digests are 64 hex for sha256 or 40 for sha1; anything malformed counts as missing", arguments: [
        ("\"\(sha256)\"", HubDigest?.some(.sha256(sha256))),
        (sha256.uppercased(), .sha256(sha256)),
        ("W/\"ce013625030ba8dba906f756967f9e9ca394464a\"", .gitBlobSHA1("ce013625030ba8dba906f756967f9e9ca394464a")),
        (String(sha256.dropLast()), nil),
        (String(sha256.dropLast()) + "g", nil),
        ("\"d41d8cd98f00b204e9800998ecf8427e\"", nil),
        ("\"\"", nil),
        ("\"", nil),
    ])
    func headerFormat(header: String, digest: HubDigest?) {
        #expect(HubDigest(header: header) == digest)
    }

    @Test("X-Linked-Etag wins over ETag and a malformed one is not replaced by ETag; X-Linked-Size wins over Content-Length")
    func responseHeaders() throws {
        let url = try #require(URL(string: "https://huggingface.co/o/r/resolve/main/m.safetensors"))
        let sha1 = "ce013625030ba8dba906f756967f9e9ca394464a"
        let linked = try #require(HTTPURLResponse(url: url, statusCode: 302, httpVersion: nil, headerFields: [
            "X-Linked-Etag": "\"\(Self.sha256)\"", "ETag": "\"\(sha1)\"", "X-Linked-Size": "2903945409", "Content-Length": "1110",
        ]))
        #expect(HubDigest(linked) == .sha256(Self.sha256))
        #expect(HubResponse.remoteSize(linked) == 2_903_945_409)

        let plain = try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: ["ETag": "\"\(sha1)\"", "Content-Length": "6"]))
        #expect(HubDigest(plain) == .gitBlobSHA1(sha1))
        #expect(HubResponse.remoteSize(plain) == 6)

        let malformed = try #require(HTTPURLResponse(url: url, statusCode: 302, httpVersion: nil, headerFields: ["X-Linked-Etag": "\"xyz\"", "ETag": "\"\(sha1)\""]))
        #expect(HubDigest(malformed) == nil)
    }

    @Test("The HEAD stops at the redirect that carries X-Linked-Size and follows any other")
    func headRedirects() async throws {
        let url = try #require(URL(string: "https://huggingface.co/o/r/resolve/main/m.safetensors"))
        let next = URLRequest(url: url)
        let session = URLSession(configuration: .ephemeral)
        let task = session.dataTask(with: next)
        let lfs = try #require(HTTPURLResponse(url: url, statusCode: 302, httpVersion: nil, headerFields: ["X-Linked-Size": "10", "X-Linked-Etag": "\"\(Self.sha256)\""]))
        let renamed = try #require(HTTPURLResponse(url: url, statusCode: 307, httpVersion: nil, headerFields: ["X-Linked-Etag": "\"ce013625030ba8dba906f756967f9e9ca394464a\""]))
        let delegate = HubHeadRedirects()
        #expect(await delegate.urlSession(session, task: task, willPerformHTTPRedirection: lfs, newRequest: next) == nil)
        #expect(await delegate.urlSession(session, task: task, willPerformHTTPRedirection: renamed, newRequest: next) == next)
    }

    @Test("Hashing the prefix on disk and then the rest equals a one-shot hash, at any split point")
    func resumedHashEqualsOneShot() throws {
        let directory = try Self.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("partial")
        let splits = [0, 1, Self.body.count - 1, Self.body.count] + (0 ..< 16).map { _ in Int.random(in: 0 ... Self.body.count) }
        for split in splits {
            try Self.body.prefix(split).write(to: url)
            var hasher = try #require(try HubDigest.sha256(Self.bodySHA256).streamingHasher(prefix: url))
            hasher.update(data: Self.body.suffix(from: split))
            #expect(HubDigest.hexString(hasher.finalize()) == Self.bodySHA256, "split at \(split)")
        }
        #expect(try HubDigest.gitBlobSHA1("ce013625030ba8dba906f756967f9e9ca394464a").streamingHasher(prefix: url) == nil)
    }

    @Test("A resume onto a corrupted prefix is caught, the file deleted, and the next attempt ends verified")
    func corruptedPrefixResume() async throws {
        let fixture = try Fixture(headers: ["X-Linked-Etag": "\"\(Self.bodySHA256)\"", "Content-Length": "\(Self.body.count)"])
        defer { fixture.remove() }
        var prefix = Self.body.prefix(100_000)
        prefix[10] ^= 0xFF
        try prefix.write(to: fixture.file.destination)

        await #expect(throws: HFError.self) { try await fixture.attempt() }
        #expect(!FileManager.default.fileExists(atPath: fixture.file.destination.path))
        #expect(fixture.ranges == ["bytes=100000-"])

        try await fixture.attempt()
        #expect(try Data(contentsOf: fixture.file.destination) == Self.body)
        #expect(HubSidecar.recorded(for: fixture.file.destination) == Self.bodySHA256)
    }

    @Test("The mismatch error names the file and both digests")
    func mismatchError() {
        let error = HFError.digestMismatch(file: "m.safetensors", expected: "aa", actual: "bb")
        #expect(error.errorDescription == "Download of m.safetensors does not match its digest: expected aa, got bb")
    }

    @Test("A stale file of the right size is replaced, not skipped")
    func staleSameSizeReplaced() async throws {
        let fixture = try Fixture(headers: ["X-Linked-Etag": "\"\(Self.bodySHA256)\"", "Content-Length": "\(Self.body.count)"])
        defer { fixture.remove() }
        try Data(repeating: 7, count: Self.body.count).write(to: fixture.file.destination)

        try await fixture.attempt()
        #expect(try Data(contentsOf: fixture.file.destination) == Self.body)
        #expect(fixture.methods == ["HEAD", "GET"])
    }

    @Test("Without Content-Length a matching digest keeps the file and no digest refuses it")
    func missingContentLength() async throws {
        let verified = try Fixture(headers: ["X-Linked-Etag": "\"\(Self.bodySHA256)\""])
        defer { verified.remove() }
        try await verified.attempt()
        #expect(try Data(contentsOf: verified.file.destination) == Self.body)

        let unverified = try Fixture(headers: [:])
        defer { unverified.remove() }
        await #expect(throws: HFError.self) { try await unverified.attempt() }
        #expect(!FileManager.default.fileExists(atPath: unverified.file.destination.path))
    }

    @Test("A sidecar matching size and mtime is trusted with no GET; one that is out of date is ignored and the file re-hashed")
    func sidecar() async throws {
        let fixture = try Fixture(headers: ["X-Linked-Etag": "\"\(Self.bodySHA256)\"", "Content-Length": "\(Self.body.count)"])
        defer { fixture.remove() }
        try await fixture.attempt()
        let path = fixture.file.destination.path
        let recorded = Date(timeIntervalSince1970: 1_700_000_000)
        try FileManager.default.setAttributes([.modificationDate: recorded], ofItemAtPath: path)
        try HubSidecar.write(Self.bodySHA256, for: fixture.file.destination)

        var corrupted = Self.body
        corrupted[50_000] ^= 0xFF
        try corrupted.write(to: fixture.file.destination)
        try FileManager.default.setAttributes([.modificationDate: recorded], ofItemAtPath: path)
        try await fixture.attempt()
        #expect(fixture.methods == ["HEAD", "GET", "HEAD"])

        try FileManager.default.setAttributes([.modificationDate: recorded.addingTimeInterval(-60)], ofItemAtPath: path)
        try await fixture.attempt()
        #expect(fixture.methods == ["HEAD", "GET", "HEAD", "HEAD", "GET"])
        #expect(try Data(contentsOf: fixture.file.destination) == Self.body)
    }

    static func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hub-digest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    struct Fixture {
        let host: String
        let hub: HuggingFaceHub
        let directory: URL
        let file: HubFile

        init(headers: [String: String]) throws {
            let body = HubDigestTests.body
            host = StubURLProtocol.host { request, _ in
                if request.httpMethod == "HEAD" { return (200, headers, Data()) }
                guard let range = request.value(forHTTPHeaderField: "Range"),
                      let offset = Int(range.dropFirst("bytes=".count).dropLast()) else { return (200, [:], body) }
                return (206, [:], body.suffix(from: offset))
            }
            hub = HuggingFaceHub(configuration: StubURLProtocol.configuration, endpoint: "https://\(host)")
            directory = try HubDigestTests.directory()
            file = HubFile(
                name: "model.safetensors",
                url: try #require(URL(string: "https://\(host)/org/tiny/resolve/main/model.safetensors")),
                destination: directory.appendingPathComponent("model.safetensors"))
        }

        func attempt() async throws {
            try await hub.downloadFileWithResume(file, token: nil) { _, _, _ in }
        }

        var methods: [String] { StubURLProtocol.requests(for: host).map { $0.0.httpMethod ?? "GET" } }
        var ranges: [String] { StubURLProtocol.requests(for: host).compactMap { $0.0.value(forHTTPHeaderField: "Range") } }

        func remove() { try? FileManager.default.removeItem(at: directory) }
    }
}
