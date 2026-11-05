import Foundation
import Testing
@testable import BasaltCore

@Suite("Hub transfer value core (#284)")
struct HubTransferTests {
    @Test("A partial file resumes, a complete one is skipped, anything else starts over", arguments: [
        (Int64(0), Int64(100), HubTransferPlan.startNew),
        (40, 100, .resume(from: 40)),
        (100, 100, .skip),
        (120, 100, .startNew),
        (40, -1, .startNew),
        (0, -1, .startNew),
        (0, 0, .startNew),
    ])
    func plan(existing: Int64, remote: Int64, expected: HubTransferPlan) {
        #expect(HubTransferPlan(existingSize: existing, remoteSize: remote) == expected)
    }

    @Test("Only a resume asks for a byte range, expects 206 and counts the bytes already on disk")
    func resumeRequest() {
        let resume = HubTransferPlan.resume(from: 40)
        #expect(resume.rangeHeader == "bytes=40-")
        #expect(resume.expectedStatus == 206)
        #expect(resume.totalSize(contentLength: 60) == 100)

        let fresh = HubTransferPlan.startNew
        #expect(fresh.rangeHeader == nil)
        #expect(fresh.expectedStatus == 200)
        #expect(fresh.totalSize(contentLength: 60) == 60)
    }

    @Test("A finished file is complete when its digest matched, else only when it matches a positive advertised size", arguments: [
        (Int64(100), Int64(100), false, true), (60, 100, false, false), (200, 100, false, false),
        (60, -1, false, false), (60, 0, false, false), (60, -1, true, true), (100, 100, true, true),
    ])
    func completeness(fileSize: Int64, remote: Int64, verified: Bool, complete: Bool) {
        #expect(HubTransferPlan.isComplete(fileSize: fileSize, remoteSize: remote, verified: verified) == complete)
    }

    @Test("Five attempts with 2, 4, 8 and 16 seconds between them")
    func retryBackoff() {
        let delays = (1 ... HubRetry.maxAttempts).map(HubRetry.delay(afterFailedAttempt:))
        #expect(delays == [.seconds(2), .seconds(4), .seconds(8), .seconds(16), nil])
    }

    @Test("File requests carry the bearer token and the range; HEAD keeps both")
    func requestHeaders() throws {
        let url = try #require(HubRequest.resolveURL(endpoint: "https://huggingface.co", repoId: "org/repo", path: "4bit/model 1.safetensors"))
        #expect(url.absoluteString == "https://huggingface.co/org/repo/resolve/main/4bit/model%201.safetensors")

        let head = HubRequest.file(url, token: "t", method: "HEAD")
        #expect(head.httpMethod == "HEAD")
        #expect(head.value(forHTTPHeaderField: "Authorization") == "Bearer t")
        #expect(head.value(forHTTPHeaderField: "Range") == nil)

        let resume = HubRequest.file(url, token: nil, range: HubTransferPlan.resume(from: 7).rangeHeader)
        #expect(resume.httpMethod == "GET")
        #expect(resume.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(resume.value(forHTTPHeaderField: "Range") == "bytes=7-")
        #expect(HubRequest.rangeHeader(8 ... 15) == "bytes=8-15")
    }

    @Test("Downloads skip GGUF and unsafe paths and stay inside the requested subfolder")
    func downloadable() {
        let names = ["config.json", "model.gguf", "../escape", "4bit/config.json", "4bit/model.safetensors", "8bit/config.json"]
        #expect(HubRequest.downloadable(names, subfolder: nil) == ["config.json", "4bit/config.json", "4bit/model.safetensors", "8bit/config.json"])
        #expect(HubRequest.downloadable(names, subfolder: "4bit") == ["4bit/config.json", "4bit/model.safetensors"])
    }

    @Test("A fetch accepts 200 for a whole file and 206 for a range; 404 names the file, anything else is an API error")
    func fetchStatus() {
        #expect(HubResponse.fetchFailure(status: 200, ranged: false, resource: "r/f") == nil)
        #expect(HubResponse.fetchFailure(status: 206, ranged: true, resource: "r/f") == nil)
        #expect(HubResponse.fetchFailure(status: 200, ranged: true, resource: "r/f")?.errorDescription == HFError.apiError(code: 200).errorDescription)
        #expect(HubResponse.fetchFailure(status: 404, ranged: true, resource: "r/f")?.errorDescription == HFError.modelNotFound("r/f").errorDescription)
        #expect(HubResponse.fetchFailure(status: nil, ranged: false, resource: "r/f")?.errorDescription == HFError.apiError(code: -1).errorDescription)
    }

    @Test("The total file size comes from the Content-Range suffix", arguments: [
        ("bytes 0-7/1234", Int64?(1234)), ("bytes 0-7/*", nil), (nil, nil),
    ])
    func contentRange(header: String?, total: Int64?) {
        #expect(HubResponse.totalSize(contentRange: header) == total)
    }
}
