import CryptoKit
import Foundation

package enum HubTransferPlan: Equatable, Sendable {
    case skip
    case resume(from: Int64)
    case startNew

    package init(existingSize: Int64, remoteSize: Int64) {
        if existingSize > 0, remoteSize > 0, existingSize == remoteSize {
            self = .skip
        } else if existingSize > 0, existingSize < remoteSize {
            self = .resume(from: existingSize)
        } else {
            self = .startNew
        }
    }

    package var offset: Int64 {
        switch self {
        case .resume(let offset): offset
        case .skip, .startNew: 0
        }
    }

    package var rangeHeader: String? {
        switch self {
        case .resume(let offset): "bytes=\(offset)-"
        case .skip, .startNew: nil
        }
    }

    package var expectedStatus: Int {
        switch self {
        case .resume: 206
        case .skip, .startNew: 200
        }
    }

    package func totalSize(contentLength: Int64) -> Int64 {
        offset + contentLength
    }

    package static func isComplete(fileSize: Int64, remoteSize: Int64, verified: Bool) -> Bool {
        verified || (remoteSize > 0 && fileSize == remoteSize)
    }
}

package enum HubRetry {
    package static let maxAttempts = 5

    package static func delay(afterFailedAttempt attempt: Int) -> Duration? {
        attempt < maxAttempts ? .seconds(1 << attempt) : nil
    }
}

package struct HubFile: Equatable, Sendable {
    package let name: String
    package let url: URL
    package let destination: URL
}

package enum HubRequest {
    package static func resolveURL(endpoint: String, repoId: String, path: String) -> URL? {
        guard let encoded = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else { return nil }
        return URL(string: "\(endpoint)/\(repoId)/resolve/main/\(encoded)")
    }

    package static func file(_ url: URL, token: String?, method: String = "GET", range: String? = nil) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let range { request.setValue(range, forHTTPHeaderField: "Range") }
        return request
    }

    package static func rangeHeader(_ range: ClosedRange<Int>) -> String {
        "bytes=\(range.lowerBound)-\(range.upperBound)"
    }

    package static func downloadable(_ names: [String], subfolder: String?) -> [String] {
        names.filter { name in
            guard HuggingFaceHub.isRelativePath(name), !name.hasSuffix(".gguf") else { return false }
            guard let subfolder else { return true }
            return name.hasPrefix("\(subfolder)/")
        }
    }
}

package enum HubResponse {
    package static func fetchFailure(status: Int?, ranged: Bool, resource: String) -> HFError? {
        if status == (ranged ? 206 : 200) { return nil }
        if status == 404 { return .modelNotFound(resource) }
        return .apiError(code: status ?? -1)
    }

    package static func totalSize(contentRange: String?) -> Int64? {
        contentRange?.split(separator: "/").last.flatMap { Int64($0) }
    }

    package static func remoteSize(_ response: URLResponse) -> Int64 {
        (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "X-Linked-Size").flatMap { Int64($0) } ?? response.expectedContentLength
    }
}

package final class HubHeadRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    package func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest
    ) async -> URLRequest? {
        response.value(forHTTPHeaderField: "X-Linked-Size") == nil ? request : nil
    }
}

package enum HubDigest: Equatable, Sendable {
    case sha256(String)
    case gitBlobSHA1(String)

    package init?(_ response: URLResponse) {
        let http = response as? HTTPURLResponse
        self.init(header: http?.value(forHTTPHeaderField: "X-Linked-Etag") ?? http?.value(forHTTPHeaderField: "ETag"))
    }

    package init?(header: String?) {
        guard var value = header?.trimmingCharacters(in: .whitespaces) else { return nil }
        if value.hasPrefix("W/") { value.removeFirst(2) }
        if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") { value = String(value.dropFirst().dropLast()) }
        let hex = value.lowercased()
        guard hex.utf8.allSatisfy({ (48 ... 57).contains($0) || (97 ... 102).contains($0) }) else { return nil }
        switch hex.utf8.count {
        case 64: self = .sha256(hex)
        case 40: self = .gitBlobSHA1(hex)
        default: return nil
        }
    }

    package var hex: String {
        switch self {
        case .sha256(let hex), .gitBlobSHA1(let hex): hex
        }
    }

    package func streamingHasher(prefix: URL?) throws -> SHA256? {
        guard case .sha256 = self else { return nil }
        var hasher = SHA256()
        if let prefix { try Self.feed(&hasher, fileAt: prefix) }
        return hasher
    }

    package func computed(fileAt url: URL) throws -> String {
        switch self {
        case .sha256:
            var hasher = SHA256()
            try Self.feed(&hasher, fileAt: url)
            return Self.hexString(hasher.finalize())
        case .gitBlobSHA1:
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(try HubSidecar.stat(url).size)\0".utf8))
            try Self.feed(&hasher, fileAt: url)
            return Self.hexString(hasher.finalize())
        }
    }

    package func verifyExisting(_ file: HubFile) throws -> Bool {
        if let recorded = HubSidecar.recorded(for: file.destination) {
            if recorded == hex { HubSidecar.logger.info("verified \(file.name) \(hex) from sidecar") }
            return recorded == hex
        }
        let start = ContinuousClock.now
        return try record(computed(fileAt: file.destination), for: file, since: start)
    }

    package func record(_ actual: String, for file: HubFile, since start: ContinuousClock.Instant) throws -> Bool {
        guard actual == hex else {
            HubSidecar.logger.warn("mismatch \(file.name): expected \(hex), got \(actual)")
            return false
        }
        try HubSidecar.write(actual, for: file.destination)
        HubSidecar.logger.info("verified \(file.name) \(hex), hashed in \(start.duration(to: .now))")
        return true
    }

    package static func verifyDownload(_ file: HubFile, digest: HubDigest?, streamed: SHA256?, remoteSize: Int64) throws {
        let start = ContinuousClock.now
        var verified = false
        if let digest {
            let actual = try streamed.map { hexString($0.finalize()) } ?? digest.computed(fileAt: file.destination)
            guard try digest.record(actual, for: file, since: start) else {
                try? FileManager.default.removeItem(at: file.destination)
                throw HFError.digestMismatch(file: file.name, expected: digest.hex, actual: actual)
            }
            verified = true
        } else {
            HubSidecar.logger.warn("unverified \(file.name): the server sent no digest")
        }
        let size = (try? HubSidecar.stat(file.destination).size) ?? 0
        guard HubTransferPlan.isComplete(fileSize: size, remoteSize: remoteSize, verified: verified) else {
            try? FileManager.default.removeItem(at: file.destination)
            throw HFError.downloadFailed(file.name)
        }
    }

    package static func feed<H: HashFunction>(_ hasher: inout H, fileAt url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
    }

    package static func hexString(_ digest: some Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

package struct HubSidecar: Codable, Equatable, Sendable {
    static let logger = BasaltLogger(module: "Hub")

    package let digest: String
    package let size: Int64
    package let modified: Double

    package static func url(for file: URL) -> URL {
        file.deletingLastPathComponent().appendingPathComponent(".\(file.lastPathComponent).sha256")
    }

    package static func stat(_ file: URL) throws -> (size: Int64, modified: Double) {
        let attributes = try FileManager.default.attributesOfItem(atPath: file.resolvingSymlinksInPath().path)
        return (attributes[.size] as? Int64 ?? 0, (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
    }

    package static func write(_ digest: String, for file: URL) throws {
        let current = try stat(file)
        let sidecar = HubSidecar(digest: digest, size: current.size, modified: current.modified)
        try JSONEncoder().encode(sidecar).write(to: url(for: file), options: .atomic)
    }

    package static func recorded(for file: URL) -> String? {
        guard let data = try? Data(contentsOf: url(for: file)),
              let sidecar = try? JSONDecoder().decode(HubSidecar.self, from: data),
              let current = try? stat(file),
              sidecar.size == current.size, sidecar.modified == current.modified else { return nil }
        return sidecar.digest
    }
}
