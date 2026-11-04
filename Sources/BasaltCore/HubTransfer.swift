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

    package static func isComplete(fileSize: Int64, remoteSize: Int64) -> Bool {
        remoteSize <= 0 || fileSize == remoteSize
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
}
