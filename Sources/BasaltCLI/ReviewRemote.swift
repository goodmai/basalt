import Foundation
import BasaltCore

struct ReviewRemote: Sendable {
    enum Forge: String, Sendable {
        case gitlab, gitea

        var tokenVariable: String {
            switch self {
            case .gitlab: "GITLAB_TOKEN"
            case .gitea: "GITEA_TOKEN"
            }
        }
    }

    struct Failure: Error, Equatable, CustomStringConvertible {
        let description: String
    }

    static let maxDiffBytes = 2 << 20
    static let gitlabPageSize = 100

    let forge: Forge
    let instance: URL
    let repo: String
    let number: Int
    let token: String?

    init(forge: Forge, instance: String, repo: String, number: Int, token: String?) throws(Failure) {
        guard let url = URL(string: instance), let host = url.host, !host.isEmpty, url.query == nil, url.fragment == nil,
              Self.allowsScheme(of: url) else {
            throw Failure(description: "Instance URL must be https:// (http:// only for loopback): \(instance)")
        }
        guard number > 0, !repo.isEmpty else { throw Failure(description: "Need a repository and a positive MR/PR number") }
        self.forge = forge
        self.instance = url
        self.repo = repo
        self.number = number
        self.token = token.flatMap { $0.isEmpty ? nil : $0 }
    }

    static func allowsScheme(of url: URL) -> Bool {
        switch url.scheme?.lowercased() {
        case "https": true
        case "http": ["127.0.0.1", "::1", "localhost"].contains(url.host?.lowercased() ?? "")
        default: false
        }
    }

    var encodedRepo: String {
        repo.addingPercentEncoding(withAllowedCharacters: .alphanumerics.union(CharacterSet(charactersIn: "-._~"))) ?? repo
    }

    var repoPath: String {
        repo.split(separator: "/").map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
    }

    var diffPath: String {
        switch forge {
        case .gitlab: "api/v4/projects/\(encodedRepo)/merge_requests/\(number)/diffs?per_page=\(Self.gitlabPageSize)"
        case .gitea: "api/v1/repos/\(repoPath)/pulls/\(number).diff"
        }
    }

    var commentPath: String {
        switch forge {
        case .gitlab: "api/v4/projects/\(encodedRepo)/merge_requests/\(number)/notes"
        case .gitea: "api/v1/repos/\(repoPath)/issues/\(number)/comments"
        }
    }

    func request(_ path: String) -> URLRequest {
        let base = instance.absoluteString.hasSuffix("/") ? instance.absoluteString : instance.absoluteString + "/"
        var request = URLRequest(url: URL(string: base + path)!)
        if let token {
            switch forge {
            case .gitlab: request.setValue(token, forHTTPHeaderField: "PRIVATE-TOKEN")
            case .gitea: request.setValue("token \(token)", forHTTPHeaderField: "Authorization")
            }
        }
        return request
    }

    func fetchDiff(session: URLSession) async throws -> String {
        guard forge == .gitlab else { return String(decoding: try await send(request(diffPath), session: session), as: UTF8.self) }
        var files: [GitLabFile] = []
        var bytes = 0
        for page in 1... {
            let data = try await send(request(diffPath + "&page=\(page)"), session: session)
            bytes += data.count
            guard bytes <= Self.maxDiffBytes else { throw Failure(description: "Diffs of \(repo) !\(number) exceed \(Self.maxDiffBytes) bytes") }
            let batch = try Self.files(gitlab: data)
            files += batch
            if batch.count < Self.gitlabPageSize { break }
        }
        return files.map { "--- a/\($0.oldPath)\n+++ b/\($0.newPath)\n\($0.diff)" }.joined(separator: "\n")
    }

    func post(_ body: String, session: URLSession) async throws {
        var request = request(commentPath)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["body": SecretRedactor.redact(body)])
        _ = try await send(request, session: session)
    }

    struct GitLabFile: Decodable {
        let oldPath: String
        let newPath: String
        let diff: String
    }

    static func files(gitlab data: Data) throws -> [GitLabFile] {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do { return try decoder.decode([GitLabFile].self, from: data) } catch {
            throw Failure(description: "GitLab returned an unexpected diff payload")
        }
    }

    func send(_ request: URLRequest, session: URLSession) async throws -> Data {
        let (bytes, response) = try await session.bytes(for: request, delegate: NoRedirects())
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
            guard data.count <= Self.maxDiffBytes else {
                throw Failure(description: "Response from \(instance.host ?? "") exceeds \(Self.maxDiffBytes) bytes")
            }
        }
        guard (200..<300).contains(status) else {
            let detail = SecretRedactor.redact(String(decoding: data.prefix(300), as: UTF8.self))
            throw Failure(description: "\(forge.rawValue) answered HTTP \(status) for \(request.url?.path ?? ""): \(detail)")
        }
        return data
    }
}

final class NoRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest
    ) async -> URLRequest? {
        nil
    }
}
