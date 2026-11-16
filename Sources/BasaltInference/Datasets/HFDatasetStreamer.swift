import CryptoKit
import Foundation
import BasaltCore

public protocol CorpusHTTPTransport: Sendable {
    func fetch(_ request: URLRequest) async throws -> (Int, Data)
}

public struct URLSessionCorpusTransport: CorpusHTTPTransport {
    public init() {}

    public func fetch(_ request: URLRequest) async throws -> (Int, Data) {
        let (data, response) = try await URLSession.shared.data(for: request)
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, data)
    }
}

public struct FetchedCorpus: Sendable, Equatable {
    public let jsonl: String
    public let rowCount: Int
    public let repo: String?
    public let subset: String?
    public let split: String?
    public let cacheFile: URL?
}

public enum HFDatasetStreamer {
    private static let serverBase = "https://datasets-server.huggingface.co"
    private static let pageSize = 100

    private struct CacheMeta: Codable {
        let config: String
        let split: String
    }

    public static func load(
        _ spec: CorpusSourceSpec,
        transport: any CorpusHTTPTransport = URLSessionCorpusTransport(),
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async throws(BasaltError) -> FetchedCorpus {
        try spec.validate(prefix: "train")
        switch spec.location {
        case .localFile(let path):
            let expanded = (path as NSString).expandingTildeInPath
            let data: Data
            do {
                data = try Data(contentsOf: URL(fileURLWithPath: expanded))
            } catch {
                throw .invalidRequestStructure(details: "train: could not read local corpus \(path): \(error.localizedDescription)")
            }
            let text = String(decoding: data, as: UTF8.self)
            let rows = try NemotronRowMapper.parseJSONLines(text)
            return FetchedCorpus(jsonl: text, rowCount: rows.count, repo: nil, subset: spec.subset, split: spec.split, cacheFile: nil)
        case .preset(let preset):
            return try await fetchRemote(repo: preset.repoID, spec: spec, transport: transport, home: home, environment: environment)
        case .huggingFace(let repo):
            return try await fetchRemote(repo: repo, spec: spec, transport: transport, home: home, environment: environment)
        }
    }

    private static func fetchRemote(
        repo: String,
        spec: CorpusSourceSpec,
        transport: any CorpusHTTPTransport,
        home: URL,
        environment: [String: String]
    ) async throws(BasaltError) -> FetchedCorpus {
        let targetFetch = min(max(spec.maxRows * 2, spec.maxRows + 16), 100_000)
        let cacheURL = cachePath(
            repo: repo,
            config: spec.resolvedSubset ?? "*",
            split: spec.resolvedSplit ?? "*",
            targetRows: targetFetch,
            home: home
        )
        let metaURL = cacheURL.appendingPathExtension("meta.json")
        if let cachedData = try? Data(contentsOf: cacheURL), !cachedData.isEmpty {
            let text = String(decoding: cachedData, as: UTF8.self)
            let parsed = try NemotronRowMapper.parseJSONLines(text)
            if !parsed.isEmpty {
                let meta = (try? Data(contentsOf: metaURL)).flatMap { try? JSONDecoder().decode(CacheMeta.self, from: $0) }
                return FetchedCorpus(
                    jsonl: text,
                    rowCount: parsed.count,
                    repo: repo,
                    subset: meta?.config ?? spec.resolvedSubset ?? "default",
                    split: meta?.split ?? spec.resolvedSplit ?? "train",
                    cacheFile: cacheURL
                )
            }
        }
        let authToken = resolveHFToken(home: home, environment: environment)
        let (config, split) = try await resolveConfigAndSplit(
            repo: repo,
            preferredSubset: spec.resolvedSubset,
            preferredSplit: spec.resolvedSplit,
            authToken: authToken,
            transport: transport
        )
        let (jsonl, count) = try await streamRowsJSONL(
            repo: repo,
            config: config,
            split: split,
            limit: targetFetch,
            authToken: authToken,
            transport: transport
        )
        try persistCache(jsonl, meta: CacheMeta(config: config, split: split), to: cacheURL)
        return FetchedCorpus(jsonl: jsonl, rowCount: count, repo: repo, subset: config, split: split, cacheFile: cacheURL)
    }

    private static func resolveConfigAndSplit(
        repo: String,
        preferredSubset: String?,
        preferredSplit: String?,
        authToken: String?,
        transport: any CorpusHTTPTransport
    ) async throws(BasaltError) -> (config: String, split: String) {
        if let preferredSubset, let preferredSplit {
            return (preferredSubset, preferredSplit)
        }
        var comps = URLComponents(string: "\(serverBase)/splits")
        comps?.queryItems = [URLQueryItem(name: "dataset", value: repo)]
        guard let url = comps?.url else {
            throw .invalidRequestStructure(details: "train: invalid dataset identifier \(repo)")
        }
        let (status, data) = try await send(url: url, authToken: authToken, repo: repo, transport: transport)
        guard status == 200 else {
            throw httpError(status: status, repo: repo, data: data)
        }
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let splits = root["splits"] as? [[String: Any]], !splits.isEmpty else {
            return (preferredSubset ?? "default", preferredSplit ?? "train")
        }
        let matched = splits.first { entry in
            let cfg = entry["config"] as? String
            let sp = entry["split"] as? String
            return (preferredSubset == nil || cfg == preferredSubset) && (preferredSplit == nil || sp == preferredSplit)
        } ?? splits[0]
        let config = preferredSubset ?? (matched["config"] as? String) ?? "default"
        let split = preferredSplit ?? (matched["split"] as? String) ?? "train"
        return (config, split)
    }

    private static func streamRowsJSONL(
        repo: String,
        config: String,
        split: String,
        limit: Int,
        authToken: String?,
        transport: any CorpusHTTPTransport
    ) async throws(BasaltError) -> (String, Int) {
        var buffer = Data()
        var count = 0
        var offset = 0
        while count < limit {
            do { try Task.checkCancellation() } catch { throw .cancelled }
            let length = min(pageSize, limit - count)
            var comps = URLComponents(string: "\(serverBase)/rows")
            comps?.queryItems = [
                URLQueryItem(name: "dataset", value: repo),
                URLQueryItem(name: "config", value: config),
                URLQueryItem(name: "split", value: split),
                URLQueryItem(name: "offset", value: String(offset)),
                URLQueryItem(name: "length", value: String(length)),
            ]
            guard let url = comps?.url else { break }
            let (status, data) = try await send(url: url, authToken: authToken, repo: repo, transport: transport)
            guard status == 200 else {
                throw httpError(status: status, repo: repo, data: data)
            }
            guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let page = root["rows"] as? [[String: Any]], !page.isEmpty else {
                break
            }
            for item in page {
                let rowObj = (item["row"] as? [String: Any]) ?? item
                if let encoded = try? JSONSerialization.data(withJSONObject: rowObj, options: [.sortedKeys]) {
                    buffer.append(encoded)
                    buffer.append(0x0A)
                    count += 1
                }
            }
            offset += page.count
            if page.count < length { break }
        }
        guard count > 0 else {
            throw .invalidRequestStructure(details: "train: dataset \(repo) (\(config)/\(split)) returned 0 rows")
        }
        return (String(decoding: buffer, as: UTF8.self), count)
    }
}

extension HFDatasetStreamer {
    private static func send(
        url: URL,
        authToken: String?,
        repo: String,
        transport: any CorpusHTTPTransport
    ) async throws(BasaltError) -> (Int, Data) {
        var request = URLRequest(url: url)
        request.timeoutInterval = 60
        if let authToken, !authToken.isEmpty {
            request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        }
        do {
            return try await transport.fetch(request)
        } catch is CancellationError {
            throw .cancelled
        } catch {
            throw .inferenceError("train: failed fetching Hugging Face dataset \(repo): \(error.localizedDescription)")
        }
    }

    private static func httpError(status: Int, repo: String, data: Data) -> BasaltError {
        if status == 401 || status == 403 {
            return .invalidRequestStructure(
                details: "train: Hugging Face returned HTTP \(status) for \(repo); accept the dataset license on huggingface.co/datasets/\(repo) and set HF_TOKEN"
            )
        }
        let preview = String(decoding: data.prefix(200), as: UTF8.self)
        return .inferenceError("train: Hugging Face datasets-server returned HTTP \(status) for \(repo): \(preview)")
    }

    private static func resolveHFToken(home: URL, environment: [String: String]) -> String? {
        for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
            if let value = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                return value
            }
        }
        let tokenFile = home.appending(path: ".cache/huggingface/token")
        if let raw = try? String(contentsOf: tokenFile, encoding: .utf8) {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    private static func cachePath(repo: String, config: String, split: String, targetRows: Int, home: URL) -> URL {
        let key = "\(repo)|\(config)|\(split)|\(targetRows)"
        let digest = SHA256.hash(data: Data(key.utf8)).map { String(format: "%02x", $0) }.joined()
        return home.appending(path: ".basalt/datasets").appending(component: "\(digest.prefix(24)).jsonl")
    }

    private static func persistCache(_ jsonl: String, meta: CacheMeta, to url: URL) throws(BasaltError) {
        let dir = url.deletingLastPathComponent()
        let staging = dir.appending(component: "\(url.lastPathComponent).tmp-\(UUID().uuidString)")
        let metaURL = url.appendingPathExtension("meta.json")
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try Data(jsonl.utf8).write(to: staging, options: .atomic)
            _ = try? FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: staging, to: url)
            if let metaData = try? JSONEncoder().encode(meta) {
                try? metaData.write(to: metaURL, options: .atomic)
            }
        } catch {
            _ = try? FileManager.default.removeItem(at: staging)
            throw .inferenceError("train: could not cache dataset shard at \(url.path): \(error.localizedDescription)")
        }
    }
}
