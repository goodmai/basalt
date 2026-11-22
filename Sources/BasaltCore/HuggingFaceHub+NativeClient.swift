import Foundation
import HuggingFace

package struct HubPublishOptions: Sendable {
    package var repoId: String
    package var isPrivate: Bool
    package var commitMessage: String
    package var token: String?
    package var maxConcurrent: Int

    package init(
        repoId: String,
        isPrivate: Bool = false,
        commitMessage: String = "Upload MLX checkpoint via Basalt",
        token: String? = nil,
        maxConcurrent: Int = 3
    ) {
        self.repoId = repoId
        self.isPrivate = isPrivate
        self.commitMessage = commitMessage
        self.token = token
        self.maxConcurrent = maxConcurrent
    }
}

extension HuggingFaceHub {
    static func parseNativeRepoID(_ repoId: String) throws(HFError) -> HuggingFace::Repo.ID {
        let parts = repoId.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, isRepoId(repoId) else {
            throw .invalidRepoId(repoId)
        }
        return HuggingFace::Repo.ID(namespace: String(parts[0]), name: String(parts[1]))
    }

    nonisolated func makeNativeHubClient(token: String?, cacheRoot: URL = ModelCache.root) throws(HFError) -> HuggingFace::HubClient {
        guard let hostURL = URL(string: baseURL) else {
            throw .invalidURL
        }
        let provider: HuggingFace::TokenProvider = token.map { .fixed(token: $0) } ?? .environment
        let hubCache = HuggingFace::HubCache(cacheDirectory: cacheRoot)
        return HuggingFace::HubClient(
            session: session,
            host: hostURL,
            userAgent: "Basalt/1.0",
            tokenProvider: provider,
            cache: hubCache
        )
    }

    package func downloadSnapshotNative(
        repoId: String,
        subfolder: String? = nil,
        token: String? = nil,
        cacheRoot: URL = ModelCache.root,
        maxConcurrent: Int = 8,
        onFile: @escaping Progress
    ) async throws -> URL {
        let repo = try Self.parseNativeRepoID(repoId)
        let client = try makeNativeHubClient(token: token, cacheRoot: cacheRoot)
        let details = try await modelDetails(repoId: repoId)
        let downloadable = Set(HubRequest.downloadable((details.siblings ?? []).map(\.rfilename), subfolder: subfolder))
        guard !downloadable.isEmpty else {
            throw HFError.modelNotFound(subfolder.map { "\(repoId)/\($0)" } ?? repoId)
        }
        let globs = subfolder.map { ["\($0)/**"] } ?? []
        let snapshotDir = try await client.downloadSnapshot(
            of: repo,
            kind: .model,
            revision: "main",
            matching: globs,
            localFilesOnly: false,
            maxConcurrentDownloads: maxConcurrent,
            progressHandler: nil
        ) { rows in
            for row in rows where downloadable.contains(row.path) {
                let total = row.sizeBytes ?? 0
                let done = Int64((Double(total) * row.fractionCompleted).rounded())
                onFile(row.path, done, total)
            }
        }
        return subfolder.map { snapshotDir.appendingPathComponent($0) } ?? snapshotDir
    }

    package func searchCatalog(
        query: String,
        limit: Int = 20
    ) async throws -> [HFModelInfo] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let bounded = min(max(limit, 1), 50)
        var components = URLComponents(string: "\(apiURL)/models")
        var items: [URLQueryItem] = [
            .init(name: "sort", value: "downloads"),
            .init(name: "direction", value: "-1"),
            .init(name: "limit", value: "\(bounded)"),
            .init(name: "filter", value: "mlx"),
        ]
        if !trimmed.isEmpty {
            items.append(.init(name: "search", value: trimmed))
        }
        components?.queryItems = items
        guard let url = components?.url else {
            throw HFError.invalidURL
        }
        let (data, response) = try await session.data(from: url)
        state = HubState.reduce(state, .fetched(bytes: data.count))
        guard let code = (response as? HTTPURLResponse)?.statusCode else {
            throw HFError.apiError(code: -1)
        }
        guard code == 200 else {
            throw HFError.apiError(code: code)
        }
        return try JSONDecoder().decode([HFModelInfo].self, from: data)
    }
}

extension HuggingFaceHub {
    struct PublishableFile: Sendable {
        let relativePath: String
        let url: URL
        let size: Int64
    }

    package func publishDirectory(
        at directory: URL,
        options: HubPublishOptions,
        onFile: @escaping Progress
    ) async throws -> HubPublishResult {
        let repo = try Self.parseNativeRepoID(options.repoId)
        let client = try makeNativeHubClient(token: options.token)
        let files = try Self.collectPublishableFiles(in: directory)
        guard !files.isEmpty else {
            throw HFError.modelNotFound(directory.path)
        }
        let visibility: HuggingFace::Repo.Visibility = options.isPrivate ? .private : .public
        let created = try await client.createRepo(
            kind: .model,
            name: repo.name,
            organization: repo.namespace,
            visibility: visibility,
            existOk: true
        )
        for item in files {
            onFile(item.relativePath, 0, item.size)
        }
        try await Self.uploadPublishableFiles(
            files,
            client: client,
            repo: repo,
            commitMessage: options.commitMessage,
            maxConcurrent: options.maxConcurrent,
            onFile: onFile
        )
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        let resolvedURL = created.url.isEmpty ? "\(baseURL)/\(options.repoId)" : created.url
        return HubPublishResult(
            repoId: options.repoId,
            url: resolvedURL,
            uploadedFiles: files.count,
            totalBytes: totalBytes
        )
    }

    private static func uploadPublishableFiles(
        _ files: [PublishableFile],
        client: HuggingFace::HubClient,
        repo: HuggingFace::Repo.ID,
        commitMessage: String,
        maxConcurrent: Int,
        onFile: @escaping Progress
    ) async throws {
        let concurrency = max(1, maxConcurrent)
        try await withThrowingTaskGroup(of: (String, Int64).self) { group in
            var active = 0
            for item in files {
                while active >= concurrency {
                    if let (finishedPath, finishedSize) = try await group.next() {
                        onFile(finishedPath, finishedSize, finishedSize)
                        active -= 1
                    }
                }
                group.addTask {
                    _ = try await client.uploadFile(
                        item.url,
                        to: item.relativePath,
                        in: repo,
                        kind: .model,
                        branch: "main",
                        message: commitMessage
                    )
                    return (item.relativePath, item.size)
                }
                active += 1
            }
            for try await (finishedPath, finishedSize) in group {
                onFile(finishedPath, finishedSize, finishedSize)
            }
        }
    }

    static func collectPublishableFiles(in directory: URL) throws -> [PublishableFile] {
        let root = directory.standardizedFileURL
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        let allowedPrefixes = publishableAllowedPrefixes(for: root)
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey]
        ) else {
            return []
        }
        var result: [PublishableFile] = []
        for case let itemURL as URL in enumerator {
            let rel = itemURL.standardizedFileURL.path.hasPrefix(prefix)
                ? String(itemURL.standardizedFileURL.path.dropFirst(prefix.count))
                : itemURL.lastPathComponent
            if rel.split(separator: "/").contains(where: { $0.hasPrefix(".") }) {
                continue
            }
            let resolved = itemURL.resolvingSymlinksInPath().standardizedFileURL
            guard allowedPrefixes.contains(where: { resolved.path.hasPrefix($0) }) else {
                continue
            }
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDir),
                  !isDir.boolValue else {
                continue
            }
            let size = (try? FileManager.default.attributesOfItem(atPath: resolved.path)[.size] as? Int64) ?? 0
            result.append(PublishableFile(relativePath: rel, url: resolved, size: size))
        }
        return result.sorted { $0.relativePath < $1.relativePath }
    }

    private static func publishableAllowedPrefixes(for root: URL) -> [String] {
        let canonicalRoot = root.resolvingSymlinksInPath().standardizedFileURL
        let rootPrefix = canonicalRoot.path.hasSuffix("/") ? canonicalRoot.path : canonicalRoot.path + "/"
        let snapshotsDir = canonicalRoot.deletingLastPathComponent()
        let repoDir = snapshotsDir.deletingLastPathComponent()
        if snapshotsDir.lastPathComponent == "snapshots", repoDir.lastPathComponent.hasPrefix("models--") {
            let blobsPath = repoDir.appendingPathComponent("blobs").path
            let blobsPrefix = blobsPath.hasSuffix("/") ? blobsPath : blobsPath + "/"
            return [rootPrefix, blobsPrefix]
        }
        return [rootPrefix]
    }
}
