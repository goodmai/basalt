import Foundation

// MARK: — HuggingFace Hub API Client
//
// Docs: https://huggingface.co/docs/hub/api
// Lists mlx-community/gemma-4-* models and downloads safetensors to
// the standard HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/

// MARK: — API Models

package struct HFModelInfo: Codable, Sendable {
    package let id: String
    package let downloads: Int?
    package let likes: Int?
    package let tags: [String]?

    package var org: String  { id.components(separatedBy: "/").first ?? id }
    package var name: String { id.components(separatedBy: "/").last  ?? id }

    // Extracts "4bit" / "8bit" / "nvfp4" / "dq" / "fp8" / "bf16" from the model name
    package var quantization: String {
        let lower = name.lowercased()
        let parts = lower.components(separatedBy: CharacterSet(charactersIn: "-_."))
        
        if parts.contains("nvfp4") || lower.contains("nvfp4") { return "nvfp4" }
        if parts.contains("dq") || lower.contains("-dq") || lower.contains("_dq") { return "dq" }
        if parts.contains("fp8") || lower.contains("fp8") { return "fp8" }
        if parts.contains("bf16") || lower.contains("bf16") { return "bf16" }
        if parts.contains("fp16") || lower.contains("fp16") { return "fp16" }
        if lower.contains("oq4e") { return "oq4e" }
        if lower.contains("oq6e") { return "oq6e" }
        if lower.contains("oq8e") { return "oq8e" }
        
        if let bitTag = parts.last(where: { $0.hasSuffix("bit") }) {
            return bitTag
        }
        if let qTag = parts.first(where: { $0.hasPrefix("q") && $0.count <= 6 && $0 != "qwen" && $0 != "qwen2" && $0 != "qwen3" }) {
            return qTag
        }
        return "unknown"
    }

    // Approximate parameter count from name: e2b→2B, 31b→31B, 27b→27B, 35b→35B
    package var parameterSize: String {
        let lower = name.lowercased()
        let parts = lower.components(separatedBy: CharacterSet(charactersIn: "-_."))
        if let sizeTag = parts.first(where: {
            $0.count <= 6 && $0.hasSuffix("b") && $0 != "it" && $0 != "hub" && $0 != "web" && $0 != "bit" && !$0.hasSuffix("bit")
        }) {
            let digits = sizeTag.filter { $0.isNumber || $0 == "." }
            if !digits.isEmpty { return "\(digits)B" }
        }
        return "?"
    }

    package var formattedDownloads: String {
        guard let d = downloads else { return "—" }
        return d >= 1_000 ? String(format: "%.0fk", Double(d) / 1_000) : "\(d)"
    }
}

package struct HFModelDetails: Codable, Sendable {
    package let id: String
    package let siblings: [HFSibling]?
    package let downloads: Int?
    package let likes: Int?
}

package struct HFSibling: Codable, Sendable {
    package let rfilename: String
    // Present in some responses
    package let size: Int?

    enum CodingKeys: String, CodingKey {
        case rfilename, size
    }
}

// MARK: — Cache

public struct ModelCache: Sendable {
    public static let root: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }()

    /// Resolves the snapshot directory for a repo ID.
    /// Supports both:
    ///  - Standard HF hub layout: refs/main → snapshots/{hash}
    ///  - Gem downloader layout: snapshots/main (direct)
    public static func cacheDir(for repoId: String) -> URL {
        let sanitized = repoId.replacingOccurrences(of: "/", with: "--")
        let modelRoot = root.appendingPathComponent("models--\(sanitized)")

        // Prefer standard HF hub layout: read refs/main → snapshot hash
        let refsMain = modelRoot.appendingPathComponent("refs/main")
        if let hash = try? String(contentsOf: refsMain, encoding: .utf8)
                            .trimmingCharacters(in: .whitespacesAndNewlines),
           !hash.isEmpty {
            return modelRoot.appendingPathComponent("snapshots/\(hash)")
        }

        // Fallback: Gem downloader writes directly to snapshots/main
        return modelRoot.appendingPathComponent("snapshots/main")
    }

    package static func claudeLocalId(from hfId: String) -> String {
        "claude-local/" + hfId.replacingOccurrences(of: "/", with: "--")
    }

    /// Inverse of `ModelsController.claudeLocalId` — strips the `claude-local/` prefix and
    /// restores the "/" separator. Plain HF IDs (or aliases) pass through unchanged.
    package static func hfRepoId(from modelId: String) -> String {
        if modelId.hasPrefix("claude-local/") {
            let body = String(modelId.dropFirst("claude-local/".count))
            return body.replacingOccurrences(of: "--", with: "/")
        }
        return modelId
    }

    /// Strips local path prefixes, snapshot dirs, or claude-local/ prefixes to yield the canonical model repo ID
    public static func canonicalRepoId(from raw: String) -> String {
        let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.hasPrefix("claude-local/") {
            return hfRepoId(from: cleaned)
        }
        if cleaned.contains("models--") {
            let parts = cleaned.components(separatedBy: "models--")
            if let last = parts.last?.components(separatedBy: "/").first {
                return last.replacingOccurrences(of: "--", with: "/")
            }
        }
        if cleaned.hasPrefix("/") || cleaned.hasPrefix("./") {
            let url = URL(fileURLWithPath: cleaned)
            let lastComponent = url.lastPathComponent
            if lastComponent.contains("--") {
                return lastComponent.replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/")
            }
            return lastComponent
        }
        return cleaned
    }

    public static func isDownloaded(repoId: String) -> Bool {
        let dir = cacheDir(for: repoId)
        let marker = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: marker.path)
    }

    /// True when the directory holds a config.json that actually parses.
    ///
    /// Existence alone is not enough: an interrupted download leaves a truncated
    /// or duplicated config that only fails much later, deep inside model loading.
    public static func hasUsableConfig(at dir: URL) -> Bool {
        config(at: dir) != nil
    }

    public static func config(at dir: URL) -> [String: Any]? {
        json("config.json", at: dir)
    }

    public static func generationConfig(at dir: URL) -> [String: Any]? {
        json("generation_config.json", at: dir)
    }

    private static func json(_ name: String, at dir: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent(name)) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data, options: .json5Allowed)) as? [String: Any]
    }

    package static let quantFolders = ["4bit", "8bit", "6bit", "2bit", "bf16", "fp16"]

    /// Resolves a repo ID (plus optional quantization subfolder) to a loadable
    /// directory, or nil when nothing usable is cached.
    ///
    /// Shared by `serve` and the benchmark so they cannot disagree about which
    /// weights are being run — the benchmark previously returned the snapshot
    /// root, which for a repo shipping only per-quant subfolders contains no
    /// config.json at all, making those models unbenchmarkable.
    ///
    /// An explicit `quant` is never silently substituted: reporting numbers for
    /// 8-bit when 4-bit was requested would be worse than reporting nothing.
    public static func resolve(repoId: String, quant: String? = nil) -> URL? {
        if repoId.hasPrefix("/") || repoId.hasPrefix(".") {
            return URL(fileURLWithPath: repoId)
        }
        guard repoId.contains("/") else { return nil }

        let snapshot = cacheDir(for: repoId)
        if let quant {
            let target = snapshot.appendingPathComponent(quant)
            return hasUsableConfig(at: target) ? target : nil
        }
        if hasUsableConfig(at: snapshot) { return snapshot }

        for folder in quantFolders {
            let target = snapshot.appendingPathComponent(folder)
            if hasUsableConfig(at: target) { return target }
        }
        return nil
    }
}

enum HubEvent {
    case fetched(bytes: Int)
    case transferStarted(URL, generation: Int, Task<Void, any Error>)
    case transferFinished(URL, generation: Int)
}

struct HubState: Sendable {
    struct Transfer: Sendable {
        let generation: Int
        let task: Task<Void, any Error>
    }

    var bytesFetched = 0
    var generation = 0
    var transfers: [URL: Transfer] = [:]

    func checkInvariants(after previous: HubState) -> [String] {
        (bytesFetched < previous.bytesFetched ? ["bytesFetched went back from \(previous.bytesFetched) to \(bytesFetched)"] : [])
            + (generation < previous.generation ? ["the generation went back from \(previous.generation) to \(generation)"] : [])
    }

    static func reduce(_ state: HubState, _ event: HubEvent) -> HubState {
        var next = state
        switch event {
        case .fetched(let bytes):
            next.bytesFetched += bytes
        case .transferStarted(let destination, let generation, let task):
            guard state.transfers[destination] == nil, generation > state.generation else { return state }
            next.generation = generation
            next.transfers[destination] = Transfer(generation: generation, task: task)
        case .transferFinished(let destination, let generation):
            guard state.transfers[destination]?.generation == generation else { return state }
            next.transfers[destination] = nil
        }
        return next
    }
}

package actor HuggingFaceHub {
    package typealias Progress = @Sendable (String, Int64, Int64) -> Void

    private let session: URLSession
    private let baseURL: String
    private let apiURL: String
    private(set) var state = HubState()
    package var bytesFetched: Int { state.bytesFetched }

    package init(configuration: URLSessionConfiguration = .default, endpoint: String = "https://huggingface.co") {
        baseURL = endpoint
        apiURL = endpoint + "/api"
        let config = configuration
        config.timeoutIntervalForRequest  = 30
        config.timeoutIntervalForResource = 24 * 3600
        session = URLSession(configuration: config)
    }
}

extension HuggingFaceHub {
    package func listModels(
        author: String = "mlx-community",
        search: String = "",
        quant: String? = nil,
        limit: Int = 20
    ) async throws -> [HFModelInfo] {
        var components = URLComponents(string: "\(apiURL)/models")!
        components.queryItems = [
            .init(name: "author",    value: author),
            .init(name: "search",    value: search),
            .init(name: "sort",      value: "downloads"),
            .init(name: "direction", value: "-1"),
            .init(name: "limit",     value: "\(limit)"),
            .init(name: "library",   value: "mlx"),
        ]
        guard let url = components.url else {
            throw HFError.invalidURL
        }

        let (data, response) = try await session.data(from: url)

        switch (response as? HTTPURLResponse)?.statusCode {
        case 200:
            var models = try JSONDecoder().decode([HFModelInfo].self, from: data)
            if let quant {
                models = models.filter { $0.quantization.lowercased() == quant.lowercased() }
            }
            return models
        case let code?:
            throw HFError.apiError(code: code)
        case .none:
            throw HFError.apiError(code: -1)
        }
    }

    package func modelDetails(repoId: String) async throws -> HFModelDetails {
        guard Self.isRepoId(repoId), let url = URL(string: "\(apiURL)/models/\(repoId)") else {
            throw HFError.invalidRepoId(repoId)
        }
        let (data, response) = try await session.data(from: url)
        state = HubState.reduce(state, .fetched(bytes: data.count))

        switch (response as? HTTPURLResponse)?.statusCode {
        case 200:
            return try JSONDecoder().decode(HFModelDetails.self, from: data)
        default:
            throw HFError.modelNotFound(repoId)
        }
    }

    package func fetch(repoId: String, path: String, range: ClosedRange<Int>? = nil) async throws -> (data: Data, fileSize: Int64) {
        guard Self.isRepoId(repoId), Self.isRelativePath(path),
              let url = HubRequest.resolveURL(endpoint: baseURL, repoId: repoId, path: path) else {
            throw HFError.invalidRepoId("\(repoId)/\(path)")
        }
        let request = HubRequest.file(url, token: nil, range: range.map(HubRequest.rangeHeader))
        let (bytes, response) = try await session.bytes(for: request)
        let http = response as? HTTPURLResponse
        if let failure = HubResponse.fetchFailure(status: http?.statusCode, ranged: range != nil, resource: "\(repoId)/\(path)") {
            bytes.task.cancel()
            throw failure
        }
        var data = Data()
        for try await byte in bytes.prefix(range?.count ?? .max) { data.append(byte) }
        state = HubState.reduce(state, .fetched(bytes: data.count))
        let total = HubResponse.totalSize(contentRange: http?.value(forHTTPHeaderField: "Content-Range"))
        return (data, range == nil ? Int64(data.count) : total ?? -1)
    }

    package func safetensorsHeader(repoId: String, path: String) async throws -> Result<ModelInspector.SafetensorsHeader, ModelInspector.ShardProblem> {
        let (prefix, fileSize) = try await fetch(repoId: repoId, path: path, range: 0 ... 7)
        guard let length = ModelInspector.safetensorsHeaderLength(prefix, fileSize: fileSize) else {
            return .failure(.malformedHeader)
        }
        let (header, _) = try await fetch(repoId: repoId, path: path, range: 8 ... 7 + length)
        guard header.count == length else { return .failure(.malformedHeader) }
        return ModelInspector.parseSafetensorsHeader(header, fileSize: fileSize)
    }
}

extension HuggingFaceHub {
    package func download(
        repoId: String,
        subfolder: String? = nil,
        token: String? = nil,
        onFile: @escaping Progress
    ) async throws -> URL {
        let details  = try await modelDetails(repoId: repoId)
        let destDir  = ModelCache.cacheDir(for: repoId)

        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let filesToDownload = HubRequest.downloadable((details.siblings ?? []).map(\.rfilename), subfolder: subfolder)
        guard !filesToDownload.isEmpty else {
            throw HFError.modelNotFound(subfolder.map { "\(repoId)/\($0)" } ?? repoId)
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for filename in filesToDownload {
                group.addTask {
                    guard let url = HubRequest.resolveURL(endpoint: self.baseURL, repoId: repoId, path: filename) else {
                        throw HFError.downloadFailed(filename)
                    }
                    let file = HubFile(name: filename, url: url, destination: destDir.appendingPathComponent(filename))
                    try await self.transfer(file, token: token, onProgress: onFile)
                }
            }
            for try await _ in group {}
        }
        return subfolder.map { destDir.appendingPathComponent($0) } ?? destDir
    }

    private func transfer(_ file: HubFile, token: String?, onProgress: @escaping Progress) async throws {
        let joined = state.transfers[file.destination] != nil
        let task = transferTask(file, token: token, onProgress: onProgress)
        try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
        if joined {
            let size = try Self.fileSize(at: file.destination)
            onProgress(file.name, size, size)
        }
    }

    func transferTask(_ file: HubFile, token: String?, onProgress: @escaping Progress) -> Task<Void, any Error> {
        if let running = state.transfers[file.destination] { return running.task }
        let generation = state.generation + 1
        let task = Task {
            defer { state = HubState.reduce(state, .transferFinished(file.destination, generation: generation)) }
            try await downloadWithRetry(file, token: token, onProgress: onProgress)
        }
        state = HubState.reduce(state, .transferStarted(file.destination, generation: generation, task))
        return task
    }

    private func downloadWithRetry(_ file: HubFile, token: String?, onProgress: @escaping Progress) async throws {
        var attempt = 1
        while true {
            do {
                return try await downloadFileWithResume(file, token: token, onProgress: onProgress)
            } catch {
                guard let delay = HubRetry.delay(afterFailedAttempt: attempt) else { throw error }
                try await Task.sleep(for: delay)
                attempt += 1
            }
        }
    }

    private func downloadFileWithResume(_ file: HubFile, token: String?, onProgress: @escaping Progress) async throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: file.destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let existingSize = try Self.fileSize(at: file.destination)

        let (_, headResponse) = try await session.data(for: HubRequest.file(file.url, token: token, method: "HEAD"))
        let remoteSize = headResponse.expectedContentLength
        let plan = HubTransferPlan(existingSize: existingSize, remoteSize: remoteSize)

        switch plan {
        case .skip:
            onProgress(file.name, remoteSize, remoteSize)
            return
        case .startNew where fileManager.fileExists(atPath: file.destination.path):
            try fileManager.removeItem(at: file.destination)
        case .startNew, .resume:
            break
        }

        let (stream, response) = try await session.bytes(for: HubRequest.file(file.url, token: token, range: plan.rangeHeader))
        guard let http = response as? HTTPURLResponse, http.statusCode == plan.expectedStatus else {
            try? fileManager.removeItem(at: file.destination)
            throw HFError.downloadFailed(file.name)
        }
        try await saveStream(stream, to: file, plan: plan, contentLength: http.expectedContentLength, onProgress: onProgress)

        guard HubTransferPlan.isComplete(fileSize: (try? Self.fileSize(at: file.destination)) ?? 0, remoteSize: remoteSize) else {
            try? fileManager.removeItem(at: file.destination)
            throw HFError.downloadFailed(file.name)
        }
    }

    private func saveStream(
        _ stream: URLSession.AsyncBytes,
        to file: HubFile,
        plan: HubTransferPlan,
        contentLength: Int64,
        onProgress: @escaping Progress
    ) async throws {
        if plan.offset == 0 {
            FileManager.default.createFile(atPath: file.destination.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: file.destination)
        if plan.offset > 0 { try handle.seekToEnd() }
        defer { try? handle.close() }

        let totalSize = plan.totalSize(contentLength: contentLength)
        var downloaded = plan.offset
        var buffer = [UInt8]()
        buffer.reserveCapacity(65_536)

        for try await byte in stream {
            buffer.append(byte)
            if buffer.count >= 65_536 {
                try handle.write(contentsOf: buffer)
                downloaded += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                onProgress(file.name, downloaded, totalSize)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            downloaded += Int64(buffer.count)
            onProgress(file.name, downloaded, totalSize)
        }
    }

    private static func fileSize(at url: URL) throws -> Int64 {
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        return try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64 ?? 0
    }
}

extension HuggingFaceHub {
    package static func isRepoId(_ id: String) -> Bool {
        let parts = id.split(separator: "/", omittingEmptySubsequences: false)
        return (1 ... 2).contains(parts.count) && parts.allSatisfy { part in
            !part.isEmpty && part.first != "." && part.count <= 96
                && part.unicodeScalars.allSatisfy { CharacterSet.alphanumerics.contains($0) && $0.isASCII || "-_.".unicodeScalars.contains($0) }
        }
    }

    package static func isRelativePath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.contains("\\") && !path.contains("\0") && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

package enum HFError: Error, LocalizedError {
    case invalidURL
    case invalidRepoId(String)
    case apiError(code: Int)
    case modelNotFound(String)
    case downloadFailed(String)

    package var errorDescription: String? {
        switch self {
        case .invalidURL:              return "Invalid HuggingFace URL."
        case .invalidRepoId(let id):   return "Not a HuggingFace repo id: \(id) (expected owner/name with letters, digits, '-', '_' or '.')"
        case .apiError(let code):      return "HuggingFace API error: HTTP \(code)"
        case .modelNotFound(let id):   return "Model not found: \(id)"
        case .downloadFailed(let f):   return "Download failed: \(f)"
        }
    }
}
