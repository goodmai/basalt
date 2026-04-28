import Foundation

// MARK: — HuggingFace Hub API Client
//
// Docs: https://huggingface.co/docs/hub/api
// Lists mlx-community/gemma-4-* models and downloads safetensors to
// the standard HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/

// MARK: — API Models

struct HFModelInfo: Codable, Sendable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let tags: [String]?

    var org: String  { id.components(separatedBy: "/").first ?? id }
    var name: String { id.components(separatedBy: "/").last  ?? id }

    // Extracts "4bit" / "8bit" / "bf16" from the model name
    var quantization: String {
        let parts = name.lowercased().components(separatedBy: "-")
        return parts.last(where: { $0.hasSuffix("bit") || $0 == "bf16" }) ?? "unknown"
    }

    // Approximate parameter count from name: e2b→2B, 31b→31B
    var parameterSize: String {
        let parts = name.lowercased().components(separatedBy: "-")
        if let sizeTag = parts.first(where: { $0.count <= 4 && $0.contains("b") && $0 != "it" }) {
            let digits = sizeTag.filter { $0.isNumber || $0 == "." }
            if !digits.isEmpty { return "\(digits)B" }
        }
        return "?"
    }

    var formattedDownloads: String {
        guard let d = downloads else { return "—" }
        return d >= 1_000 ? String(format: "%.0fk", Double(d) / 1_000) : "\(d)"
    }
}

struct HFModelDetails: Codable, Sendable {
    let id: String
    let siblings: [HFSibling]?
    let downloads: Int?
    let likes: Int?
}

struct HFSibling: Codable, Sendable {
    let rfilename: String
    // Present in some responses
    let size: Int?

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
    ///  - GemmaServer downloader layout: snapshots/main (direct)
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

        // Fallback: GemmaServer downloader writes directly to snapshots/main
        return modelRoot.appendingPathComponent("snapshots/main")
    }

    public static func isDownloaded(repoId: String) -> Bool {
        let dir = cacheDir(for: repoId)
        let marker = dir.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: marker.path)
    }
}

// MARK: — Hub

actor HuggingFaceHub {

    private let session: URLSession
    private let baseURL = "https://huggingface.co"
    private let apiURL  = "https://huggingface.co/api"

    init(configuration: URLSessionConfiguration = .default) {
        let config = configuration
        config.timeoutIntervalForRequest  = 30
        config.timeoutIntervalForResource = 3600   // 1h for large downloads
        session = URLSession(configuration: config)
    }

    // MARK: — Listing

    /// Lists MLX community models sorted by downloads.
    func listModels(
        author: String = "mlx-community",
        search: String = "gemma-4",
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

    /// Detailed info for a single repo including file list.
    func modelDetails(repoId: String) async throws -> HFModelDetails {
        let url = URL(string: "\(apiURL)/models/\(repoId)")!
        let (data, response) = try await session.data(from: url)
        
        switch (response as? HTTPURLResponse)?.statusCode {
        case 200:
            return try JSONDecoder().decode(HFModelDetails.self, from: data)
        default:
            throw HFError.modelNotFound(repoId)
        }
    }

    // MARK: — Download

    /// Downloads all files for a model repo into the HF cache.
    func download(
        repoId: String,
        token: String? = nil,
        onFile: @Sendable @escaping (String, Int64, Int64) -> Void
    ) async throws -> URL {
        let details  = try await modelDetails(repoId: repoId)
        let siblings = details.siblings ?? []
        let destDir  = ModelCache.cacheDir(for: repoId)

        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let filesToDownload = siblings.map(\.rfilename).filter { name in
            !name.hasSuffix(".gguf")
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for filename in filesToDownload {
                group.addTask {
                    let dest = destDir.appendingPathComponent(filename)
                    let fileURL = URL(string: "\(self.baseURL)/\(repoId)/resolve/main/\(filename)")!
                    
                    // Retry loop
                    var success = false
                    var attempts = 0
                    let maxAttempts = 5
                    
                    while !success && attempts < maxAttempts {
                        attempts += 1
                        do {
                            try await self.downloadFileWithResume(
                                url: fileURL,
                                destination: dest,
                                filename: filename,
                                token: token,
                                onProgress: onFile
                            )
                            success = true
                        } catch {
                            if attempts >= maxAttempts { throw error }
                            // Exponential backoff
                            let delay = UInt64(pow(2.0, Double(attempts))) * 1_000_000_000
                            try await Task.sleep(nanoseconds: delay)
                        }
                    }
                }
            }
            for try await _ in group {}
        }
        return destDir
    }

    // MARK: — Private

    /// Downloads a single file with resume support (Range header).
    private func downloadFileWithResume(
        url: URL,
        destination: URL,
        filename: String,
        token: String?,
        onProgress: @Sendable @escaping (String, Int64, Int64) -> Void
    ) async throws {
        let fileManager = FileManager.default
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)

        let existingSize: Int64
        if fileManager.fileExists(atPath: destination.path) {
            let attrs = try fileManager.attributesOfItem(atPath: destination.path)
            existingSize = attrs[.size] as? Int64 ?? 0
        } else {
            existingSize = 0
        }

        // Get remote file size
        var headRequest = URLRequest(url: url)
        headRequest.httpMethod = "HEAD"
        if let token { headRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        
        let (_, headResponse) = try await session.data(for: headRequest)
        let totalSize = headResponse.expectedContentLength
        
        // Use pattern matching to decide next action based on file state
        enum DownloadAction {
            case skip
            case resume(from: Int64)
            case startNew
        }
        
        let action: DownloadAction = switch (existingSize, totalSize) {
        case (let s, let t) where s > 0 && t > 0 && s == t:
            .skip
        case (let s, let t) where s > 0 && s < t:
            .resume(from: s)
        default:
            .startNew
        }

        switch action {
        case .skip:
            onProgress(filename, totalSize, totalSize)
            return
            
        case .resume(let offset):
            var req = URLRequest(url: url)
            if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            req.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            
            let (stream, response) = try await session.bytes(for: req)
            guard let http = response as? HTTPURLResponse, http.statusCode == 206 else {
                // If 206 fails, fallback to startNew
                try await downloadFileWithResume(url: url, destination: destination, filename: filename, token: token, onProgress: onProgress)
                return
            }
            try await saveStream(stream, to: destination, isResume: true, existingSize: offset, totalSize: offset + http.expectedContentLength, filename: filename, onProgress: onProgress)
            
        case .startNew:
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            var req = URLRequest(url: url)
            if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            
            let (stream, response) = try await session.bytes(for: req)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw HFError.downloadFailed(filename)
            }
            try await saveStream(stream, to: destination, isResume: false, existingSize: 0, totalSize: http.expectedContentLength, filename: filename, onProgress: onProgress)
        }
    }

    private func saveStream(
        _ stream: URLSession.AsyncBytes,
        to destination: URL,
        isResume: Bool,
        existingSize: Int64,
        totalSize: Int64,
        filename: String,
        onProgress: @Sendable @escaping (String, Int64, Int64) -> Void
    ) async throws {
        if !isResume {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        let handle = isResume ? try FileHandle(forWritingTo: destination) : try FileHandle(forWritingTo: destination)
        if isResume { try handle.seekToEnd() }
        
        defer { try? handle.close() }

        var downloaded = isResume ? existingSize : 0
        var buffer = [UInt8]()
        buffer.reserveCapacity(65_536)
        
        for try await byte in stream {
            buffer.append(byte)
            if buffer.count >= 65_536 {
                try handle.write(contentsOf: buffer)
                downloaded += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                onProgress(filename, downloaded, totalSize)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            downloaded += Int64(buffer.count)
            onProgress(filename, downloaded, totalSize)
        }
    }
}

// MARK: — Errors

enum HFError: Error, LocalizedError {
    case invalidURL
    case apiError(code: Int)
    case modelNotFound(String)
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:              return "Invalid HuggingFace URL."
        case .apiError(let code):      return "HuggingFace API error: HTTP \(code)"
        case .modelNotFound(let id):   return "Model not found: \(id)"
        case .downloadFailed(let f):   return "Download failed: \(f)"
        }
    }
}
