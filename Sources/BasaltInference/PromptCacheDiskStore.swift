import Foundation
import CryptoKit
import MLX
import MLXLMCommon
import BasaltCore

struct PromptCacheDiskEntry: Sendable, Equatable {
    let name: String
    let fingerprint: String
    let namespace: String
    let tokens: [Int]
    let servesAnyPrefix: Bool
    let bytes: Int64

    func usableTokens(for prompt: [Int]) -> Int {
        guard servesAnyPrefix else { return prompt.starts(with: tokens) ? tokens.count : 0 }
        return zip(tokens, prompt).prefix { $0 == $1 }.count
    }
}

enum PromptCacheDiskError: Error, CustomStringConvertible {
    case unusable(String)

    var description: String {
        switch self {
        case .unusable(let reason): reason
        }
    }
}

public struct PromptCacheDiskStore: Sendable {
    static let formatVersion = 1
    static let fileExtension = "safetensors"
    static let temporaryPrefix = ".tmp-"
    static let lockName = ".lock"
    static let modelDescriptionFiles = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
        "generation_config.json", "special_tokens_map.json",
    ]

    enum Key {
        static let format = "basalt_format"
        static let fingerprint = "basalt_fingerprint"
        static let model = "basalt_model"
        static let namespace = "basalt_namespace"
        static let tokens = "basalt_tokens"
        static let servesAnyPrefix = "basalt_serves_any_prefix"

        static func inHeader(_ key: String) -> String {
            "1." + key
        }
    }

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/prompt-cache")
    }

    public let directory: URL
    public let byteQuota: Int64
    public let retentionDays: Int
    private var headers: [String: (bytes: Int64, entry: PromptCacheDiskEntry?)] = [:]
    private let logger = GemLogger(module: "PromptCacheDisk")

    public init(directory: URL = Self.defaultDirectory, byteQuota: Int64, retentionDays: Int) {
        self.directory = directory
        self.byteQuota = byteQuota
        self.retentionDays = retentionDays
    }

    public static func fingerprint(modelDirectory: URL) -> String {
        var hasher = SHA256()
        hasher.update(data: Data("basalt-prompt-cache-v\(formatVersion) \(HealthResponse.version)\n".utf8))
        let files = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let name = file.lastPathComponent
            let resolved = file.resolvingSymlinksInPath()
            if file.pathExtension == fileExtension {
                let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1
                hasher.update(data: Data("weights \(name) \(size)\n".utf8))
            } else if modelDescriptionFiles.contains(name), let contents = try? Data(contentsOf: resolved) {
                hasher.update(data: Data("file \(name) \(hex(SHA256.hash(data: contents)))\n".utf8))
            }
        }
        return hex(hasher.finalize())
    }

    static func hex(_ digest: some Sequence<UInt8>) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    static func fileName(fingerprint: String, namespace: String, tokens: [Int]) -> String {
        let key = "\(fingerprint)\u{0}\(namespace)\u{0}\(tokens.map(String.init).joined(separator: ","))"
        return hex(SHA256.hash(data: Data(key.utf8))) + "." + fileExtension
    }

    func url(_ name: String) -> URL {
        directory.appendingPathComponent(name)
    }

    func prepareDirectory() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    }

    func withExclusiveLock<T>(_ body: () throws -> T) throws -> T {
        try prepareDirectory()
        let descriptor = open(url(Self.lockName).path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw PromptCacheDiskError.unusable("cannot open \(url(Self.lockName).path): errno \(errno)") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw PromptCacheDiskError.unusable("cannot lock \(directory.path): errno \(errno)") }
        defer { flock(descriptor, LOCK_UN) }
        return try body()
    }

    struct StoredFile {
        let name: String
        let bytes: Int64
        let modified: Date
    }

    func storedFiles() -> [StoredFile] {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
        return urls.compactMap { url in
            let name = url.lastPathComponent
            guard Self.isStoreFileName(name), let values = try? url.resourceValues(forKeys: Set(keys)), values.isRegularFile == true
            else { return nil }
            return StoredFile(name: name, bytes: Int64(values.fileSize ?? 0), modified: values.contentModificationDate ?? .distantPast)
        }
    }

    static func isStoreFileName(_ name: String) -> Bool {
        let stem = name.dropLast(fileExtension.count + 1)
        return name.hasSuffix("." + fileExtension) && stem.count == 64 && stem.allSatisfy(\.isHexDigit)
    }

    func temporaryFiles() -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix(Self.temporaryPrefix) }
    }

    mutating func entries() -> [PromptCacheDiskEntry] {
        let files = storedFiles()
        headers = headers.filter { cached in files.contains { $0.name == cached.key && $0.bytes == cached.value.bytes } }
        return files.compactMap { file in
            if let cached = headers[file.name] { return cached.entry }
            let entry = readEntry(file)
            headers[file.name] = (file.bytes, entry)
            return entry
        }
    }

    private func readEntry(_ file: StoredFile) -> PromptCacheDiskEntry? {
        let parsed = ModelInspector.readSafetensors(at: url(file.name), fileSize: file.bytes)
            .mapError { PromptCacheDiskError.unusable($0.message) }
            .flatMap { header -> Result<PromptCacheDiskEntry?, PromptCacheDiskError> in
                let metadata = header.metadata
                guard let format = metadata[Key.inHeader(Key.format)] else {
                    return .failure(.unusable("no basalt prompt-cache metadata"))
                }
                guard format == String(Self.formatVersion) else { return .success(nil) }
                guard let fingerprint = metadata[Key.inHeader(Key.fingerprint)],
                      let namespace = metadata[Key.inHeader(Key.namespace)],
                      let tokens = metadata[Key.inHeader(Key.tokens)].flatMap(Self.tokens(from:)), !tokens.isEmpty,
                      Self.fileName(fingerprint: fingerprint, namespace: namespace, tokens: tokens) == file.name
                else { return .failure(.unusable("incomplete basalt prompt-cache metadata")) }
                return .success(PromptCacheDiskEntry(
                    name: file.name, fingerprint: fingerprint, namespace: namespace, tokens: tokens,
                    servesAnyPrefix: metadata[Key.inHeader(Key.servesAnyPrefix)] == "true", bytes: file.bytes))
            }
        switch parsed {
        case .success(let entry):
            return entry
        case .failure(let problem):
            logger.warn("Removing unreadable prompt-cache file \(file.name): \(problem)")
            try? FileManager.default.removeItem(at: url(file.name))
            return nil
        }
    }

    static func tokens(from text: String) -> [Int]? {
        let parts = text.split(separator: ",")
        let tokens = parts.compactMap { Int($0) }
        return tokens.count == parts.count ? tokens : nil
    }

    mutating func longestPrefix(
        fingerprint: String, namespace: String, of prompt: [Int], longerThan stored: Int, maxBytes: Int
    ) -> (entry: PromptCacheDiskEntry, usable: Int)? {
        entries()
            .filter { $0.fingerprint == fingerprint && $0.namespace == namespace && $0.bytes <= Int64(maxBytes) }
            .map { (entry: $0, usable: $0.usableTokens(for: prompt)) }
            .filter { $0.usable > stored && Self.worthLoading($0.usable, of: $0.entry.tokens.count) }
            .max { ($0.usable, -$0.entry.bytes) < ($1.usable, -$1.entry.bytes) }
    }

    static func worthLoading(_ usable: Int, of stored: Int) -> Bool {
        usable * 2 >= stored
    }

    func load(_ entry: PromptCacheDiskEntry, layout: [String]) throws -> PromptCache {
        let snapshot = try loadPromptCacheSnapshot(url: url(entry.name))
        guard snapshot.metadata[Key.fingerprint] == entry.fingerprint,
              snapshot.metadata[Key.tokens].flatMap(Self.tokens(from:)) == entry.tokens
        else {
            throw PromptCacheDiskError.unusable("metadata changed after the header was read")
        }
        let found = snapshot.cache.map(Self.layoutName)
        guard found == layout else {
            throw PromptCacheDiskError.unusable("layer layout \(found) does not match the model's \(layout)")
        }
        for case let recurrent as ArraysCache in snapshot.cache {
            recurrent.offset = entry.tokens.count
        }
        let cache = PromptCache(snapshot.cache)
        cache.state = snapshot.state
        guard cache.tokenCount >= entry.tokens.count, cache.isSharable || cache.isSnapshot(of: entry.tokens.count) else {
            throw PromptCacheDiskError.unusable("holds \(cache.tokenCount) tokens in a layout the prompt cache cannot lease for \(entry.tokens.count)")
        }
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url(entry.name).path)
        return cache
    }

    static func layoutName(_ layer: KVCache) -> String {
        String(describing: type(of: layer))
    }

    mutating func discard(_ entry: PromptCacheDiskEntry) {
        try? FileManager.default.removeItem(at: url(entry.name))
        headers[entry.name] = nil
    }

    static func estimatedBytes(of cache: PromptCache, tokens: Int) -> Int64 {
        Int64(cache.byteCount + tokens * 8 + cache.layers.count * 512 + (64 << 10))
    }

    @discardableResult
    mutating func store(
        fingerprint: String, model: String, namespace: String, tokens: [Int], cache: PromptCache, keeping kept: Set<String> = []
    ) -> String? {
        let name = Self.fileName(fingerprint: fingerprint, namespace: namespace, tokens: tokens)
        let final = url(name)
        if FileManager.default.fileExists(atPath: final.path) {
            try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: final.path)
            return name
        }
        let estimate = Self.estimatedBytes(of: cache, tokens: tokens.count)
        guard estimate <= byteQuota else {
            logger.info("Not persisting \(tokens.count) prompt tokens: \(estimate >> 20) MB exceeds --prompt-cache-disk-mb \(byteQuota >> 20)")
            return nil
        }
        let temporary = url(Self.temporaryPrefix + UUID().uuidString + "." + Self.fileExtension)
        do {
            return try withExclusiveLock {
                guard makeRoom(for: estimate, keeping: kept.union([name])) else {
                    logger.info("Not persisting \(tokens.count) prompt tokens: the hotter files already fill --prompt-cache-disk-mb")
                    return nil
                }
                defer { try? FileManager.default.removeItem(at: temporary) }
                try savePromptCache(
                    url: temporary, cache: cache.layers,
                    metadata: [
                        Key.format: String(Self.formatVersion), Key.fingerprint: fingerprint, Key.model: model,
                        Key.namespace: namespace, Key.tokens: tokens.map(String.init).joined(separator: ","),
                        Key.servesAnyPrefix: String(cache.isSharable),
                    ],
                    state: cache.state)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
                try Self.flushToDisk(temporary)
                let written = Int64((try FileManager.default.attributesOfItem(atPath: temporary.path)[.size] as? Int) ?? 0)
                guard written <= estimate || makeRoom(for: written, keeping: kept.union([name])) else {
                    logger.info("Not persisting \(tokens.count) prompt tokens: \(written >> 20) MB on disk does not fit --prompt-cache-disk-mb")
                    return nil
                }
                guard rename(temporary.path, final.path) == 0 else {
                    throw PromptCacheDiskError.unusable("rename failed: errno \(errno)")
                }
                return name
            }
        } catch {
            logger.warn("Could not persist \(tokens.count) prompt tokens to \(directory.path): \(error)")
            return nil
        }
    }

    static func flushToDisk(_ file: URL) throws {
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.synchronize()
    }

    private func makeRoom(for bytes: Int64, keeping kept: Set<String>) -> Bool {
        var files = storedFiles().sorted { $0.modified < $1.modified }
        var total = files.reduce(Int64(0)) { $0 + $1.bytes }
        while total + bytes > byteQuota {
            guard let victim = files.firstIndex(where: { !kept.contains($0.name) }) else { return false }
            try? FileManager.default.removeItem(at: url(files[victim].name))
            total -= files[victim].bytes
            files.remove(at: victim)
        }
        return true
    }

    @discardableResult
    public func prune(now: Date = Date()) throws -> (files: Int, bytes: Int64) {
        try withExclusiveLock {
            var removed = 0
            var freed: Int64 = 0
            for temporary in temporaryFiles() {
                let size = (try? temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                if (try? FileManager.default.removeItem(at: temporary)) != nil {
                    removed += 1
                    freed += Int64(size)
                }
            }
            let cutoff = now.addingTimeInterval(-TimeInterval(retentionDays) * 86_400)
            var files = storedFiles().sorted { $0.modified < $1.modified }
            var total = files.reduce(Int64(0)) { $0 + $1.bytes }
            while let oldest = files.first, oldest.modified < cutoff || total > byteQuota {
                if (try? FileManager.default.removeItem(at: url(oldest.name))) != nil {
                    removed += 1
                    freed += oldest.bytes
                }
                total -= oldest.bytes
                files.removeFirst()
            }
            return (removed, freed)
        }
    }
}
