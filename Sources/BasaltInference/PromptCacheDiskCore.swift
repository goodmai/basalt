import Foundation
import CryptoKit
import BasaltCore

extension PromptCacheDiskStore {
    static func hex(_ digest: some Sequence<UInt8>) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    static func fileName(fingerprint: String, namespace: String, tokens: [Int]) -> String {
        let key = "\(fingerprint)\u{0}\(namespace)\u{0}\(PromptCacheDiskManifest.text(of: tokens))"
        return hex(SHA256.hash(data: Data(key.utf8))) + "." + fileExtension
    }

    static func isStoreFileName(_ name: String) -> Bool {
        let stem = name.dropLast(fileExtension.count + 1)
        return name.hasSuffix("." + fileExtension) && stem.count == 64 && stem.allSatisfy(\.isHexDigit)
    }

    static func worthLoading(_ usable: Int, of stored: Int) -> Bool {
        usable * 2 >= stored
    }

    static func isTransient(_ error: any Error) -> Bool {
        let domain = (error as NSError).domain
        return error is POSIXError || domain == NSPOSIXErrorDomain || domain == NSCocoaErrorDomain
    }
}

struct PromptCacheFingerprint {
    enum Input: Equatable {
        case weights
        case description
        case ignored

        init(name: String, pathExtension: String) {
            if pathExtension == PromptCacheDiskStore.fileExtension {
                self = .weights
            } else if PromptCacheDiskStore.modelDescriptionFiles.contains(name) {
                self = .description
            } else {
                self = .ignored
            }
        }
    }

    private var hasher = SHA256()

    init(version: String) {
        add("basalt-prompt-cache-v\(PromptCacheDiskStore.formatVersion) \(version)")
    }

    mutating func addWeights(_ name: String, identity: String) {
        add("weights \(name) \(identity)")
    }

    mutating func addDescription(_ name: String, contents: Data) {
        add("file \(name) \(PromptCacheDiskStore.hex(SHA256.hash(data: contents)))")
    }

    func finalize() -> String {
        PromptCacheDiskStore.hex(hasher.finalize())
    }

    private mutating func add(_ line: String) {
        hasher.update(data: Data("\(line)\n".utf8))
    }
}

enum PromptCacheDiskManifest {
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

    static func text(of tokens: [Int]) -> String {
        tokens.map(String.init).joined(separator: ",")
    }

    static func tokens(from text: String) -> [Int]? {
        let parts = text.split(separator: ",")
        let tokens = parts.compactMap { Int($0) }
        return tokens.count == parts.count ? tokens : nil
    }

    static func encode(fingerprint: String, model: String, namespace: String, tokens: [Int], servesAnyPrefix: Bool) -> [String: String] {
        [
            Key.format: String(PromptCacheDiskStore.formatVersion), Key.fingerprint: fingerprint, Key.model: model,
            Key.namespace: namespace, Key.tokens: text(of: tokens), Key.servesAnyPrefix: String(servesAnyPrefix),
        ]
    }

    static func decode(header metadata: [String: String], name: String, bytes: Int64) -> Result<PromptCacheDiskEntry?, PromptCacheDiskError> {
        guard let format = metadata[Key.inHeader(Key.format)] else {
            return .failure(.unusable("no basalt prompt-cache metadata"))
        }
        guard format == String(PromptCacheDiskStore.formatVersion) else { return .success(nil) }
        guard let fingerprint = metadata[Key.inHeader(Key.fingerprint)],
              let namespace = metadata[Key.inHeader(Key.namespace)],
              let tokens = metadata[Key.inHeader(Key.tokens)].flatMap(tokens(from:)), !tokens.isEmpty,
              PromptCacheDiskStore.fileName(fingerprint: fingerprint, namespace: namespace, tokens: tokens) == name
        else { return .failure(.unusable("incomplete basalt prompt-cache metadata")) }
        return .success(PromptCacheDiskEntry(
            name: name, fingerprint: fingerprint, namespace: namespace, tokens: tokens,
            servesAnyPrefix: metadata[Key.inHeader(Key.servesAnyPrefix)] == "true", bytes: bytes))
    }

    static func matches(snapshot metadata: [String: String], _ entry: PromptCacheDiskEntry) -> Bool {
        metadata[Key.fingerprint] == entry.fingerprint && metadata[Key.tokens].flatMap(tokens(from:)) == entry.tokens
    }
}

enum PromptCacheDiskSelection {
    struct StoredFile: Equatable {
        let name: String
        let bytes: Int64
        let modified: Date
    }

    static let supersededTailTokens = 64

    static func longestPrefix(
        in entries: [PromptCacheDiskEntry], fingerprint: String, namespace: String, of prompt: [Int], longerThan stored: Int, maxBytes: ByteCount
    ) -> (entry: PromptCacheDiskEntry, usable: Int)? {
        entries
            .filter { $0.fingerprint == fingerprint && $0.namespace == namespace && $0.bytes <= maxBytes.rawValue }
            .map { (entry: $0, usable: $0.usableTokens(for: prompt)) }
            .filter { $0.usable > stored && PromptCacheDiskStore.worthLoading($0.usable, of: $0.entry.tokens.count) }
            .max { ($0.usable, -$0.entry.bytes) < ($1.usable, -$1.entry.bytes) }
    }

    static func supersedes(_ tokens: [Int], _ older: PromptCacheDiskEntry) -> Bool {
        older.servesAnyPrefix && older.tokens.count < tokens.count
            && zip(older.tokens, tokens).prefix { $0 == $1 }.count
                >= max(older.tokens.count - supersededTailTokens, (older.tokens.count + 1) / 2)
    }

    static func superseded(
        by tokens: [Int], in entries: [PromptCacheDiskEntry], fingerprint: String, namespace: String, keeping kept: Set<String>
    ) -> [PromptCacheDiskEntry] {
        entries.filter { older in
            older.fingerprint == fingerprint && older.namespace == namespace && !kept.contains(older.name) && supersedes(tokens, older)
        }
    }

    static func room(for bytes: Int64, in files: [StoredFile], quota: Int64, keeping kept: Set<String>) -> (evict: [StoredFile], fits: Bool) {
        var remaining = files.sorted { $0.modified < $1.modified }
        var total = remaining.reduce(Int64(0)) { $0 + $1.bytes }
        var evict: [StoredFile] = []
        while total + bytes > quota {
            guard let victim = remaining.firstIndex(where: { !kept.contains($0.name) }) else { return (evict, false) }
            evict.append(remaining[victim])
            total -= remaining[victim].bytes
            remaining.remove(at: victim)
        }
        return (evict, true)
    }

    static func expired(_ files: [StoredFile], now: Date, retentionDays: Int, quota: Int64) -> [StoredFile] {
        let cutoff = now.addingTimeInterval(-TimeInterval(retentionDays) * 86_400)
        var total = files.reduce(Int64(0)) { $0 + $1.bytes }
        var expired: [StoredFile] = []
        for file in files.sorted(by: { $0.modified < $1.modified }) {
            guard file.modified < cutoff || total > quota else { break }
            expired.append(file)
            total -= file.bytes
        }
        return expired
    }
}
