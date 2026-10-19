import Foundation
import BasaltCore

public struct HarnessKnowledgeEntry: Codable, Sendable, Equatable {
    public let slug: String
    public let topic: String
    public let domain: String
    public let content: String
    public let cue: String
    public let confidence: String
    public let attributions: [HarnessComponentAttribution]
    public let delta: [String: String]?
    public let createdAt: String
}

public struct HarnessLibraryLimits: Sendable, Equatable {
    public var maxTopicChars = 200
    public var maxCueChars = 500
    public var maxDomainChars = 64
    public var maxContentBytes = 32 * 1024
    public var maxDeltaBytes = 64 * 1024
    public var maxEntries = 1000
    public var maxLibraryBytes = 8 * 1024 * 1024

    public init() {}
}

public enum HarnessLibraryError: Error, Equatable, CustomStringConvertible {
    case invalidArgument(String)
    case full(String)
    case io(String)

    public var description: String {
        switch self {
        case .invalidArgument(let message), .full(let message), .io(let message): message
        }
    }
}

public struct HarnessExperienceLibrary: Sendable {
    public static let confidences: Set<String> = ["low", "medium", "high"]
    public static var defaultRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".basalt/harness-library", isDirectory: true)
    }

    public let rootDirectory: URL
    public let limits: HarnessLibraryLimits
    private let logger = GemLogger(module: "HarnessLibrary")

    public init(rootDirectory: URL, limits: HarnessLibraryLimits = HarnessLibraryLimits()) {
        self.rootDirectory = rootDirectory
        self.limits = limits
    }

    var entriesURL: URL { rootDirectory.appendingPathComponent("entries.json") }

    @discardableResult
    public func addKnowledge(
        topic: String,
        content: String,
        domain: String = "general",
        cue: String = "",
        confidence: String = "medium",
        attributions: [HarnessComponentAttribution] = [],
        delta: [String: String]? = nil
    ) throws -> HarnessKnowledgeEntry {
        let topic = Self.singleLine(topic)
        let cue = Self.singleLine(cue.isEmpty ? topic : cue)
        let domainSlug = Self.slugify(domain.isEmpty ? "general" : domain)
        try check(!topic.isEmpty && topic.count <= limits.maxTopicChars, "topic must be 1...\(limits.maxTopicChars) characters")
        try check(cue.count <= limits.maxCueChars, "cue must be at most \(limits.maxCueChars) characters")
        try check(domain.count <= limits.maxDomainChars, "domain must be at most \(limits.maxDomainChars) characters")
        try check(!content.isEmpty && content.utf8.count <= limits.maxContentBytes, "content must be 1...\(limits.maxContentBytes) bytes")
        try check(Self.confidences.contains(confidence), "confidence must be one of low, medium, high")
        if let token = HarnessLeakGuard.forbiddenToken(in: topic + "\n" + cue) {
            throw HarnessLibraryError.invalidArgument("topic and cue reach the harness proposer and must not contain '\(token)'")
        }
        let deltaBytes = delta.map { $0.reduce(0) { $0 + $1.key.utf8.count + $1.value.utf8.count } } ?? 0
        try check(deltaBytes <= limits.maxDeltaBytes, "harness delta must be at most \(limits.maxDeltaBytes) bytes")

        return try withLock {
            var entries = try loadEntries()
            guard entries.count < limits.maxEntries else {
                throw HarnessLibraryError.full("library holds the maximum of \(limits.maxEntries) entries")
            }
            let domainDir = rootDirectory.appendingPathComponent(domainSlug, isDirectory: true)
            let slug = uniqueSlug(Self.slugify(topic), domain: domainSlug, entries: entries, domainDir: domainDir)
            let entry = HarnessKnowledgeEntry(
                slug: slug, topic: topic, domain: domainSlug, content: content, cue: cue,
                confidence: confidence, attributions: attributions, delta: delta,
                createdAt: ISO8601DateFormatter().string(from: Date())
            )
            entries.append(entry)
            let encoded = try Self.encoder.encode(entries)
            guard encoded.count <= limits.maxLibraryBytes else {
                throw HarnessLibraryError.full("library would exceed \(limits.maxLibraryBytes) bytes")
            }

            let taskDir = domainDir.appendingPathComponent("harnesses/\(slug)", isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: taskDir, withIntermediateDirectories: true)
                try Data(Self.renderExperience(entry).utf8).write(to: taskDir.appendingPathComponent("EXPERIENCE.md"), options: .atomic)
                try encoded.write(to: entriesURL, options: .atomic)
            } catch {
                try? FileManager.default.removeItem(at: taskDir)
                throw HarnessLibraryError.io("cannot write the library: \(error.localizedDescription)")
            }
            let index = Self.renderIndex(domain: domainSlug, entries: entries.filter { $0.domain == domainSlug })
            try? Data(index.utf8).write(to: domainDir.appendingPathComponent("INDEX.md"), options: .atomic)
            return entry
        }
    }

    public func query(_ text: String? = nil, domain: String? = nil) throws -> [HarnessKnowledgeEntry] {
        let needle = text?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        let domainSlug = domain.flatMap { $0.isEmpty ? nil : Self.slugify($0) }
        return try withLock { try loadEntries() }.filter { entry in
            if let domainSlug, entry.domain != domainSlug { return false }
            return needle.isEmpty || [entry.topic, entry.cue, entry.content].contains { $0.lowercased().contains(needle) }
        }
    }

    public func index(domain: String) throws -> String {
        let domainSlug = Self.slugify(domain)
        let indexable = try query(domain: domainSlug).filter { HarnessLeakGuard.forbiddenToken(in: $0.topic + "\n" + $0.cue) == nil }
        return Self.renderIndex(domain: domainSlug, entries: indexable)
    }

    private func check(_ condition: Bool, _ message: String) throws {
        if !condition { throw HarnessLibraryError.invalidArgument(message) }
    }

    private func withLock<T>(_ body: () throws -> T) throws -> T {
        do {
            try FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        } catch {
            throw HarnessLibraryError.io("cannot create \(rootDirectory.path): \(error.localizedDescription)")
        }
        let descriptor = open(rootDirectory.appendingPathComponent(".lock").path, O_CREAT | O_RDWR, 0o600)
        guard descriptor >= 0 else { throw HarnessLibraryError.io("cannot open the library lock: errno \(errno)") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw HarnessLibraryError.io("cannot lock the library: errno \(errno)") }
        defer { flock(descriptor, LOCK_UN) }
        return try body()
    }

    private func loadEntries() throws -> [HarnessKnowledgeEntry] {
        guard FileManager.default.fileExists(atPath: entriesURL.path) else { return [] }
        let data: Data
        do {
            data = try Data(contentsOf: entriesURL)
        } catch {
            throw HarnessLibraryError.io("cannot read \(entriesURL.path): \(error.localizedDescription)")
        }
        if let entries = try? JSONDecoder().decode([HarnessKnowledgeEntry].self, from: data) {
            return entries
        }
        let quarantine = rootDirectory.appendingPathComponent("entries.json.corrupt-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString.prefix(8))")
        do {
            try FileManager.default.moveItem(at: entriesURL, to: quarantine)
        } catch {
            throw HarnessLibraryError.io("\(entriesURL.path) is corrupt and cannot be moved aside: \(error.localizedDescription)")
        }
        logger.warn("\(entriesURL.path) is not a valid entry list; moved to \(quarantine.lastPathComponent), starting an empty library")
        return []
    }

    private func uniqueSlug(_ base: String, domain: String, entries: [HarnessKnowledgeEntry], domainDir: URL) -> String {
        let taken = Set(entries.filter { $0.domain == domain }.map(\.slug))
        var candidate = base
        var counter = 2
        while taken.contains(candidate)
            || FileManager.default.fileExists(atPath: domainDir.appendingPathComponent("harnesses/\(candidate)").path) {
            candidate = "\(base)-\(counter)"
            counter += 1
        }
        return candidate
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    static func singleLine(_ text: String) -> String {
        let scalars = text.unicodeScalars.map { CharacterSet.controlCharacters.union(.newlines).contains($0) ? " " : Character($0) }
        return String(scalars).split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
    }

    public static func slugify(_ input: String) -> String {
        var slug = ""
        for scalar in input.lowercased().unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar), scalar.isASCII {
                slug.unicodeScalars.append(scalar)
            } else if !slug.isEmpty, !slug.hasSuffix("-") {
                slug += "-"
            }
        }
        let trimmed = String(slug.prefix(64)).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "entry" : trimmed
    }

    static func yamlString(_ text: String) -> String {
        let data = (try? JSONEncoder().encode(singleLine(text))) ?? Data("\"\"".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    static func fenced(_ text: String) -> String {
        let longestRun = text.split(whereSeparator: { $0 != "`" }).map(\.count).max() ?? 0
        let fence = String(repeating: "`", count: max(3, longestRun + 1))
        return "\(fence)text\n\(text)\n\(fence)"
    }

    static func renderExperience(_ entry: HarnessKnowledgeEntry) -> String {
        var lines = [
            "---",
            "topic: \(yamlString(entry.topic))",
            "domain: \(yamlString(entry.domain))",
            "cue: \(yamlString(entry.cue))",
            "confidence: \(yamlString(entry.confidence))",
            "created_at: \(yamlString(entry.createdAt))",
            "---",
            "",
            "# \(entry.topic)",
            "",
            fenced(entry.content)
        ]
        if !entry.attributions.isEmpty {
            lines += ["", "## Component attribution", "", "| candidate | component | paths | verdict |", "|---|---|---|---|"]
            lines += entry.attributions.map { attribution in
                let paths = attribution.paths.map { $0.replacingOccurrences(of: "|", with: "\\|") }.joined(separator: ", ")
                return "| \(attribution.candidateId) | \(attribution.component.rawValue) | \(paths) | \(attribution.verdict.rawValue) |"
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func renderIndex(domain: String, entries: [HarnessKnowledgeEntry]) -> String {
        var lines = ["# Experience Library index (\(domain))", ""]
        lines += entries.map { entry in
            "- [\(entry.confidence)] \(entry.topic) | cue: \(entry.cue) | harnesses/\(entry.slug)/EXPERIENCE.md"
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
