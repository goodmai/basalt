import Foundation

public enum SessionLogReader {
    public struct Session: Sendable, Equatable {
        public let id: String
        public let file: URL
        public let modified: Date
    }

    public struct Content: Decodable, Sendable {
        public let messages: [ChatMessageInput]?
        public let prompt: String?
        public let parent: String?
        public let prefixCount: Int?

        enum CodingKeys: String, CodingKey { case messages, prompt, parent, prefixCount = "prefix_count" }
    }

    public struct Output: Decodable, Sendable {
        public let text: String?
        public let chars: Int?
    }

    public struct Tokens: Decodable, Sendable {
        public let prompt: Int
        public let completion: Int
        public let cached: Int
    }

    public struct Timing: Decodable, Sendable {
        public let ttftMs: Int
        public let generationMs: Int
        public let tokensPerSecond: Double

        enum CodingKeys: String, CodingKey { case ttftMs = "ttft_ms", generationMs = "generation_ms", tokensPerSecond = "tokens_per_second" }
    }

    public struct Line: Decodable, Sendable {
        public let type: String
        public let ts: String?
        public let id: String?
        public let requestId: String?
        public let source: String?
        public let status: String?
        public let mode: String?
        public let content: Content?
        public let output: Output?
        public let tokens: Tokens?
        public let timing: Timing?

        enum CodingKeys: String, CodingKey {
            case type, ts, id, requestId = "request_id", source, status, mode, content, output, tokens, timing
        }
    }

    public static func sessions(in directory: URL) -> [Session] {
        let fm = FileManager.default
        let files = (fm.enumerator(at: directory, includingPropertiesForKeys: [.contentModificationDateKey])?
            .compactMap { $0 as? URL } ?? []).filter { $0.pathExtension == "jsonl" }
        return files.map { file in
            Session(id: file.deletingPathExtension().lastPathComponent, file: file,
                    modified: (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast)
        }.sorted { $0.modified > $1.modified }
    }

    public static func file(for id: String, in directory: URL) -> URL? {
        sessions(in: directory).first { $0.id == id }?.file
    }

    public static func lines(of file: URL) -> [Line] {
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        return text.split(separator: "\n").compactMap { try? decoder.decode(Line.self, from: Data($0.utf8)) }
    }

    public static func mode(of lines: [Line]) -> String? {
        lines.first { $0.type == "session_start" }?.mode
    }

    public static func conversation(of requestId: String, in lines: [Line]) -> [ChatMessageInput]? {
        let requests = Dictionary(lines.filter { $0.type == "request" }.compactMap { line in line.id.map { ($0, line) } },
                                  uniquingKeysWith: { first, _ in first })
        var visited: Set<String> = []
        func rebuild(_ id: String) -> [ChatMessageInput]? {
            guard visited.insert(id).inserted, let content = requests[id]?.content else { return nil }
            let own = content.messages ?? content.prompt.map { [ChatMessageInput(role: .user, text: $0)] } ?? []
            guard let parent = content.parent, let count = content.prefixCount else { return own }
            guard let prefix = rebuild(parent), prefix.count >= count else { return nil }
            return Array(prefix.prefix(count)) + own
        }
        return rebuild(requestId)
    }

    public static func recentConversations(in lines: [Line], limit: Int) -> [(id: String, messages: [ChatMessageInput])] {
        lines.filter { $0.type == "request" && $0.content?.messages != nil }.suffix(limit).compactMap { line in
            guard let id = line.id, let messages = conversation(of: id, in: lines) else { return nil }
            return (id, messages)
        }
    }

    public static func lastExchange(in lines: [Line]) -> (messages: [ChatMessageInput], reply: String?)? {
        guard let request = lines.last(where: { $0.type == "request" && $0.content != nil }), let id = request.id,
              let messages = conversation(of: id, in: lines)
        else { return nil }
        let reply = lines.last { $0.type == "response" && $0.requestId == id }?.output?.text
        return (messages, reply)
    }
}
