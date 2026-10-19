import Foundation

public struct Frontmatter: Sendable, Equatable {
    public let fields: [String: String]
    public let body: String

    public init(fields: [String: String], body: String) {
        self.fields = fields
        self.body = body
    }

    public static func parse(_ content: String) -> Frontmatter? {
        let lines = content.components(separatedBy: .newlines)
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---",
              let closing = lines.indices.dropFirst().first(where: { lines[$0].trimmingCharacters(in: .whitespaces) == "---" })
        else { return nil }
        let fields = parseFields(Array(lines[1..<closing]))
        let body = lines[(closing + 1)...].joined(separator: "\n")
        return Frontmatter(fields: fields, body: body)
    }

    static func parseFields(_ lines: [String]) -> [String: String] {
        var fields: [String: String] = [:]
        var index = 0
        while index < lines.count {
            let line = lines[index]
            index += 1
            guard let first = line.first, !isIndent(first), first != "#",
                  let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces)
            let raw = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            var block: [String] = []
            while index < lines.count, lines[index].first.map(isIndent) ?? true {
                block.append(lines[index])
                index += 1
            }
            guard !key.isEmpty, let value = scalar(raw, block: block) else { continue }
            fields[key] = value
        }
        return fields
    }

    private static func isIndent(_ character: Character) -> Bool {
        character == " " || character == "\t"
    }

    private static func scalar(_ raw: String, block: [String]) -> String? {
        if let style = raw.first, style == ">" || style == "|",
           raw.dropFirst().allSatisfy({ "+-0123456789".contains($0) }) {
            let lines = dedented(block)
            return (style == "|" ? lines.joined(separator: "\n") : folded(lines))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if raw.isEmpty { return nil }
        let continuation = block.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        return unquoted(([raw] + continuation).joined(separator: " "))
    }

    private static func dedented(_ block: [String]) -> [String] {
        var lines = block
        while lines.last?.trimmingCharacters(in: .whitespaces).isEmpty == true { lines.removeLast() }
        let indent = lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.prefix(while: isIndent).count }.min() ?? 0
        return lines.map { String($0.dropFirst(min(indent, $0.prefix(while: isIndent).count))) }
    }

    private static func folded(_ lines: [String]) -> String {
        var result = ""
        var pendingBreak = false
        for line in lines {
            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                result += "\n"
                pendingBreak = false
                continue
            }
            if pendingBreak { result += " " }
            result += line
            pendingBreak = true
        }
        return result
    }

    private static func unquoted(_ value: String) -> String {
        guard value.count >= 2, let quote = value.first, quote == "\"" || quote == "'", value.last == quote else {
            return value
        }
        let inner = String(value.dropFirst().dropLast())
        if quote == "'" { return inner.replacingOccurrences(of: "''", with: "'") }
        if let data = value.data(using: .utf8), let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }
        return inner
    }
}
