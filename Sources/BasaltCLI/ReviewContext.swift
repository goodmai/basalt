import Foundation
import BasaltUI

enum ReviewContext {
    struct Symbol: Sendable, Equatable {
        let file: String
        let signature: String
    }

    static let instructions = """
        Review this change as a senior engineer. List concrete bugs, security problems and missing tests, each with the file \
        and symbol it concerns, most severe first. Say "No issues found." when there are none. Do not restate the diff.
        """

    static let declarations: [String: NSRegularExpression] = {
        let swift = #"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|internal|fileprivate|open|package|static|final|override|mutating|nonmutating|nonisolated|convenience|required|indirect|distributed|class)\s+)*(?:func|init|deinit|subscript|struct|class|enum|actor|protocol|extension)\b"#
        let rust = #"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|const|unsafe|default)\s+)*(?:fn|struct|enum|trait|impl|type|mod)\b"#
        let python = #"^\s*(?:async\s+)?(?:def|class)\s+\w+"#
        let script = #"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:function\*?|class|interface|type|enum|namespace)\s+\w+"#
        let table = ["swift": swift, "rs": rust, "py": python, "ts": script, "tsx": script, "js": script, "jsx": script, "mjs": script]
        return table.compactMapValues { try? NSRegularExpression(pattern: $0) }
    }()

    static func path(of hunk: DiffHunk) -> String {
        let raw = hunk.newFile.flatMap { $0.hasPrefix("/dev/null") ? nil : $0 } ?? hunk.oldFile ?? ""
        let name = raw.split(separator: "\t", maxSplits: 1).first.map(String.init) ?? raw
        return name.hasPrefix("a/") || name.hasPrefix("b/") ? String(name.dropFirst(2)) : name
    }

    static func symbols(in hunks: [DiffHunk]) -> [Symbol] {
        var seen = Set<String>()
        var found: [Symbol] = []
        for hunk in hunks {
            let file = path(of: hunk)
            guard let pattern = declarations[(file as NSString).pathExtension.lowercased()] else { continue }
            let enclosing = hunk.header.split(separator: "@@", omittingEmptySubsequences: false).dropFirst(2).joined(separator: "@@")
            for line in [enclosing] + hunk.lines.map(\.text) where pattern.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil {
                let signature = trimmed(line)
                if seen.insert(file + "\u{0}" + signature).inserted { found.append(Symbol(file: file, signature: signature)) }
            }
        }
        return found
    }

    static func trimmed(_ line: String) -> String {
        var text = line.trimmingCharacters(in: .whitespaces)
        while let last = text.last, "{:".contains(last) { text.removeLast() }
        return text.trimmingCharacters(in: .whitespaces)
    }

    static func hunkText(_ hunk: DiffHunk) -> String {
        let body = hunk.lines.map { line in
            switch line.type {
            case .addition: "+" + line.text
            case .deletion: "-" + line.text
            case .context: " " + line.text
            }
        }
        return ([hunk.header] + body).joined(separator: "\n") + "\n"
    }

    static func prompt(diff: String, budgetChars: Int) -> String {
        let hunks = DiffParser.parseUnifiedDiff(diff)
        let symbols = symbols(in: hunks)
        var out = instructions + "\n\n## Touched symbols\n"
        out += symbols.isEmpty ? "(none recognised)\n" : symbols.map { "- \($0.file): `\($0.signature)`\n" }.joined()
        out += "\n## Diff\n"
        var included = 0
        var used = out.count
        var lastFile: String?
        for hunk in hunks {
            let file = path(of: hunk)
            let piece = (file == lastFile ? "" : "\n### \(file)\n") + hunkText(hunk)
            used += piece.count
            guard used <= budgetChars else { break }
            out += piece
            lastFile = file
            included += 1
        }
        if included < hunks.count { out += "\n[diff truncated: \(included) of \(hunks.count) hunks fit the context budget]\n" }
        return String(out.prefix(budgetChars))
    }
}
