import Foundation
import Rainbow
import BasaltCore

// MARK: - Epic 16.4: Code Diff Viewer
// Unified diff parser and renderer with inline and side-by-side modes
// Supports clipboard integration (macOS pbcopy/pbpaste)

// MARK: - Data Structures

/// Represents a single line in a diff
public struct DiffLine: Sendable, Equatable {
    public enum LineType: Sendable, Equatable {
        case addition   // Lines starting with +
        case deletion   // Lines starting with -
        case context    // Lines starting with space (context)
    }
    
    public let type: LineType
    public let text: String
    public let lineNumber: Int?
    
    public init(type: LineType, text: String, lineNumber: Int? = nil) {
        self.type = type
        self.text = text
        self.lineNumber = lineNumber
    }
}

/// Represents a hunk (chunk) of changes in a diff
public struct DiffHunk: Sendable, Equatable {
    public let oldFile: String?
    public let newFile: String?
    public let header: String  // e.g., "@@ -12,8 +12,5 @@"
    public let lines: [DiffLine]
    
    public init(oldFile: String?, newFile: String?, header: String, lines: [DiffLine]) {
        self.oldFile = oldFile
        self.newFile = newFile
        self.header = header
        self.lines = lines
    }
}

// MARK: - Diff Format

public enum DiffFormat: Sendable, Equatable {
    case unified    // Git-style diff -u
    case context    // Old-style diff -c
    case auto       // Auto-detect
}

public struct DiffParser: Sendable {
    public init() {}

    public static func parseUnifiedDiff(_ diff: String) -> [DiffHunk] {
        guard !diff.isEmpty else { return [] }
        let rawLines = diff.components(separatedBy: .newlines)
        var parser = HunkState()
        var index = 0
        while index < rawLines.count {
            index += parser.consume(at: index, in: rawLines)
        }
        parser.flush()
        return parser.hunks
    }

    private struct HunkState {
        var hunks: [DiffHunk] = []
        var oldFile: String?
        var newFile: String?
        var header: String?
        var lines: [DiffLine] = []
        var remaining = (old: 0, new: 0)

        mutating func flush() {
            if let header, !lines.isEmpty {
                hunks.append(DiffHunk(oldFile: oldFile, newFile: newFile, header: header, lines: lines))
            }
        }

        mutating func consume(at index: Int, in rawLines: [String]) -> Int {
            let line = rawLines[index]
            if line.hasPrefix("diff --git ") {
                flush()
                (oldFile, newFile, header, lines, remaining) = (nil, nil, nil, [], (0, 0))
                return 1
            }
            if header == nil, line.hasPrefix("index ") { return 1 }
            if DiffParser.isFileHeader(at: index, in: rawLines, hasHunk: header != nil, remaining: remaining) {
                flush()
                (header, lines, remaining) = (nil, [], (0, 0))
                oldFile = String(line.dropFirst(4))
                guard index + 1 < rawLines.count, rawLines[index + 1].hasPrefix("+++ ") else { return 1 }
                newFile = String(rawLines[index + 1].dropFirst(4))
                return 2
            }
            if header == nil, line.hasPrefix("+++ ") {
                newFile = String(line.dropFirst(4))
                return 1
            }
            if line.hasPrefix("@@") {
                flush()
                header = line
                lines = []
                remaining = DiffParser.hunkCounts(line) ?? (0, 0)
                return 1
            }
            if header != nil, let parsed = DiffParser.hunkLine(line, remaining: &remaining) {
                lines.append(parsed)
            }
            return 1
        }
    }

    static func isFileHeader(at index: Int, in rawLines: [String], hasHunk: Bool, remaining: (old: Int, new: Int)) -> Bool {
        guard rawLines[index].hasPrefix("--- ") else { return false }
        if !hasHunk || (remaining.old <= 0 && remaining.new <= 0) { return true }
        return index + 2 < rawLines.count && rawLines[index + 1].hasPrefix("+++ ") && rawLines[index + 2].hasPrefix("@@")
    }

    static func hunkCounts(_ header: String) -> (old: Int, new: Int)? {
        let parts = header.split(separator: " ")
        guard parts.count >= 3, parts[0] == "@@", parts[1].hasPrefix("-"), parts[2].hasPrefix("+") else { return nil }
        func spanCount(_ token: Substring) -> Int? {
            let nums = token.dropFirst().split(separator: ",", maxSplits: 1)
            guard !nums.isEmpty, Int(nums[0]) != nil else { return nil }
            return nums.count == 2 ? Int(nums[1]) : 1
        }
        guard let old = spanCount(parts[1]), let new = spanCount(parts[2]) else { return nil }
        return (old, new)
    }

    static func hunkLine(_ line: String, remaining: inout (old: Int, new: Int)) -> DiffLine? {
        if line.isEmpty || line.hasPrefix("\\ ") { return nil }
        if line.hasPrefix("+"), remaining.new > 0 || !line.hasPrefix("+++") {
            remaining.new = max(0, remaining.new - 1)
            return DiffLine(type: .addition, text: String(line.dropFirst()))
        }
        if line.hasPrefix("-"), remaining.old > 0 || !line.hasPrefix("---") {
            remaining.old = max(0, remaining.old - 1)
            return DiffLine(type: .deletion, text: String(line.dropFirst()))
        }
        if line.hasPrefix(" ") {
            remaining.old = max(0, remaining.old - 1)
            remaining.new = max(0, remaining.new - 1)
            return DiffLine(type: .context, text: String(line.dropFirst()))
        }
        return DiffLine(type: .context, text: line)
    }

    static func diffLine(_ line: String) -> DiffLine? {
        if line.hasPrefix("\\ ") { return nil }
        switch line.first {
        case "+" where !line.hasPrefix("+++"): return DiffLine(type: .addition, text: String(line.dropFirst()))
        case "-" where !line.hasPrefix("---"): return DiffLine(type: .deletion, text: String(line.dropFirst()))
        case " ": return DiffLine(type: .context, text: String(line.dropFirst()))
        case nil: return nil
        default: return DiffLine(type: .context, text: line)
        }
    }

    public static func detectFormat(_ diff: String) -> DiffFormat {
        if diff.contains("---") && diff.contains("+++") && diff.contains("@@") {
            return .unified
        } else if diff.contains("***") && diff.contains("---") {
            return .context
        }
        return .unified
    }
}

// MARK: - Diff Renderer

public struct DiffRenderer: Sendable {
    public enum Mode {
        case inline       // Unified diff (default)
        case sideBySide   // Split view
    }
    
    /// Main rendering function
    public static func render(
        _ diff: String,
        mode: Mode = .inline,
        format: DiffFormat = .auto,
        colorize: Bool = true,
        syntaxHighlight: Bool = false
    ) -> String {
        let hunks = DiffParser.parseUnifiedDiff(diff)
        
        switch mode {
        case .inline:
            return renderInline(hunks: hunks, colorize: colorize)
        case .sideBySide:
            return renderSideBySide(hunks: hunks, colorize: colorize)
        }
    }
    
    /// Render inline mode (unified diff style)
    private static func renderInline(hunks: [DiffHunk], colorize: Bool) -> String {
        var output = ""
        
        for hunk in hunks {
            // Render file headers
            if let oldFile = hunk.oldFile {
                let header = "--- \(oldFile)"
                output += colorize ? TerminalUI.dim(header) + "\n" : header + "\n"
            }
            if let newFile = hunk.newFile {
                let header = "+++ \(newFile)"
                output += colorize ? TerminalUI.dim(header) + "\n" : header + "\n"
            }
            
            // Render hunk header
            let headerLine = hunk.header
            output += colorize ? TerminalUI.info(headerLine) + "\n" : headerLine + "\n"
            
            // Render lines
            for line in hunk.lines {
                let rendered: String
                switch line.type {
                case .addition:
                    let lineText = "+\(line.text)"
                    rendered = colorize ? TerminalUI.success(lineText) : lineText
                case .deletion:
                    let lineText = "-\(line.text)"
                    rendered = colorize ? TerminalUI.error(lineText) : lineText
                case .context:
                    let lineText = " \(line.text)"
                    rendered = colorize ? TerminalUI.dim(lineText) : lineText
                }
                output += rendered + "\n"
            }
        }
        
        return output
    }
    
    /// Render side-by-side mode (split view)
    private static func renderSideBySide(hunks: [DiffHunk], colorize: Bool) -> String {
        guard !hunks.isEmpty else { return "" }
        
        var output = ""
        
        for hunk in hunks {
            // Extract file names
            let oldFileName = hunk.oldFile?.components(separatedBy: "/").last ?? "Before"
            let newFileName = hunk.newFile?.components(separatedBy: "/").last ?? "After"
            
            // Build table
            var table = TableBuilder()
            table = table.addHeader(oldFileName)
            table = table.addHeader(newFileName)
            table = table.setStyle(.unicode)
            
            // Group lines by changes
            var oldLines: [String] = []
            var newLines: [String] = []
            
            for line in hunk.lines {
                switch line.type {
                case .deletion:
                    oldLines.append(line.text)
                case .addition:
                    newLines.append(line.text)
                case .context:
                    // For context, add to both sides
                    oldLines.append(line.text)
                    newLines.append(line.text)
                }
            }
            
            // Balance line counts
            let maxLines = max(oldLines.count, newLines.count)
            while oldLines.count < maxLines {
                oldLines.append("")
            }
            while newLines.count < maxLines {
                newLines.append("")
            }
            
            // Add rows
            for i in 0..<maxLines {
                table = table.addRow([oldLines[i], newLines[i]])
            }
            
            output += table.build() + "\n"
        }
        
        return output
    }
    
    /// Copy diff to clipboard (macOS only)
    public static func copyDiff(_ diff: String) async throws {
        #if os(macOS)
        let clipboard = ClipboardManager()
        try await clipboard.copy(diff)
        #else
        throw NSError(domain: "DiffRenderer", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Clipboard not supported on this platform"
        ])
        #endif
    }
}
