import Foundation
import Rainbow

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

// MARK: - Diff Parser

public struct DiffParser: Sendable {
    public init() {}
    
    /// Parse unified diff format (diff -u)
    public static func parseUnifiedDiff(_ diff: String) -> [DiffHunk] {
        guard !diff.isEmpty else { return [] }
        
        let lines = diff.components(separatedBy: .newlines)
        var hunks: [DiffHunk] = []
        var currentOldFile: String?
        var currentNewFile: String?
        var currentHeader: String?
        var currentLines: [DiffLine] = []
        
        for line in lines {
            // File headers
            if line.hasPrefix("--- ") {
                currentOldFile = String(line.dropFirst(4))
                continue
            } else if line.hasPrefix("+++ ") {
                currentNewFile = String(line.dropFirst(4))
                continue
            } else if line.hasPrefix("diff --git ") {
                // Git-style header, skip
                continue
            } else if line.hasPrefix("index ") {
                // Git index line, skip
                continue
            }
            
            // Hunk header
            if line.hasPrefix("@@") {
                // Save previous hunk if exists
                if let header = currentHeader, !currentLines.isEmpty {
                    hunks.append(DiffHunk(
                        oldFile: currentOldFile,
                        newFile: currentNewFile,
                        header: header,
                        lines: currentLines
                    ))
                }
                
                // Start new hunk
                currentHeader = line
                currentLines = []
                continue
            }
            
            // Diff lines (only process after we have a header)
            guard currentHeader != nil else { continue }
            
            if line.hasPrefix("+") && !line.hasPrefix("+++") {
                let text = String(line.dropFirst())
                currentLines.append(DiffLine(type: .addition, text: text))
            } else if line.hasPrefix("-") && !line.hasPrefix("---") {
                let text = String(line.dropFirst())
                currentLines.append(DiffLine(type: .deletion, text: text))
            } else if line.hasPrefix(" ") {
                let text = String(line.dropFirst())
                currentLines.append(DiffLine(type: .context, text: text))
            } else if !line.isEmpty {
                // Treat non-empty lines without prefix as context
                currentLines.append(DiffLine(type: .context, text: line))
            }
        }
        
        // Add final hunk
        if let header = currentHeader, !currentLines.isEmpty {
            hunks.append(DiffHunk(
                oldFile: currentOldFile,
                newFile: currentNewFile,
                header: header,
                lines: currentLines
            ))
        }
        
        return hunks
    }
    
    /// Auto-detect diff format
    public static func detectFormat(_ diff: String) -> DiffFormat {
        if diff.contains("---") && diff.contains("+++") && diff.contains("@@") {
            return .unified
        } else if diff.contains("***") && diff.contains("---") {
            return .context
        }
        return .unified // Default to unified
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
