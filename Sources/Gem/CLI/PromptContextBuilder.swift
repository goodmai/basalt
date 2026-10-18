import Foundation

public struct PromptContextBuilder: Sendable {
    /// Epic 16.12: Extract file references from prompt without reading them
    public static func extractFileReferences(from prompt: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: "(?<!\\\\)@([\\w\\.\\-\\/]+)") else { return [] }
        let matches = regex.matches(in: prompt, range: NSRange(prompt.startIndex..., in: prompt))
        return matches.compactMap { match in
            if let range = Range(match.range(at: 1), in: prompt) {
                return String(prompt[range])
            }
            return nil
        }
    }

    /// Parses a prompt string for `@filepath` syntax, reads the files, and appends them to the prompt.
    public static func build(prompt: String) async throws -> String {
        var finalPrompt = prompt
        var appendedContext = ""
        
        let filesToInject = extractFileReferences(from: prompt)
        
        if !filesToInject.isEmpty {
            appendedContext += "\n\n--- Context Files ---\n"
            for filePath in filesToInject.reversed() {
                let url = URL(fileURLWithPath: filePath)
                
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw GemError.invalidRequestStructure(details: "File not found: \(filePath)")
                }
                
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                let size = attributes[.size] as? UInt64 ?? 0
                
                if size > 1024 * 1024 { // 1MB limit for safety
                    throw GemError.invalidRequestStructure(details: "File too large to inject (>1MB): \(filePath)")
                }
                
                let content = try String(contentsOf: url, encoding: .utf8)
                appendedContext += "\nFile: \(filePath)\n```\n\(content)\n```\n"
            }
            finalPrompt += appendedContext
        }
        
        finalPrompt = finalPrompt.replacingOccurrences(of: "\\@", with: "@")
        
        return finalPrompt
    }

    // MARK: — Tool-Prefix Compaction (Issue #26)
    // Groundwork for slot-based prefix reuse (#56); no request path calls these yet.

    /// Extracts the immutable tool definitions / system prefix from a full prompt.
    public static func extractToolPrefix(from prompt: String) -> (toolPrefix: String, conversationalTail: String) {
        let markers = [
            "--- Tool Definitions ---",
            "Available tools:",
            "<tools>",
            "[Tools]",
            "## Tools"
        ]

        for marker in markers {
            if let range = prompt.range(of: marker) {
                // Look for tool definition ending, e.g. closing </tools>, separator, or first [User] turn
                if let endMarker = prompt.range(of: "</tools>", range: range.upperBound..<prompt.endIndex) {
                    let prefix = String(prompt[..<endMarker.upperBound])
                    let tail = String(prompt[endMarker.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    return (prefix, tail)
                } else if let userTurn = prompt.range(of: "[User]:", range: range.upperBound..<prompt.endIndex) {
                    let prefix = String(prompt[..<userTurn.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let tail = String(prompt[userTurn.lowerBound...])
                    return (prefix, tail)
                }
            }
        }

        // Fallback: If prompt begins with [System]: take up to the first user/assistant turn
        if prompt.hasPrefix("[System]:") {
            if let firstUser = prompt.range(of: "\n\n[User]:") {
                let prefix = String(prompt[..<firstUser.lowerBound])
                let tail = String(prompt[firstUser.upperBound...])
                return (prefix, tail)
            }
        }

        return ("", prompt)
    }

    /// Compacts conversation history turns while strictly preserving the tool prefix at the head.
    ///
    /// The session's single tool prefix is preserved verbatim when the history is compacted.
    /// The newest turns (at the tail) are retained, while intermediate old turns are compacted.
    public static func compactWithToolPrefixProtection(
        toolPrefix: String,
        historyTurns: [String],
        maxBudgetChars: Int
    ) -> String {
        let cleanPrefix = toolPrefix.trimmingCharacters(in: .whitespacesAndNewlines)
        if historyTurns.isEmpty { return cleanPrefix }

        let prefixCost = cleanPrefix.isEmpty ? 0 : cleanPrefix.count + 2
        let remainingBudget = max(100, maxBudgetChars - prefixCost)

        // Try including all turns
        let fullHistory = historyTurns.joined(separator: "\n\n")
        if fullHistory.count <= remainingBudget {
            return cleanPrefix.isEmpty ? fullHistory : "\(cleanPrefix)\n\n\(fullHistory)"
        }

        // If history exceeds budget: protect the toolPrefix, keep the newest turn(s),
        // and drop/compact the oldest intermediate turns
        var keptTurns: [String] = []
        var currentLen = 0
        let compactionNotice = "... [earlier turns compacted to preserve tool prefix] ..."
        let noticeCost = compactionNotice.count + 4

        // Work backwards from the newest turn
        for turn in historyTurns.reversed() {
            let turnCost = turn.count + 2
            if currentLen + turnCost + noticeCost <= remainingBudget {
                keptTurns.insert(turn, at: 0)
                currentLen += turnCost
            } else {
                break
            }
        }

        // Ensure at least the very latest turn is kept if possible
        if keptTurns.isEmpty, let last = historyTurns.last {
            keptTurns = [String(last.suffix(remainingBudget))]
        }

        let compactedHistory = "\(compactionNotice)\n\n\(keptTurns.joined(separator: "\n\n"))"
        return cleanPrefix.isEmpty ? compactedHistory : "\(cleanPrefix)\n\n\(compactedHistory)"
    }
}

