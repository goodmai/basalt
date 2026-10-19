import Foundation
import BasaltCore

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
                    throw BasaltError.invalidRequestStructure(details: "File not found: \(filePath)")
                }
                
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                let size = attributes[.size] as? UInt64 ?? 0
                
                if size > 1024 * 1024 { // 1MB limit for safety
                    throw BasaltError.invalidRequestStructure(details: "File too large to inject (>1MB): \(filePath)")
                }
                
                let content = try String(contentsOf: url, encoding: .utf8)
                appendedContext += "\nFile: \(filePath)\n```\n\(content)\n```\n"
            }
            finalPrompt += appendedContext
        }
        
        finalPrompt = finalPrompt.replacingOccurrences(of: "\\@", with: "@")
        
        return finalPrompt
    }
}

