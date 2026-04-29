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
        
        let regex = try NSRegularExpression(pattern: "(?<!\\\\)@([\\w\\.\\-\\/]+)")
        let matches = regex.matches(in: prompt, range: NSRange(prompt.startIndex..., in: prompt))
        
        var filesToInject: [String] = []
        
        for match in matches.reversed() { 
            if let range = Range(match.range(at: 1), in: prompt) {
                let filePath = String(prompt[range])
                filesToInject.append(filePath)
            }
        }
        
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
}
