import Foundation

public struct PromptContextBuilder: Sendable {
    
    /// Parses a prompt string for `@filepath` syntax, reads the files, and appends them to the prompt.
    public static func build(prompt: String) async throws -> String {
        var finalPrompt = prompt
        var appendedContext = ""
        
        let regex = try NSRegularExpression(pattern: "(?<!\\\\)@([\\w\\.\\-\\/]+)")
        let matches = regex.matches(in: prompt, range: NSRange(prompt.startIndex..., in: prompt))
        
        var filesToInject: [String] = []
        
        for match in matches.reversed() { // Reverse to not mess up indices if we were replacing
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
                    throw GemmaServerError.invalidRequestStructure(details: "File not found: \(filePath)")
                }
                
                // Read file, limit size to prevent blowing up the context accidentally
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                let size = attributes[.size] as? UInt64 ?? 0
                
                if size > 1024 * 1024 { // 1MB limit for safety
                    throw GemmaServerError.invalidRequestStructure(details: "File too large to inject (>1MB): \(filePath)")
                }
                
                let content = try String(contentsOf: url, encoding: .utf8)
                appendedContext += "\nFile: \(filePath)\n```\n\(content)\n```\n"
            }
            finalPrompt += appendedContext
        }
        
        // Remove escape characters from \@
        finalPrompt = finalPrompt.replacingOccurrences(of: "\\@", with: "@")
        
        return finalPrompt
    }
}
