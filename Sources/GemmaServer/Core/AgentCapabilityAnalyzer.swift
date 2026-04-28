import Foundation

public struct Parameter: Codable, Sendable {
    public let name: String
    public let type: String
    public let isRequired: Bool
    public let description: String
}

public struct AgentCapability: Codable, Sendable {
    public let name: String
    public let description: String
    public let parameters: [Parameter]
    public let returnType: String?
    public let source: CapabilitySource
    
    public enum CapabilitySource: String, Codable, Sendable {
        case agentsMd = "agents.md"
        case geminiMd = "gemini.md"
        case claudeSkill = "claude-skill.md"
        case unknown
    }
}

public actor AgentCapabilityAnalyzer {
    
    public init() {}
    
    public func parse(file: URL) async throws -> [AgentCapability] {
        let content = try String(contentsOf: file, encoding: .utf8)
        let format = detectFormat(fileName: file.lastPathComponent, content: content)
        
        switch format {
        case .agentsMd:   
            return try parseAgentsMd(content)
        case .geminiMd:   
            return try parseGeminiMd(content)
        case .claudeSkill: 
            return try parseClaudeSkill(content)
        case .unknown:
            return []
        }
    }
    
    private func detectFormat(fileName: String, content: String) -> AgentCapability.CapabilitySource {
        let lower = fileName.lowercased()
        if lower.contains("agents.md") { return .agentsMd }
        if lower.contains("gemini.md") || content.contains("<available_skills>") { return .geminiMd }
        if lower.contains("claude-skill") { return .claudeSkill }
        return .unknown
    }
    
    private func parseAgentsMd(_ content: String) throws -> [AgentCapability] {
        var capabilities: [AgentCapability] = []
        let lines = content.components(separatedBy: .newlines)
        
        var currentName: String?
        var currentDescription: String = ""
        var currentParams: [Parameter] = []
        var currentReturn: String?
        var inParams = false
        
        func commit() {
            if let name = currentName {
                capabilities.append(AgentCapability(
                    name: name,
                    description: currentDescription,
                    parameters: currentParams,
                    returnType: currentReturn,
                    source: .agentsMd
                ))
            }
            currentName = nil
            currentDescription = ""
            currentParams = []
            currentReturn = nil
            inParams = false
        }
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            
            if trimmed.hasPrefix("## Tool:") {
                commit()
                currentName = trimmed.replacingOccurrences(of: "## Tool:", with: "").trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("Description:") {
                currentDescription = trimmed.replacingOccurrences(of: "Description:", with: "").trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("Return Type:") {
                currentReturn = trimmed.replacingOccurrences(of: "Return Type:", with: "").trimmingCharacters(in: .whitespaces)
                inParams = false
            } else if trimmed.hasPrefix("Parameters:") {
                inParams = true
            } else if inParams && trimmed.hasPrefix("-") {
                // e.g. - expression (String, required): The math expression (e.g. "2 + 2").
                let paramContent = trimmed.dropFirst().trimmingCharacters(in: .whitespaces)
                let parts = paramContent.components(separatedBy: ":")
                if parts.count >= 2 {
                    let desc = parts[1...].joined(separator: ":").trimmingCharacters(in: .whitespaces)
                    let nameTypePart = parts[0].trimmingCharacters(in: .whitespaces)
                    
                    if let startParen = nameTypePart.firstIndex(of: "("),
                       let endParen = nameTypePart.firstIndex(of: ")") {
                        let name = String(nameTypePart[..<startParen]).trimmingCharacters(in: .whitespaces)
                        let typeInfo = String(nameTypePart[nameTypePart.index(after: startParen)..<endParen])
                        let typeParts = typeInfo.components(separatedBy: ",")
                        
                        let type = typeParts[0].trimmingCharacters(in: .whitespaces)
                        let req = typeParts.count > 1 ? typeParts[1].trimmingCharacters(in: .whitespaces).lowercased() == "required" : false
                        
                        currentParams.append(Parameter(name: name, type: type, isRequired: req, description: desc))
                    }
                }
            }
        }
        commit()
        
        return capabilities
    }
    
    private func parseGeminiMd(_ content: String) throws -> [AgentCapability] {
        var capabilities: [AgentCapability] = []
        
        // Simple XML/Regex parser for `<skill>` blocks
        let skillPattern = "(?s)<skill>(.*?)</skill>"
        let namePattern = "<name>(.*?)</name>"
        let descPattern = "<description>(.*?)</description>"
        
        let regex = try NSRegularExpression(pattern: skillPattern)
        let matches = regex.matches(in: content, range: NSRange(content.startIndex..., in: content))
        
        for match in matches {
            guard let skillRange = Range(match.range(at: 1), in: content) else { continue }
            let skillContent = String(content[skillRange])
            
            var name = ""
            var desc = ""
            
            if let nMatch = try? NSRegularExpression(pattern: namePattern).firstMatch(in: skillContent, range: NSRange(skillContent.startIndex..., in: skillContent)),
               let nRange = Range(nMatch.range(at: 1), in: skillContent) {
                name = String(skillContent[nRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            if let dMatch = try? NSRegularExpression(pattern: descPattern).firstMatch(in: skillContent, range: NSRange(skillContent.startIndex..., in: skillContent)),
               let dRange = Range(dMatch.range(at: 1), in: skillContent) {
                desc = String(skillContent[dRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            if !name.isEmpty {
                capabilities.append(AgentCapability(
                    name: name,
                    description: desc,
                    parameters: [], // Gemini skills in XML typically don't list parameters this way, but can be added
                    returnType: nil,
                    source: .geminiMd
                ))
            }
        }
        
        return capabilities
    }
    
    private func parseClaudeSkill(_ content: String) throws -> [AgentCapability] {
        // Placeholder for Claude format
        return []
    }
}
