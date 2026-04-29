import Foundation

/// Epic 16.8: Rich Loading States
/// Represents different states during model operations
public enum LoadingState: String, Sendable, CustomStringConvertible {
    case initializing = "Initializing engine..."
    case readingFiles = "Reading context files..."
    case loadingMCP = "Querying MCP servers..."
    case analyzingSkills = "Analyzing agent skills..."
    case thinking = "Thinking..."
    case generating = "Generating tokens..."
    case finalizing = "Finalizing response..."
    
    public var description: String {
        return rawValue
    }
    
    /// Get colored description for terminal output
    public var coloredDescription: String {
        return TerminalUI.dim(rawValue)
    }
}

/// Epic 16.8: Context Statistics for CLI display
/// Thread-safe statistics for context information
public struct ContextStats: Sendable, CustomStringConvertible {
    public let files: Int
    public let systemPrompts: Int
    public let mcpServers: Int
    public let skills: Int
    
    public init(files: Int = 0, systemPrompts: Int = 0, mcpServers: Int = 0, skills: Int = 0) {
        self.files = files
        self.systemPrompts = systemPrompts
        self.mcpServers = mcpServers
        self.skills = skills
    }
    
    /// Returns true if there is any active context
    public var hasContext: Bool {
        return files > 0 || systemPrompts > 0 || mcpServers > 0 || skills > 0
    }
    
    public var description: String {
        var parts: [String] = []
        if files > 0 { parts.append("\(files) file(s)") }
        if systemPrompts > 0 { parts.append("\(systemPrompts) prompt(s)") }
        if mcpServers > 0 { parts.append("\(mcpServers) MCP server(s)") }
        if skills > 0 { parts.append("\(skills) skill(s)") }
        return parts.isEmpty ? "No context" : parts.joined(separator: ", ")
    }
    
    /// Get colored description for terminal output
    public var coloredDescription: String {
        guard hasContext else {
            return TerminalUI.dim("No context")
        }
        
        var parts: [String] = []
        if files > 0 {
            parts.append(TerminalUI.info("\(files) file(s)"))
        }
        if systemPrompts > 0 {
            parts.append(TerminalUI.info("\(systemPrompts) prompt(s)"))
        }
        if mcpServers > 0 {
            parts.append(TerminalUI.success("\(mcpServers) MCP server(s)"))
        }
        if skills > 0 {
            parts.append(TerminalUI.warning("\(skills) skill(s)"))
        }
        
        return parts.joined(separator: ", ")
    }
}
