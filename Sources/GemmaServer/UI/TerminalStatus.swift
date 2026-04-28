import Foundation

/// Epic 16.10: Rich Loading States
public enum LoadingState: String, Sendable {
    case initializing = "Initializing engine..."
    case readingFiles = "Reading context files..."
    case loadingMCP = "Querying MCP servers..."
    case analyzingSkills = "Analyzing agent skills..."
    case thinking = "Thinking..."
    case generating = "Generating tokens..."
    case finalizing = "Finalizing response..."
    
    public var description: String {
        return self.rawValue
    }
}

/// Epic 16.10: Context Statistics for CLI display
public struct ContextStats: Sendable {
    public var files: Int
    public var systemPrompts: Int
    public var mcpServers: Int
    public var skills: Int
    
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
}
