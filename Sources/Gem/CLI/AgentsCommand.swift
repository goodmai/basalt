import Foundation
import ArgumentParser

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct AgentsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "agents",
        abstract: "Analyze and manage agent capabilities",
        subcommands: [AnalyzeSubcommand.self]
    )
}

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct AnalyzeSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "analyze",
        abstract: "Analyze an agent's capability documentation (.md file)"
    )

    @Argument(help: "Path to the markdown file (e.g. agents.md, gemini.md)")
    var file: String

    mutating func run() async throws {
        let fileURL = URL(fileURLWithPath: file)
        
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("Error: File not found at path '\(file)'")
            throw ExitCode.failure
        }
        
        let analyzer = AgentCapabilityAnalyzer()
        let capabilities = try await analyzer.parse(file: fileURL)
        
        if capabilities.isEmpty {
            print("No agent capabilities found in \(file).")
            return
        }
        
        print("\nFound \(capabilities.count) capabilities in \(file):\n")
        
        for (i, cap) in capabilities.enumerated() {
            print("[\(i + 1)] Tool: \(cap.name) (from \(cap.source.rawValue))")
            print("    Description: \(cap.description)")
            if let returnType = cap.returnType {
                print("    Returns:     \(returnType)")
            }
            if !cap.parameters.isEmpty {
                print("    Parameters:")
                for p in cap.parameters {
                    let req = p.isRequired ? "required" : "optional"
                    print("      - \(p.name) (\(p.type), \(req)): \(p.description)")
                }
            }
            print("")
        }
    }
}
