import ArgumentParser
import Foundation

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
public struct CloudCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "cloud",
        abstract: "Manage cloud models, configuration, and costs",
        subcommands: [Configure.self, Test.self, Models.self, Cost.self]
    )
    
    public init() {}
    
    // Config structures for saving to ~/.gem/cloud.json
    public struct CloudConfig: Codable {
        public var apiKey: String
        public var dailyBudget: Double
        public var monthlyBudget: Double
        
        public init(apiKey: String, dailyBudget: Double, monthlyBudget: Double) {
            self.apiKey = apiKey
            self.dailyBudget = dailyBudget
            self.monthlyBudget = monthlyBudget
        }
    }
    
    public static func configURL() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".gem/cloud.json")
    }
    
    public static func loadConfig() -> CloudConfig? {
        guard let data = try? Data(contentsOf: configURL()) else { return nil }
        return try? JSONDecoder().decode(CloudConfig.self, from: data)
    }

    public struct Configure: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(abstract: "Configure OpenRouter API and budgets")
        
        @Option(name: .long, help: "OpenRouter API Key")
        var apiKey: String?
        
        @Option(name: .long, help: "Daily budget limit in USD")
        var dailyBudget: Double?
        
        @Option(name: .long, help: "Monthly budget limit in USD")
        var monthlyBudget: Double?
        
        public init() {}
        
        public mutating func run() async throws {
            var finalApiKey = apiKey ?? ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? ""
            if finalApiKey.isEmpty {
                print("Enter OpenRouter API Key: ", terminator: "")
                if let input = readLine(), !input.isEmpty {
                    finalApiKey = input
                }
            }
            
            var finalDaily = dailyBudget
            if finalDaily == nil {
                print("Enter Daily Budget limit (USD) [5.0]: ", terminator: "")
                if let input = readLine(), let val = Double(input) {
                    finalDaily = val
                } else {
                    finalDaily = 5.0
                }
            }
            
            var finalMonthly = monthlyBudget
            if finalMonthly == nil {
                print("Enter Monthly Budget limit (USD) [50.0]: ", terminator: "")
                if let input = readLine(), let val = Double(input) {
                    finalMonthly = val
                } else {
                    finalMonthly = 50.0
                }
            }
            
            let config = CloudConfig(apiKey: finalApiKey, dailyBudget: finalDaily!, monthlyBudget: finalMonthly!)
            
            let url = CloudCommand.configURL()
            let dir = url.deletingLastPathComponent()
            if !FileManager.default.fileExists(atPath: dir.path) {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            }
            
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(config)
            
            FileManager.default.createFile(atPath: url.path, contents: data, attributes: [.posixPermissions: NSNumber(value: 0o600)])
            print("Cloud configuration saved to \(url.path)")
        }
    }
    
    public struct Test: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(abstract: "Test cloud connection")
        public init() {}
        public mutating func run() async throws {
            guard let config = CloudCommand.loadConfig() else {
                print("Cloud config not found. Run 'Gem cloud configure' first.")
                return
            }
            
            let clientConfig = OpenRouterClient.Config(
                apiKey: config.apiKey,
                baseURL: URL(string: "https://openrouter.ai/api/v1")!,
                timeout: 60.0,
                maxRetries: 1
            )
            let client = try OpenRouterClient(config: clientConfig)
            
            print("Testing connection...")
            let request = OpenRouterClient.ChatRequest(
                model: "google/gemma-2-9b-it:free",
                messages: [.init(role: "user", content: "Say 'Hello, Gem Cloud!' and nothing else.")],
                maxTokens: 20
            )
            
            do {
                let response = try await client.chat(request: request)
                if let content = response.choices.first?.message.content {
                    print("✅ Connection successful!")
                    print("Response: \(content)")
                } else {
                    print("❌ No response content")
                }
            } catch {
                print("❌ Connection failed: \(error)")
            }
        }
    }
    
    public struct Models: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(abstract: "List available cloud models")
        public init() {}
        public mutating func run() async throws {
            guard let config = CloudCommand.loadConfig() else {
                print("Cloud config not found. Run 'Gem cloud configure' first.")
                return
            }
            
            let clientConfig = OpenRouterClient.Config(
                apiKey: config.apiKey,
                baseURL: URL(string: "https://openrouter.ai/api/v1")!,
                timeout: 60.0,
                maxRetries: 1
            )
            let client = try OpenRouterClient(config: clientConfig)
            
            print("Fetching cloud models...")
            do {
                let models = try await client.getModels()
                print("✅ Found \(models.count) models.")
                for model in models.prefix(20) {
                    print("- \(model.id) (\(model.name))")
                }
                if models.count > 20 {
                    print("... and \(models.count - 20) more.")
                }
            } catch {
                print("❌ Failed to fetch models: \(error)")
            }
        }
    }
    
    public struct Cost: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(abstract: "Show cloud cost statistics")
        public init() {}
        public mutating func run() async throws {
            guard let config = CloudCommand.loadConfig() else {
                print("Cloud config not found. Run 'Gem cloud configure' first.")
                return
            }
            
            let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".gem/cost.json")
            let budget = CostTracker.Budget(dailyLimit: config.dailyBudget, monthlyLimit: config.monthlyBudget)
            let tracker = try CostTracker(storageURL: url, budget: budget)
            
            let stats = await tracker.getStats()
            print("--- Cloud Cost Statistics ---")
            print("Daily Usage:   $\(String(format: "%.4f", stats.dailyUsage)) / $\(config.dailyBudget)")
            print("Monthly Usage: $\(String(format: "%.4f", stats.monthlyUsage)) / $\(config.monthlyBudget)")
            print("\nModel Breakdown:")
            if stats.modelBreakdown.isEmpty {
                print("  No usage recorded yet.")
            } else {
                for (model, cost) in stats.modelBreakdown {
                    print("  - \(model): $\(String(format: "%.4f", cost))")
                }
            }
        }
    }
}
