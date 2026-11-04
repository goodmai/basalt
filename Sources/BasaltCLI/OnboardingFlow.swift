
import Foundation
import BasaltCore

public actor OnboardingFlow {
    
    private let profiler: SystemProfiler
    private let configPath: String
    private let serverConfigPath: String
    
    public struct OnboardingState: Codable, Sendable {
        public let completed: Bool
        public let completedAt: Date?
        public let selectedModel: String?
        public let benchmarkCompleted: Bool
        public let systemResources: SystemProfiler.SystemResources?
        
        public init(
            completed: Bool = false,
            completedAt: Date? = nil,
            selectedModel: String? = nil,
            benchmarkCompleted: Bool = false,
            systemResources: SystemProfiler.SystemResources? = nil
        ) {
            self.completed = completed
            self.completedAt = completedAt
            self.selectedModel = selectedModel
            self.benchmarkCompleted = benchmarkCompleted
            self.systemResources = systemResources
        }
    }
    
    public init(
        configPath: String = "~/.basalt/onboarding.json",
        serverConfigPath: String = "~/.basalt/config.json"
    ) {
        self.profiler = SystemProfiler()
        let resolvedConfig = (configPath as NSString).expandingTildeInPath
        let legacyConfig = ("~/.gem/onboarding.json" as NSString).expandingTildeInPath
        self.configPath = (!FileManager.default.fileExists(atPath: resolvedConfig) && FileManager.default.fileExists(atPath: legacyConfig) && configPath == "~/.basalt/onboarding.json") ? legacyConfig : resolvedConfig
        let resolvedServerConfig = (serverConfigPath as NSString).expandingTildeInPath
        let legacyServerConfig = ("~/.gem/config.json" as NSString).expandingTildeInPath
        self.serverConfigPath = (!FileManager.default.fileExists(atPath: resolvedServerConfig) && FileManager.default.fileExists(atPath: legacyServerConfig) && serverConfigPath == "~/.basalt/config.json") ? legacyServerConfig : resolvedServerConfig
    }
    
    static func launchArgs(existing: [String]?, model: String) -> [String] {
        guard var args = existing, !args.isEmpty else { return ["serve", "--model", model] }
        if let i = args.firstIndex(of: "--model"), i + 1 < args.count {
            args[i + 1] = model
        } else {
            args += ["--model", model]
        }
        return args
    }

    public func isCompleted() -> Bool {
        guard let state = loadState() else {
            return false
        }
        return state.completed
    }
    
    public func run() async throws {
        Self.printWelcome()
        
        print("🔍 Detecting system resources...")
        let resources = await profiler.detectResources()
        Self.printProfile(resources)
        
        let recommendation = await profiler.recommendModel(resources: resources)
        Self.printRecommendation(recommendation)
        Self.printSetupOptions()
        
        guard let choice = readLine()?.trimmingCharacters(in: .whitespaces) else {
            print("❌ Invalid input. Exiting onboarding.")
            return
        }
        
        let selectedModel = Self.selectedModel(choice, recommendation: recommendation)
        
        let state = OnboardingState(
            completed: true,
            completedAt: Date(),
            selectedModel: selectedModel,
            benchmarkCompleted: false,
            systemResources: resources
        )
        
        try saveState(state)
        saveLaunchConfig(model: selectedModel ?? recommendation.modelId)
        Self.printFarewell()
    }

    static func selectedModel(_ choice: String, recommendation: SystemProfiler.ModelRecommendation) -> String? {
        switch choice {
        case "1":
            printNextSteps("✅ Selected: \(recommendation.modelName)", model: recommendation.modelId)
            return recommendation.modelId
            
        case "2":
            print("")
            print("📝 Enter model ID (e.g., 'mlx-community/Qwen2.5-7B-Instruct-4bit'): ", terminator: "")
            guard let customModel = readLine()?.trimmingCharacters(in: .whitespaces), !customModel.isEmpty else {
                print("❌ Invalid model ID. Using recommended model.")
                return recommendation.modelId
            }
            printNextSteps("✅ Selected custom model: \(customModel)", model: customModel)
            return customModel
            
        case "3":
            print("")
            print("⏭️  Skipping setup. You can run onboarding again with:")
            print("   basalt onboard")
            print("")
            return nil
            
        default:
            print("❌ Invalid choice. Defaulting to recommended model.")
            return recommendation.modelId
        }
    }

    private func saveLaunchConfig(model: String) {
        let serverConfigURL = URL(fileURLWithPath: serverConfigPath)
        let exe = Bundle.main.executablePath ?? ProcessInfo.processInfo.arguments.first ?? "basalt"
        do {
            let existingArgs = (try? ServerConfig.load(from: serverConfigURL))?["args"] as? [String]
            try ServerConfig.mergeAndSave(
                to: serverConfigURL,
                systemUpdates: [
                    "exe": exe,
                    "args": Self.launchArgs(existing: existingArgs, model: model)
                ],
                defaults: ["port": 8080, "host": "127.0.0.1"],
                createBackup: true
            )
        } catch {
            print("⚠️  Could not update \(serverConfigURL.path): \(error.localizedDescription)")
            print("   Onboarding state was saved; fix the file and re-run `onboard` to write the launch config.")
        }
    }
    
    public func reset() throws {
        let url = URL(fileURLWithPath: configPath)
        try? FileManager.default.removeItem(at: url)
    }
    
    
    private func loadState() -> OnboardingState? {
        let url = URL(fileURLWithPath: configPath)
        
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let state = try? JSONDecoder().decode(OnboardingState.self, from: data) else {
            return nil
        }
        
        return state
    }
    
    private func saveState(_ state: OnboardingState) throws {
        let url = URL(fileURLWithPath: configPath)
        
        let directory = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
        }
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        
        let data = try encoder.encode(state)
        try data.write(to: url)
        
        print("💾 Onboarding state saved to: \(configPath)")
    }
}

extension OnboardingFlow {
    static func printWelcome() {
        print("")
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                                                                      ║")
        print("║                 🚀 Welcome to Basalt! 🚀                     ║")
        print("║                                                                      ║")
        print("║            Your Local LLM Inference Server for Apple Silicon        ║")
        print("║                                                                      ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print("")
    }

    static func printProfile(_ resources: SystemProfiler.SystemResources) {
        print("")
        print("📊 System Profile:")
        print("  ┌─────────────────────────────────────────────────")
        print("  │ RAM:       \(resources.totalRAMGB) GB total, \(resources.totalRAMGB - resources.totalRAMGB / 4) GB recommended available")
        print("  │ GPU:       \(resources.gpuName)")
        print("  │ GPU Memory:\(resources.gpuMemoryGB > 0 ? " \(resources.gpuMemoryGB) GB" : " Shared with system")")
        print("  │ CPU:       \(resources.cpuCores) cores")
        print("  │ Chip:      \(resources.chipModel)")
        print("  │ Disk:      \(resources.diskSpaceGB) GB available")
        print("  │ macOS:     \(resources.osVersion)")
        print("  └─────────────────────────────────────────────────")
        print("")
    }

    static func printRecommendation(_ recommendation: SystemProfiler.ModelRecommendation) {
        print("✨ Recommended Model:")
        print("  ┌─────────────────────────────────────────────────")
        print("  │ Model:     \(recommendation.modelName)")
        print("  │ ID:        \(recommendation.modelId)")
        print("  │ RAM Usage: ~\(recommendation.estimatedRAM) MB")
        print("  │ Speed:     ~\(recommendation.estimatedTPS) tokens/second")
        print("  │ Reason:    \(recommendation.reason)")
        print("  └─────────────────────────────────────────────────")
        print("")
    }

    static func printSetupOptions() {
        print("🎯 Setup Options:")
        print("  1️⃣  Quick Start - Download and test recommended model now (~5 min)")
        print("  2️⃣  Custom Model - I'll specify my own model")
        print("  3️⃣  Skip Setup  - I'll configure manually later")
        print("")
        print("Enter your choice (1/2/3): ", terminator: "")
    }

    static func printNextSteps(_ selection: String, model: String) {
        print("")
        print(selection)
        print("")
        print("📥 Next steps:")
        print("  1. Download model: mlx_lm.download(\"\(model)\")")
        print("  2. Start server:   basalt serve --model \(model)")
        print("")
    }

    static func printFarewell() {
        print("")
        print("✅ Onboarding completed!")
        print("")
        print("📚 Helpful commands:")
        print("  • List models:     basalt models list")
        print("  • Start server:    basalt serve --model <model-id>")
        print("  • Check fit:       basalt models check <model-id>")
        print("  • Get help:        basalt --help")
        print("")
        print("🎉 Happy inferencing!")
        print("")
    }
}
