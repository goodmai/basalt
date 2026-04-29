// MODULE: OnboardingFlow
// OWNER: AI Assistant
// STATUS: IN_PROGRESS
// LAST_MODIFIED: 2025-04-28
// TODO: Add model download with progress bar
// TODO: Add scheduled benchmark support

import Foundation

/// Interactive onboarding flow for first-time users
/// Guides users through system profiling, model recommendation, and initial setup
public actor OnboardingFlow {
    
    private let profiler: SystemProfiler
    private let configPath: String
    
    /// Onboarding state persisted to disk
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
    
    public init(configPath: String = "~/.gem/onboarding.json") {
        self.profiler = SystemProfiler()
        self.configPath = (configPath as NSString).expandingTildeInPath
    }
    
    /// Check if onboarding has been completed
    public func isCompleted() -> Bool {
        guard let state = loadState() else {
            return false
        }
        return state.completed
    }
    
    /// Run interactive onboarding flow
    public func run() async throws {
        print("")
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                                                                      ║")
        print("║                 🚀 Welcome to Gem! 🚀                        ║")
        print("║                                                                      ║")
        print("║            Your Local LLM Inference Server for Apple Silicon        ║")
        print("║                                                                      ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print("")
        
        // Step 1: Detect system resources
        print("🔍 Detecting system resources...")
        let resources = await profiler.detectResources()
        
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
        
        // Step 2: Recommend model
        let recommendation = await profiler.recommendModel(resources: resources)
        
        print("✨ Recommended Model:")
        print("  ┌─────────────────────────────────────────────────")
        print("  │ Model:     \(recommendation.modelName)")
        print("  │ ID:        \(recommendation.modelId)")
        print("  │ RAM Usage: ~\(recommendation.estimatedRAM) MB")
        print("  │ Speed:     ~\(recommendation.estimatedTPS) tokens/second")
        print("  │ Reason:    \(recommendation.reason)")
        print("  └─────────────────────────────────────────────────")
        print("")
        
        // Step 3: Ask user preference
        print("🎯 Setup Options:")
        print("  1️⃣  Quick Start - Download and test recommended model now (~5 min)")
        print("  2️⃣  Custom Model - I'll specify my own model")
        print("  3️⃣  Skip Setup  - I'll configure manually later")
        print("")
        print("Enter your choice (1/2/3): ", terminator: "")
        
        guard let choice = readLine()?.trimmingCharacters(in: .whitespaces) else {
            print("❌ Invalid input. Exiting onboarding.")
            return
        }
        
        let selectedModel: String?
        
        switch choice {
        case "1":
            selectedModel = recommendation.modelId
            print("")
            print("✅ Selected: \(recommendation.modelName)")
            print("")
            print("📥 Next steps:")
            print("  1. Download model: mlx_lm.download(\"\(recommendation.modelId)\")")
            print("  2. Start server:   Gem serve --model \(recommendation.modelId)")
            print("")
            
        case "2":
            print("")
            print("📝 Enter model ID (e.g., 'mlx-community/Qwen2.5-7B-Instruct-4bit'): ", terminator: "")
            guard let customModel = readLine()?.trimmingCharacters(in: .whitespaces), !customModel.isEmpty else {
                print("❌ Invalid model ID. Using recommended model.")
                selectedModel = recommendation.modelId
                break
            }
            selectedModel = customModel
            print("")
            print("✅ Selected custom model: \(customModel)")
            print("")
            print("📥 Next steps:")
            print("  1. Download model: mlx_lm.download(\"\(customModel)\")")
            print("  2. Start server:   Gem serve --model \(customModel)")
            print("")
            
        case "3":
            print("")
            print("⏭️  Skipping setup. You can run onboarding again with:")
            print("   Gem onboard")
            print("")
            selectedModel = nil
            
        default:
            print("❌ Invalid choice. Defaulting to recommended model.")
            selectedModel = recommendation.modelId
        }
        
        // Step 4: Save onboarding state
        let state = OnboardingState(
            completed: true,
            completedAt: Date(),
            selectedModel: selectedModel,
            benchmarkCompleted: false,
            systemResources: resources
        )
        
        try saveState(state)
        
        print("")
        print("✅ Onboarding completed!")
        print("")
        print("📚 Helpful commands:")
        print("  • List models:     Gem list-models")
        print("  • Start server:    Gem serve --model <model-id>")
        print("  • Run benchmark:   Gem benchmark --model <model-id>")
        print("  • Get help:        Gem --help")
        print("")
        print("🎉 Happy inferencing!")
        print("")
    }
    
    /// Reset onboarding state (for testing or re-running)
    public func reset() throws {
        let url = URL(fileURLWithPath: configPath)
        try? FileManager.default.removeItem(at: url)
    }
    
    // MARK: - Private Helpers
    
    /// Load onboarding state from disk
    private func loadState() -> OnboardingState? {
        let url = URL(fileURLWithPath: configPath)
        
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              let state = try? JSONDecoder().decode(OnboardingState.self, from: data) else {
            return nil
        }
        
        return state
    }
    
    /// Save onboarding state to disk
    private func saveState(_ state: OnboardingState) throws {
        let url = URL(fileURLWithPath: configPath)
        
        // Create directory if needed
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
