// MODULE: OnboardCommand
// OWNER: AI Assistant
// STATUS: STABLE
// LAST_MODIFIED: 2025-04-28

import Foundation
import ArgumentParser

/// CLI command for interactive onboarding
struct OnboardCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "onboard",
        abstract: "Interactive first-time setup wizard",
        discussion: """
            Guides you through:
            • System resource detection (RAM, GPU, CPU)
            • Model recommendation based on your hardware
            • Initial model download and testing
            
            This command is automatically run on first launch.
            You can re-run it anytime to reconfigure.
            """
    )
    
    @Flag(name: .long, help: "Reset onboarding state and run again")
    var reset: Bool = false
    
    @Flag(name: .long, help: "Show system profile only (no interactive setup)")
    var profileOnly: Bool = false
    
    func run() async throws {
        let onboarding = OnboardingFlow()
        
        // Reset if requested
        if reset {
            try await onboarding.reset()
            print("✅ Onboarding state reset")
            print("")
        }
        
        // Profile-only mode
        if profileOnly {
            await showProfileOnly()
            return
        }
        
        // Check if already completed
        if await onboarding.isCompleted() && !reset {
            print("")
            print("✅ Onboarding already completed!")
            print("")
            print("To re-run onboarding, use:")
            print("  Gem onboard --reset")
            print("")
            print("To see system profile:")
            print("  Gem onboard --profile-only")
            print("")
            return
        }
        
        // Run interactive onboarding
        try await onboarding.run()
    }
    
    /// Show system profile without interactive setup
    private func showProfileOnly() async {
        let profiler = SystemProfiler()
        let resources = await profiler.detectResources()
        
        print("")
        print("📊 System Profile")
        print("═══════════════════════════════════════════════════")
        print("")
        print("Hardware:")
        print("  RAM:          \(resources.totalRAMGB) GB total")
        print("  Available RAM:\(String(format: "%.1f", Double(resources.availableRAM) / 1_073_741_824)) GB")
        print("  GPU:          \(resources.gpuName)")
        print("  GPU Memory:   \(resources.gpuMemoryGB > 0 ? "\(resources.gpuMemoryGB) GB" : "Shared with system")")
        print("  CPU Cores:    \(resources.cpuCores)")
        print("  Chip:         \(resources.chipModel)")
        print("")
        print("Storage:")
        print("  Available:    \(resources.diskSpaceGB) GB")
        print("")
        print("System:")
        print("  macOS:        \(resources.osVersion)")
        print("")
        
        let recommendation = await profiler.recommendModel(resources: resources)
        
        print("Recommended Model:")
        print("  Name:         \(recommendation.modelName)")
        print("  ID:           \(recommendation.modelId)")
        print("  RAM Usage:    ~\(recommendation.estimatedRAM) MB")
        print("  Speed:        ~\(recommendation.estimatedTPS) tokens/second")
        print("  Reason:       \(recommendation.reason)")
        print("")
        print("═══════════════════════════════════════════════════")
        print("")
    }
}
