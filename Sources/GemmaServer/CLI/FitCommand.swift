import Foundation
import ArgumentParser

/// Epic 16.7: fit Command
/// Analyzes hardware and recommends optimal models
struct FitCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fit",
        abstract: "Analyze hardware and recommend optimal models",
        discussion: """
        The fit command profiles your hardware (RAM, GPU, CPU) and recommends
        which MLX models will run best on your system.
        
        It shows:
        • Current hardware specifications
        • Compatible models ranked by fit score
        • Estimated performance (tokens/sec)
        • Memory requirements
        
        Examples:
          gemmaserver fit                    # Show all recommendations
          gemmaserver fit --top 3            # Show top 3 only
          gemmaserver fit --min-score 70     # Only show good fits
          gemmaserver fit --format json      # Output as JSON
        """
    )
    
    @Option(name: .shortAndLong, help: "Number of top recommendations to show")
    var top: Int = 10
    
    @Option(name: .shortAndLong, help: "Minimum fit score (0-100) to display")
    var minScore: Double = 40.0
    
    @Option(name: .shortAndLong, help: "Output format: table, json, plain")
    var format: OutputFormat = .table
    
    @Flag(name: .long, help: "Show detailed analysis for each model")
    var detailed: Bool = false
    
    @Flag(name: .long, help: "Include experimental models")
    var includeExperimental: Bool = false
    
    enum OutputFormat: String, ExpressibleByArgument {
        case table
        case json
        case plain
    }
    
    mutating func run() async throws {
        print(TerminalUI.info("🔍 Analyzing hardware..."))
        print()
        
        // Profile system
        let profiler = SystemProfiler()
        let resources = await profiler.detectResources()
        
        // Display hardware info
        displayHardwareInfo(resources)
        print()
        
        // Get available models
        let models = getAvailableModels(includeExperimental: includeExperimental)
        
        print(TerminalUI.info("📊 Analyzing \(models.count) models..."))
        print()
        
        // Analyze fit
        let recommendations = ModelFitAnalyzer.analyzeModels(
            models,
            resources: resources,
            minScore: minScore
        )
        
        let topRecommendations = Array(recommendations.prefix(top))
        
        if topRecommendations.isEmpty {
            print(TerminalUI.warning("⚠️  No models meet the minimum score of \(minScore)"))
            print(TerminalUI.dim("Try lowering --min-score or upgrading your hardware"))
            return
        }
        
        // Display results
        switch format {
        case .table:
            displayTable(topRecommendations, resources: resources, detailed: detailed)
        case .json:
            displayJSON(topRecommendations, resources: resources)
        case .plain:
            displayPlain(topRecommendations)
        }
        
        // Summary
        print()
        displaySummary(topRecommendations, total: recommendations.count)
    }
    
    // MARK: - Hardware Display
    
    private func displayHardwareInfo(_ resources: SystemProfiler.SystemResources) {
        print(TerminalUI.heading("💻 Hardware Profile"))
        print()
        
        let specs = [
            ("Chip", resources.chipModel),
            ("RAM", "\(resources.totalRAMGB) GB"),
            ("GPU", resources.gpuName),
            ("GPU Memory", "\(resources.gpuMemoryGB) GB"),
            ("CPU Cores", "\(resources.cpuCores)"),
            ("Disk Space", "\(resources.diskSpaceGB) GB available"),
            ("OS", resources.osVersion)
        ]
        
        for (label, value) in specs {
            let paddedLabel = label.padding(toLength: 15, withPad: " ", startingAt: 0)
            print("  \(TerminalUI.dim(paddedLabel)) \(TerminalUI.bold(value))")
        }
    }
    
    // MARK: - Table Display
    
    private func displayTable(
        _ recommendations: [ModelFitAnalyzer.ModelFitResult],
        resources: SystemProfiler.SystemResources,
        detailed: Bool
    ) {
        print(TerminalUI.heading("🎯 Model Recommendations"))
        print()
        
        let headers = ["Model", "Fit", "Score", "RAM", "TPS", "Recommendation"]
        var rows: [[String]] = []
        
        for rec in recommendations {
            let fitIndicator = Self.formatCategory(rec.fitCategory)
            let score = String(format: "%.1f", rec.fitScore)
            let ram = Self.formatRAM(rec.sizeMB)
            let tps = "\(rec.estimatedTPS) TPS"
            
            let recommendation: String
            if detailed {
                recommendation = rec.recommendation
            } else {
                recommendation = rec.fitCategory.rawValue
            }
            
            rows.append([
                rec.modelName,
                fitIndicator,
                score,
                ram,
                tps,
                recommendation
            ])
        }
        
        let table = TableRenderer.render(
            headers: headers,
            rows: rows,
            style: .unicode,
            colorHeaders: true
        )
        
        print(table)
    }
    
    // MARK: - JSON Display
    
    private func displayJSON(
        _ recommendations: [ModelFitAnalyzer.ModelFitResult],
        resources: SystemProfiler.SystemResources
    ) {
        let output: [String: Any] = [
            "hardware": [
                "chip": resources.chipModel,
                "ram_gb": resources.totalRAMGB,
                "gpu": resources.gpuName,
                "gpu_memory_gb": resources.gpuMemoryGB
            ],
            "recommendations": recommendations.map { rec in
                [
                    "model_id": rec.modelId,
                    "model_name": rec.modelName,
                    "fit_score": rec.fitScore,
                    "fit_category": rec.fitCategory.rawValue,
                    "size_mb": rec.sizeMB,
                    "estimated_tps": rec.estimatedTPS,
                    "recommendation": rec.recommendation
                ]
            }
        ]
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
        }
    }
    
    // MARK: - Plain Display
    
    private func displayPlain(_ recommendations: [ModelFitAnalyzer.ModelFitResult]) {
        print("Model Recommendations:")
        print()
        
        for (index, rec) in recommendations.enumerated() {
            print("\(index + 1). \(rec.modelName)")
            print("   Fit Score: \(String(format: "%.1f", rec.fitScore))")
            print("   Category: \(rec.fitCategory.rawValue)")
            print("   Size: \(Self.formatRAM(rec.sizeMB))")
            print("   Est. TPS: \(rec.estimatedTPS)")
            print("   \(rec.recommendation)")
            print()
        }
    }
    
    // MARK: - Summary
    
    private func displaySummary(_ shown: [ModelFitAnalyzer.ModelFitResult], total: Int) {
        let perfectCount = shown.filter { $0.fitCategory == .perfect }.count
        let goodCount = shown.filter { $0.fitCategory == .good }.count
        let tightCount = shown.filter { $0.fitCategory == .tight }.count
        
        print(TerminalUI.dim("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
        print()
        print(TerminalUI.bold("Summary"))
        print()
        
        if perfectCount > 0 {
            print("  🟢 \(perfectCount) perfect fit(s)")
        }
        if goodCount > 0 {
            print("  🟡 \(goodCount) good fit(s)")
        }
        if tightCount > 0 {
            print("  🟠 \(tightCount) tight fit(s)")
        }
        
        if shown.count < total {
            print()
            print(TerminalUI.dim("  Showing top \(shown.count) of \(total) models"))
            print(TerminalUI.dim("  Use --top N or --min-score X to adjust"))
        }
        
        print()
        print(TerminalUI.dim("Run `gemmaserver models download <model-id>` to install"))
    }
    
    // MARK: - Helpers
    
    static func formatCategory(_ category: ModelFitAnalyzer.FitCategory) -> String {
        return "\(category.emoji) \(category.rawValue)"
    }
    
    static func formatRAM(_ mb: Int) -> String {
        if mb < 1024 {
            return "\(mb) MB"
        } else {
            let gb = Double(mb) / 1024.0
            return String(format: "%.1f GB", gb)
        }
    }
    
    private func getAvailableModels(includeExperimental: Bool) -> [ModelInfo] {
        // TODO: Integrate with HuggingFace Hub API to get real model list
        // For now, return curated list of popular MLX models
        
        var models = [
            ModelInfo(
                id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                name: "Qwen 2.5 0.5B (4-bit)",
                sizeMB: 512,
                recommendedRAMGB: 4,
                description: "Ultra-fast tiny model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                name: "Qwen 2.5 1.5B (4-bit)",
                sizeMB: 1100,
                recommendedRAMGB: 4,
                description: "Fast small model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-3B-Instruct-4bit",
                name: "Qwen 2.5 3B (4-bit)",
                sizeMB: 2300,
                recommendedRAMGB: 8,
                description: "Balanced model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                name: "Qwen 2.5 7B (4-bit)",
                sizeMB: 4800,
                recommendedRAMGB: 16,
                description: "High quality model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-14B-Instruct-4bit",
                name: "Qwen 2.5 14B (4-bit)",
                sizeMB: 9500,
                recommendedRAMGB: 24,
                description: "Professional model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-32B-Instruct-4bit",
                name: "Qwen 2.5 32B (4-bit)",
                sizeMB: 19000,
                recommendedRAMGB: 48,
                description: "Advanced model"
            ),
            ModelInfo(
                id: "mlx-community/Qwen2.5-72B-Instruct-4bit",
                name: "Qwen 2.5 72B (4-bit)",
                sizeMB: 42000,
                recommendedRAMGB: 96,
                description: "Top-tier model"
            )
        ]
        
        if includeExperimental {
            models.append(contentsOf: [
                ModelInfo(
                    id: "mlx-community/gemma-2-2b-it",
                    name: "Gemma 2 2B",
                    sizeMB: 2700,
                    recommendedRAMGB: 8,
                    description: "Google's Gemma 2"
                ),
                ModelInfo(
                    id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                    name: "Llama 3.2 1B (4-bit)",
                    sizeMB: 900,
                    recommendedRAMGB: 4,
                    description: "Meta's Llama 3.2"
                )
            ])
        }
        
        return models
    }
}
