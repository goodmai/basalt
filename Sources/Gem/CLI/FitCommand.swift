import Foundation
import ArgumentParser

/// Epic 16.7: fit Command
/// Analyzes hardware and recommends optimal models
struct FitCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fit",
        abstract: "Analyze hardware and recommend optimal models"
    )
    
    @Option(help: "Filter by task (chat, code, vision, audio)")
    var task: String?
    
    @Option(help: "Filter by modality (text, image, audio, multimodal)")
    var modality: String?
    
    @Flag(help: "Output in JSON format")
    var json = false
    
    @Flag(help: "Show all models, not just top 10")
    var all = false
    
    mutating func run() async throws {
        if !json {
            print(TerminalUI.info("🔍 Analyzing hardware..."))
            print()
        }
        
        let profiler = SystemProfiler()
        let profile = await profiler.detectResources()
        
        if !json {
            displayHardwareInfo(profile)
            print()
        }
        
        // Load model database and filter
        var models = ModelDatabase.allModels
        if let taskStr = task, let taskEnum = ModelTask(rawValue: taskStr.lowercased()) {
            models = models.filter { $0.task == taskEnum }
        }
        if let modStr = modality, let modEnum = Modality(rawValue: modStr.lowercased()) {
            models = models.filter { $0.modality == modEnum }
        }
        
        if !json {
            print(TerminalUI.info("📊 Analyzing \(models.count) models..."))
            print()
        }
        
        let scorer = FitScorer(profile: profile)
        var scored: [FitScore] = []
        for model in models {
            scored.append(await scorer.score(model))
        }
        
        let recommendations = scored
            .sorted { $0.score > $1.score }
            .prefix(all ? scored.count : 10)
        
        if json {
            struct DeviceOutput: Encodable {
                let chip: String
                let total_ram: Int64
                let available_ram: Int64
                let model_budget: Int64
            }
            struct RecOutput: Encodable {
                let model: String
                let fit_level: String
                let score: Double
                let ram_mb: Int64
                let estimated_tps: Int
                let context_window: Int
            }
            struct JsonOutput: Encodable {
                let device: DeviceOutput
                let recommendations: [RecOutput]
            }
            
            let output = JsonOutput(
                device: DeviceOutput(
                    chip: profile.chipModel,
                    total_ram: profile.totalRAM,
                    available_ram: profile.availableRAM,
                    model_budget: Int64(Double(profile.availableRAM) * 0.85)
                ),
                recommendations: recommendations.map { r in
                    RecOutput(
                        model: r.modelName,
                        fit_level: r.fitLevel.rawValue.lowercased(),
                        score: r.score,
                        ram_mb: r.estimatedRAM,
                        estimated_tps: r.estimatedTPS,
                        context_window: r.contextWindow
                    )
                }
            )
            
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            if let data = try? encoder.encode(output), let str = String(data: data, encoding: .utf8) {
                print(str)
            }
        } else {
            if recommendations.isEmpty {
                print(TerminalUI.warning("⚠️  No models match the filters."))
                return
            }
            displayTable(Array(recommendations))

            print()
            print(TerminalUI.dim("Tip: Run `gemm models download <model>` to install"))
            }
            }
    
    private func displayHardwareInfo(_ resources: SystemProfiler.SystemResources) {
        let budget = Double(resources.availableRAM) * 0.85 / 1073741824.0
        let avail = Double(resources.availableRAM) / 1073741824.0
        
        print("Device: \(resources.chipModel) | \(resources.totalRAMGB) GB RAM | \(String(format: "%.1f", avail)) GB available | model budget: \(String(format: "%.1f", budget)) GB")
        print("GPU: \(resources.gpuName) | metal | unified memory: true")
    }
    
    private func displayTable(_ recommendations: [FitScore]) {
        print(TerminalUI.heading("Top Recommendations:"))
        
        let headers = ["Model", "Fit", "Score", "RAM", "TPS", "Context"]
        var rows: [[String]] = []
        
        for rec in recommendations {
            let fitIndicator = "\(rec.fitLevel.emoji) \(rec.fitLevel.rawValue.prefix(4))"
            let score = String(format: "%.1f", rec.score)
            let ram = formatRAM(rec.estimatedRAM)
            let tps = "\(rec.estimatedTPS)"
            let context = "\(rec.contextWindow / 1000)k"
            
            rows.append([
                rec.modelName,
                fitIndicator,
                score,
                ram,
                tps,
                context
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
    
    private func formatRAM(_ mb: Int64) -> String {
        if mb < 1024 {
            return "\(mb) MB"
        } else {
            let gb = Double(mb) / 1024.0
            return String(format: "%.1f GB", gb)
        }
    }
}
