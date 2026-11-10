import Foundation
import ArgumentParser
import BasaltCore
import BasaltUI

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
    
    @Flag(name: .long, help: "Perform quick hardware diagnostic check and output machine-readable JSON without downloading or ranking models")
    var check = false

    @Option(name: .customLong("pipeline-preset"), help: "Show the compute plan of a pipeline preset on this Mac instead of ranking models: its models, where each runs, resident bytes against the working set, the power-degraded state, and the refusal when the set does not fit (\(PipelinePreset.names))")
    var pipelinePreset: PipelinePreset?

    @Option(name: .customLong("pipeline-memory"), help: "With --pipeline-preset: resident, sequential or auto, as serve --pipeline-memory (default: auto)")
    var pipelineMemory: PipelineMemory = .auto

    mutating func run() async throws {
        if let pipelinePreset {
            return Self.printPlan(pipelinePreset.plan(budgetBytes: await SystemProfiler().detectResources().gpuMemory, power: PowerSource.current(), memory: pipelineMemory), json: json)
        }
        if check {
            try printJSON(await SystemProfiler().generateReport())
            return
        }

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
        
        let models = filteredModels()
        
        if !json {
            print(TerminalUI.info("📊 Analyzing \(models.count) models..."))
            print()
        }
        
        let scorer = FitScorer(profile: profile)
        var scored: [FitScore] = []
        for model in models {
            scored.append(await scorer.score(model))
        }
        
        let recommendations = Array(scored
            .sorted { $0.score > $1.score }
            .prefix(all ? scored.count : 10))
        
        if json {
            Self.printReport(JSONReport(profile: profile, recommendations: recommendations))
        } else if recommendations.isEmpty {
            print(TerminalUI.warning("⚠️  No models match the filters."))
        } else {
            displayTable(recommendations)

            print()
            print(TerminalUI.dim("Tip: Run `basalt models download <model>` to install"))
        }
    }

    private func filteredModels() -> [ModelDefinition] {
        var models = ModelDatabase.allModels
        if let taskStr = task, let taskEnum = ModelTask(rawValue: taskStr.lowercased()) {
            models = models.filter { $0.task == taskEnum }
        }
        if let modStr = modality, let modEnum = Modality(rawValue: modStr.lowercased()) {
            models = models.filter { $0.modality == modEnum }
        }
        return models
    }

    static func printPlan(_ plan: PipelinePresetPlan, json: Bool) {
        guard json else { return print(plan.lines.joined(separator: "\n")) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.keyEncodingStrategy = .convertToSnakeCase
        print(String(decoding: (try? encoder.encode(plan)) ?? Data(), as: UTF8.self))
    }

    private static func printReport(_ report: JSONReport) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        if let data = try? encoder.encode(report), let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
    
    private func displayHardwareInfo(_ resources: SystemProfiler.SystemResources) {
        let budget = Double(resources.availableRAM) * 0.85 / 1073741824.0
        let avail = Double(resources.availableRAM) / 1073741824.0
        
        print("Device: \(resources.chipModel) | \(resources.totalRAMGB) GB RAM | \(String(format: "%.1f", avail)) GB available | model budget: \(String(format: "%.1f", budget)) GB")
        print("GPU: \(resources.gpuName) | \(AppleSiliconChip.gpuSummary(architecture: resources.gpuArchitecture, cores: resources.gpuCores, neuralAccelerators: resources.neuralAccelerators)) | metal | unified memory: true")
    }
    
    private func displayTable(_ recommendations: [FitScore]) {
        print(TerminalUI.heading("Top Recommendations:"))
        
        let headers = ["Model", "Fit", "Score", "RAM", "TPS", "Context", "Best width", "TTFT 4k / 16k"]
        var rows: [[String]] = []
        
        for rec in recommendations {
            rows.append(Self.tableRow(rec))
        }
        
        let table = TableRenderer.render(
            headers: headers,
            rows: rows,
            style: .unicode,
            highlight: .headers
        )
        
        print(table)
    }
    
    static func tableRow(_ rec: FitScore) -> [String] {
        [
            rec.modelName,
            "\(rec.fitLevel.emoji) \(rec.fitLevel.rawValue.prefix(4))",
            String(format: "%.1f", rec.score),
            formatRAM(rec.estimatedRAM),
            rec.isMeasured ? "\(rec.estimatedTPS) measured ✓" : "\(rec.estimatedTPS)",
            "\(rec.contextWindow / 1000)k",
            rec.optimalQuantization ?? "—",
            [4096, 16384].map { rec.estimatedTTFT(promptTokens: $0).map { String(format: "%.1fs", $0) } ?? "—" }.joined(separator: " / ")
        ]
    }

    private static func formatRAM(_ mb: Int64) -> String {
        if mb < 1024 {
            return "\(mb) MB"
        } else {
            let gb = Double(mb) / 1024.0
            return String(format: "%.1f GB", gb)
        }
    }
}

extension FitCommand {
    struct JSONReport: Encodable {
        struct Device: Encodable {
            let chip: String
            let total_ram: Int64
            let available_ram: Int64
            let model_budget: Int64
            let gpu_architecture: String?
            let gpu_cores: Int?
            let neural_accelerators: Bool?
        }

        struct Recommendation: Encodable {
            let model: String
            let fit_level: String
            let score: Double
            let ram_mb: Int64
            let estimated_tps: Int
            let context_window: Int
            let measured: Bool
            let calibration_factor: Double
            let quantization: String
            let optimal_quantization: String?
            let estimated_prefill_tps: Double?
            let estimated_ttft_4k_s: Double?
            let estimated_ttft_16k_s: Double?
        }

        let device: Device
        let recommendations: [Recommendation]

        init(profile: SystemProfiler.SystemResources, recommendations: [FitScore]) {
            device = Device(
                chip: profile.chipModel,
                total_ram: profile.totalRAM,
                available_ram: profile.availableRAM,
                model_budget: Int64(Double(profile.availableRAM) * 0.85),
                gpu_architecture: profile.gpuArchitecture,
                gpu_cores: profile.gpuCores,
                neural_accelerators: profile.neuralAccelerators
            )
            self.recommendations = recommendations.map { r in
                Recommendation(
                    model: r.modelName,
                    fit_level: r.fitLevel.rawValue.lowercased(),
                    score: r.score,
                    ram_mb: r.estimatedRAM,
                    estimated_tps: r.estimatedTPS,
                    context_window: r.contextWindow,
                    measured: r.isMeasured,
                    calibration_factor: r.calibrationFactor,
                    quantization: r.selectedQuantization,
                    optimal_quantization: r.optimalQuantization,
                    estimated_prefill_tps: r.estimatedPrefillTPS,
                    estimated_ttft_4k_s: r.estimatedTTFT(promptTokens: 4096),
                    estimated_ttft_16k_s: r.estimatedTTFT(promptTokens: 16384)
                )
            }
        }
    }
}
