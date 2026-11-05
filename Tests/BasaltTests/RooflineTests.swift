import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI

@Suite("Generation-aware roofline (#272)")
struct RooflineTests {

    static func resources(chip: String, cores: Int? = nil, accelerators: Bool? = nil) -> SystemProfiler.SystemResources {
        SystemProfiler.SystemResources(
            totalRAM: 64 << 30, availableRAM: 48 << 30, cpuCores: 12, gpuName: "Apple \(chip)",
            gpuMemory: 48 << 30, diskSpace: 500 << 30, osVersion: "27.0", chipModel: chip,
            gpuArchitecture: accelerators == true ? "applegpu_g17s" : "applegpu_g16s", gpuCores: cores,
            neuralAccelerators: accelerators)
    }

    static func model(_ id: String, active: Double, quantization: String = "4-bit") -> ModelDefinition {
        ModelDefinition(id: id, name: id, family: .qwen, task: .chat, modality: .text,
                        spec: .init(ramMB: 8000, contextWindow: 32768, quantization: quantization, quality: 0.9,
                                    totalParameters: active, activeParameters: active))
    }

    static func measurement(_ id: String, chip: String = "M4 Pro", promptTokens: Int?, ttft: Double) -> BenchmarkMeasurement {
        var measurement = BenchmarkMeasurement(modelId: id, chipModel: chip, measuredTPS: 30, measuredTTFT: ttft)
        measurement.promptTokens = promptTokens
        return measurement
    }

    static func profile(_ measurements: BenchmarkMeasurement...) -> HardwareBenchmarkProfile {
        HardwareBenchmarkProfile(chipModel: "M4 Pro", measurements: Dictionary(uniqueKeysWithValues: measurements.map { ($0.modelId, $0) }))
    }

    @Test("Every chip string maps to Apple's published bandwidth", arguments: [
        ("Apple M1", nil, 68.0), ("Apple M1 Pro", nil, 200), ("Apple M1 Max", nil, 400), ("Apple M1 Ultra", nil, 800),
        ("Apple M2", nil, 100), ("Apple M2 Pro", nil, 200), ("Apple M2 Max", nil, 400), ("Apple M2 Ultra", nil, 800),
        ("Apple M3", nil, 100), ("Apple M3 Pro", nil, 150), ("Apple M3 Max", 40, 400), ("Apple M3 Max", 30, 300),
        ("Apple M3 Ultra", nil, 819),
        ("Apple M4", nil, 120), ("Apple M4 Pro", 16, 273), ("Apple M4 Max", 40, 546), ("Apple M4 Max", 32, 410),
        ("Apple M5", nil, 153), ("Apple M5 Pro", nil, 307), ("Apple M5 Max", 40, 614), ("Apple M5 Max", 32, 460),
        ("Apple M5 Max", nil, 614), ("Apple M5 Ultra", nil, 1200),
        ("Apple M6", 12, 170), ("Apple M6 Pro", nil, 170),
        ("Apple Silicon", nil, 100), ("Apple M10", nil, 100),
    ] as [(String, Int?, Double)])
    func bandwidthTable(chip: String, cores: Int?, expected: Double) {
        #expect(FitScorer.memoryBandwidthGBs(chip: chip, gpuName: chip, gpuCores: cores) == expected)
    }

    @Test("Chip strings parse to generation and tier")
    func chipParsing() {
        #expect(AppleSiliconChip("M4 Pro") == AppleSiliconChip("Apple M4 Pro"))
        #expect(AppleSiliconChip("Apple M6")?.generation == 6)
        #expect(AppleSiliconChip("Apple M5 Max")?.tier == .max)
        #expect(AppleSiliconChip("Apple M10")?.generation == 10)
        #expect(AppleSiliconChip("Intel") == nil)
        #expect(AppleSiliconChip("Apple M4")?.hasNeuralAccelerators == false)
        #expect(AppleSiliconChip("Apple M5")?.hasNeuralAccelerators == true)
        #expect(AppleSiliconChip("Apple M6")?.hasNeuralAccelerators == true)
    }

    @Test("Neural accelerators need GPU generation 17 and macOS 26.2", arguments: [
        ("applegpu_g17s", 26, 2, true), ("applegpu_g17s", 26, 1, false), ("applegpu_g16s", 26, 2, false),
        ("applegpu_g18p", 27, 0, true), ("applegpu_g17d", 25, 9, false), ("", 27, 0, false),
    ])
    func neuralAcceleratorFlag(architecture: String, major: Int, minor: Int, expected: Bool) {
        let os = OperatingSystemVersion(majorVersion: major, minorVersion: minor, patchVersion: 0)
        #expect(AppleSiliconChip.neuralAcceleratorsActive(architecture: architecture, os: os) == expected)
    }

    @Test("Decode is η · bandwidth over the active bytes per token")
    func decodeFormula() {
        let model = Self.model("test/roofline-dense-10b", active: 10)
        let m5Pro = FitScorer.decodeSpeed(model: model, bits: 4, contextTokens: 0,
                                          resources: Self.resources(chip: "M5 Pro"), profile: HardwareBenchmarkProfile())
        #expect(abs(m5Pro.tps - 307 * 0.5 / 5.01) < 1e-9)
        #expect(m5Pro.measured == false)

        let m6 = FitScorer.decodeSpeed(model: model, bits: 3, contextTokens: 0,
                                       resources: Self.resources(chip: "M6"), profile: HardwareBenchmarkProfile())
        #expect(abs(m6.tps - 170 * 0.5 / 3.76) < 1e-9)

        let binned = FitScorer.decodeSpeed(model: model, bits: 4, contextTokens: 0,
                                           resources: Self.resources(chip: "M5 Max", cores: 32), profile: HardwareBenchmarkProfile())
        #expect(abs(binned.tps - 460 * 0.5 / 5.01) < 1e-9)
    }

    @Test("Prefill is compute over 2·P_active FLOPs per token, ×3.5 with neural accelerators")
    func prefillFormula() {
        #expect(FitScorer.prefillSpeed(activeParameters: 10, computeTFLOPS: 10, neuralAccelerators: false) == 500)
        #expect(FitScorer.prefillSpeed(activeParameters: 10, computeTFLOPS: 10, neuralAccelerators: true) == 1750)
        #expect(FitScorer.prefillSpeed(activeParameters: 4, computeTFLOPS: 8, neuralAccelerators: false) == 1000)
    }

    @Test("A prefill measurement of the model itself is used as is")
    func prefillOwnMeasurement() throws {
        let profile = Self.profile(Self.measurement("org/own-8b", promptTokens: 4096, ttft: 2))
        let speed = try #require(FitScorer.prefillSpeed(model: Self.model("org/own-8b", active: 8),
                                                        resources: Self.resources(chip: "M4 Pro"), profile: profile))
        #expect(speed.tps == 2048)
        #expect(speed.measured)
    }

    @Test("Other models on this chip calibrate the compute figure through their median")
    func prefillAnchors() throws {
        let profile = Self.profile(
            Self.measurement("org/anchor-20b", promptTokens: 4000, ttft: 4),
            Self.measurement("org/anchor-10b", promptTokens: 4000, ttft: 4),
            Self.measurement("org/anchor-30B-A30B", promptTokens: 4000, ttft: 4))
        for accelerators in [false, true] {
            let speed = try #require(FitScorer.prefillSpeed(model: Self.model("org/target-10b", active: 10),
                                                            resources: Self.resources(chip: "M4 Pro", accelerators: accelerators),
                                                            profile: profile))
            #expect(abs(speed.tps - 2000) < 1e-9)
            #expect(!speed.measured)
        }
    }

    @Test("No prefill estimate from short prompts, another chip, or an empty profile")
    func prefillWithoutAnchor() {
        let model = Self.model("org/target-10b", active: 10)
        let m4Pro = Self.resources(chip: "M4 Pro")
        #expect(FitScorer.prefillSpeed(model: model, resources: m4Pro, profile: HardwareBenchmarkProfile()) == nil)
        #expect(FitScorer.prefillSpeed(model: model, resources: m4Pro,
                                       profile: Self.profile(Self.measurement("org/a-20b", promptTokens: 512, ttft: 0.5))) == nil)
        #expect(FitScorer.prefillSpeed(model: model, resources: m4Pro,
                                       profile: Self.profile(Self.measurement("org/a-20b", chip: "M4 Max", promptTokens: 4096, ttft: 2))) == nil)
        #expect(FitScorer.prefillSpeed(model: model, resources: m4Pro,
                                       profile: Self.profile(Self.measurement("org/a-20b", promptTokens: nil, ttft: 2))) == nil)
    }

    @Test("Measurements match chips past M5 and never M1 against M10")
    func chipMatchingPastM5() {
        let m6 = Self.profile(Self.measurement("org/m", chip: "M6", promptTokens: nil, ttft: 0.1))
        #expect(m6.measurement(for: "org/m", currentChip: "Apple M6") != nil)
        #expect(m6.measurement(for: "org/m", currentChip: "M6 Pro") == nil)
        let m1 = Self.profile(Self.measurement("org/m", chip: "M1", promptTokens: nil, ttft: 0.1))
        #expect(m1.measurement(for: "org/m", currentChip: "M10") == nil)
    }

    @Test("fit scores decode from the generation table and adds the prefill estimate")
    func fitUsesGenerationTable() async {
        let model = Self.model("test/roofline-dense-8b", active: 8)
        let kv = FitScorer.kvReadGB(model.kvGeometry, contextTokens: model.contextWindow)
        let profile = Self.profile(Self.measurement("org/anchor-16b", chip: "M6", promptTokens: 4096, ttft: 4))
        let score = await FitScorer(profile: Self.resources(chip: "M6", cores: 12, accelerators: true), benchmarkProfile: profile).score(model)
        #expect(score.estimatedTPS == Int((170 * 0.5 / (8 * 0.5 + kv)).rounded()))
        #expect(score.estimatedPrefillTPS.map { abs($0 - 2048) < 1e-9 } == true)
        #expect(score.estimatedTTFT(promptTokens: 4096).map { abs($0 - 2) < 1e-9 } == true)
    }

    @Test("fit shows TTFT for 4k and 16k prompts and the GPU in its JSON")
    func fitOutput() throws {
        var score = FitScore(modelName: "M", fitLevel: .good, score: 80, estimatedRAM: 8000, estimatedTPS: 40, contextWindow: 32768)
        #expect(FitCommand.tableRow(score)[7] == "— / —")
        score.estimatedPrefillTPS = 2048
        #expect(FitCommand.tableRow(score)[7] == "2.0s / 8.0s")

        let report = FitCommand.JSONReport(profile: Self.resources(chip: "M6", cores: 12, accelerators: true), recommendations: [score])
        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any])
        let device = try #require(json["device"] as? [String: Any])
        #expect(device["gpu_architecture"] as? String == "applegpu_g17s")
        #expect(device["gpu_cores"] as? Int == 12)
        #expect(device["neural_accelerators"] as? Bool == true)
        let recommendation = try #require((json["recommendations"] as? [[String: Any]])?.first)
        #expect(recommendation["estimated_prefill_tps"] as? Double == 2048)
        #expect(recommendation["estimated_ttft_4k_s"] as? Double == 2)
        #expect(recommendation["estimated_ttft_16k_s"] as? Double == 8)
    }

    @Test("/api/v1/stats carries the GPU architecture, core count and accelerator flag")
    func telemetryContract() async throws {
        let hardware = await SystemProfiler().detectResources()
        let snapshot = await TelemetryTracker().snapshot()
        #expect(snapshot.gpuArchitecture == hardware.gpuArchitecture)
        #expect(snapshot.gpuCores == hardware.gpuCores)
        #expect(snapshot.neuralAccelerators == hardware.neuralAccelerators)

        var fixed = ServerTelemetrySnapshot()
        fixed.gpuArchitecture = "applegpu_g17s"
        fixed.gpuCores = 10
        fixed.neuralAccelerators = true
        let json = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(fixed)) as? [String: Any])
        #expect(json["gpuArchitecture"] as? String == "applegpu_g17s")
        #expect(json["gpuCores"] as? Int == 10)
        #expect(json["neuralAccelerators"] as? Bool == true)
    }
}
