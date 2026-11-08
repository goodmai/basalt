import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
@testable import BasaltCore
@testable import BasaltInference

@Suite("train_jepa: LoRA backbone and projector trained with LeJEPA, clipped and scheduled, saved away from the weights (#386)", .serialized)
struct LeJEPATrainTests {
    static let scenes = """
    ["The cat sat on the warm mat by the door.", "A cat is resting on a mat near the doorway.", "On the mat by the door, a cat sits."]
    {"views": ["Rain fell all night over the quiet city.", "The quiet city had rain the whole night.", "All night, rain on the silent town."]}
    ["She bought three apples at the market today.", "Today at the market she got three apples.", "Three apples were bought by her at the market."]
    ["The train left the station ten minutes late.", "Ten minutes late, the train departed the station.", "The station saw the train leave late by ten minutes."]
    """

    static func scratch() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "lab-jepa-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func weights(_ container: ModelContainer) async -> [String: [Float]] {
        await container.perform { context in
            Dictionary(uniqueKeysWithValues: context.model.parameters().flattened().map { ($0.0, $0.1.asType(.float32).asArray(Float.self)) })
        }
    }

    @Test("20 steps on a tiny model lower the loss, keep base weights bit-identical, save a LoRA that loads and a projector, and report a non-collapsed rank")
    func trains() async throws {
        let container = try TinyModel.container(.qwen3)
        let directory = try Self.scratch().appending(path: "adapter")
        var request = LeJEPATrainRequest(datasetPath: "-", iterations: 20)
        (request.rank, request.layers, request.batchSize, request.maxSeqLen, request.learningRate, request.warmupRatio) = (4, [1, 3], 4, 16, 3e-3, 0.1)
        let dataset = try GoodViewDataset.parse(jsonl: Self.scenes)
        let before = await Self.weights(container)
        let report = try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
            try MLXInferenceEngine.trainJEPA(request, dataset: dataset, directory: directory, context: context)
        }
        #expect(report.losses.count == 20 && report.gradNorms.count == 20 && report.invarianceLosses.count == 20)
        let early = report.losses.prefix(5).reduce(0, +), late = report.losses.suffix(5).reduce(0, +)
        #expect(late < early, "\(report.losses)")
        #expect(await Self.weights(container) == before)
        let rankBefore = try #require(report.embeddingRankBefore), rankAfter = try #require(report.embeddingRankAfter)
        #expect(rankAfter > 1 && rankAfter * 2 >= rankBefore, "rank \(rankBefore) -> \(rankAfter)")
        #expect(report.adapterDirectory == directory.path)
        #expect(try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int == 0o700)
        let adapter = try LoRAContainer.from(directory: directory)
        let keys = adapter.parameters.flattened().map(\.0)
        #expect(!keys.isEmpty && keys.allSatisfy { $0.hasPrefix("model.layers.1.") || $0.hasPrefix("model.layers.3.") }, "\(keys)")
        #expect(adapter.parameters.flattened().contains { MLX.abs($0.1).max().item(Float.self) > 0 && $0.0.hasSuffix("lora_b") })
        try adapter.load(into: try TinyModel.make(.qwen3))
        let projector = try MLX.loadArrays(url: URL(fileURLWithPath: try #require(report.projectorPath)))
        #expect(projector["fc4.weight"]?.shape == [MLXInferenceEngine.jepaProjection, MLXInferenceEngine.jepaHidden])
    }

    @Test("A gradient spike is clipped to the max norm before the update; the reported norm is the norm before clipping")
    func clipping() {
        let spike = ModuleParameters.unflattened(["a": MLXArray([Float(300), -400]), "b": MLXArray([Float(1200)])])
        let (clipped, norm) = clipGradNorm(gradients: spike, maxNorm: 1)
        let after = sqrt(clipped.flattened().map { ($0.1 * $0.1).sum() }.reduce(MLXArray(Float(0)), +)).item(Float.self)
        #expect(norm.item(Float.self) == 1300)
        #expect(after <= 1 + 1e-4)
    }

    @Test("Warmup rises linearly to the peak, then cosine decays toward zero")
    func schedule() {
        var request = LeJEPATrainRequest(datasetPath: "-", iterations: 100)
        (request.learningRate, request.warmupRatio) = (1e-3, 0.1)
        let schedule = MLXInferenceEngine.jepaSchedule(request)
        #expect(schedule(0) == 0)
        #expect(abs(schedule(5) - 5e-4) < 1e-7)
        #expect(abs(schedule(10) - 1e-3) < 1e-7)
        #expect(schedule(99) < 1e-5)
    }

    @Test("Batches cycle through the scenes and never repeat a scene inside one batch")
    func batches() {
        let scenes = (0 ..< 5).map { GoodViewScene(views: ["\($0)a", "\($0)b"]) }
        #expect(MLXInferenceEngine.jepaScenes(scenes, batchSize: 2, step: 0).map(\.views[0]) == ["0a", "1a"])
        #expect(MLXInferenceEngine.jepaScenes(scenes, batchSize: 2, step: 2).map(\.views[0]) == ["4a", "0a"])
        #expect(MLXInferenceEngine.jepaScenes(scenes, batchSize: 9, step: 3).count == 5)
    }

    @Test("A hybrid model without an attention-only encoder path is refused before LoRA is attached, and the model is unchanged")
    func hybridRefused() async throws {
        let container = try TinyModel.container(.qwen35)
        let before = await Self.weights(container)
        let dataset = try GoodViewDataset.parse(jsonl: Self.scenes)
        let directory = try Self.scratch().appending(path: "never")
        var request = LeJEPATrainRequest(datasetPath: "-", iterations: 1)
        request.batchSize = 2
        let error = await #expect(throws: BasaltError.self) {
            try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
                try MLXInferenceEngine.trainJEPA(request, dataset: dataset, directory: directory, context: context)
            }
        }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("unexpected \(String(describing: error))"); return }
        #expect(details.contains("attention-only"), "\(details)")
        #expect(await Self.weights(container) == before)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Out-of-range settings, a missing dataset and an existing output are refused before anything runs")
    func refused() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let data = try Self.scratch().appending(path: "scenes.jsonl")
        try Data(Self.scenes.utf8).write(to: data)
        func request(_ change: (inout LeJEPATrainRequest) -> Void) -> LeJEPATrainRequest {
            var request = LeJEPATrainRequest(datasetPath: data.path, iterations: 1)
            change(&request)
            return request
        }
        let existing = try Self.scratch().path
        let cases: [(LeJEPATrainRequest, String)] = [
            (request { $0.rank = 0 }, "rank"), (request { $0.iterations = 0 }, "iterations"), (request { $0.batchSize = 5000 }, "batchSize"),
            (request { $0.maxSeqLen = 2 }, "maxSeqLen"), (request { $0.learningRate = .nan }, "learningRate"), (request { $0.warmupRatio = 2 }, "warmupRatio"),
            (request { $0.maxGradNorm = 0 }, "maxGradNorm"), (request { $0.datasetPath = "/nonexistent/x.jsonl" }, "could not read"),
            (request { $0.output = existing }, "already exists"),
        ]
        for (request, fragment) in cases {
            let error = await #expect(throws: BasaltError.self) { try await engine.trainJEPA(request) }
            guard case .invalidRequestStructure(let details)? = error else { Issue.record("unexpected \(String(describing: error))"); continue }
            #expect(details.contains(fragment), "\(details)")
        }
    }
}
