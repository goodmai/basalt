import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Self-training: checkable tasks, verified answers only, a LoRA saved away from the weights (#332)", .serialized)
struct LabTrainTests {
    static func scratch() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "lab-train-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func request(output: URL? = nil) -> TrainRequest {
        var request = TrainRequest(tasks: .arithmetic(count: 5, seed: 3), gate: GateRequest(heldOut: ["Name a colour."]))
        request.output = output?.path
        return request
    }

    static func parameterNames(_ container: ModelContainer) async -> Set<String> {
        await container.perform { context in Set(context.model.parameters().flattened().map(\.0)) }
    }

    @Test("Arithmetic tasks repeat for a seed, change with it, use all three operations and carry the right answers")
    func arithmetic() throws {
        let tasks = TrainingTask.arithmetic(count: 60, seed: 7)
        #expect(tasks == TrainingTask.arithmetic(count: 60, seed: 7))
        #expect(tasks != TrainingTask.arithmetic(count: 60, seed: 8))
        #expect([" + ", " - ", " * "].allSatisfy { symbol in tasks.contains { $0.prompt.contains(symbol) } })
        for task in tasks {
            let match = try #require(task.prompt.firstMatch(of: /What is (\d+) ([-+*]) (\d+)\?/))
            let a = try #require(Int(match.output.1))
            let b = try #require(Int(match.output.3))
            let value = switch match.output.2 {
            case "+": a + b
            case "-": a - b
            default: a * b
            }
            #expect(task.answer == String(value), "\(task.prompt)")
            #expect(task.accepts(task.answer))
        }
    }

    @Test("An answer is kept only if its final number, or its final line, matches the known answer", arguments: [
        ("42", "The answer is 42.", true),
        ("42", "42", true),
        ("-7", "5 - 12 = -7", true),
        ("1042", "It is 1,042.", true),
        ("Paris", "Answer: Paris.", true),
        ("Paris", "paris", true),
        ("42", "41", false),
        ("42", "42 or 43", false),
        ("42", "-42", false),
        ("42", "forty-two", false),
        ("42", "", false),
        ("42", "99999999999999999999999", false),
        ("-7", "7", false),
        ("Paris", "London", false),
        ("Paris", "", false),
        ("Paris", "Paris, I think\nLondon", false),
    ])
    func verification(answer: String, output: String, accepted: Bool) {
        #expect(TrainingTask(prompt: "q", answer: answer).accepts(output) == accepted)
    }

    @Test("A JSONL task file loads; a malformed line, an empty answer or a single task is refused")
    func taskFile() throws {
        let directory = try Self.scratch()
        func write(_ name: String, _ text: String) throws -> String {
            let url = directory.appending(path: name)
            try Data(text.utf8).write(to: url)
            return url.path
        }
        let good = try write("good.jsonl", #"{"prompt": "Capital of France?", "answer": "Paris"}"# + "\n\n" + #"{"prompt": "2 + 2?", "answer": "4"}"# + "\n")
        #expect(try TrainingTask.load(.file(good)) == [TrainingTask(prompt: "Capital of France?", answer: "Paris"), TrainingTask(prompt: "2 + 2?", answer: "4")])
        for bad in [#"{"prompt": "a", "answer": "1"}"# + "\nnot json", #"{"prompt": "a", "answer": ""}"# + "\n" + #"{"prompt": "b", "answer": "2"}"#, #"{"prompt": "a", "answer": "1"}"#] {
            let path = try write("bad-\(UUID().uuidString).jsonl", bad)
            #expect(throws: BasaltError.self) { try TrainingTask.load(.file(path)) }
        }
        #expect(throws: BasaltError.self) { try TrainingTask.load(.file(directory.appending(path: "missing.jsonl").path)) }
    }

    @Test("A few LoRA iterations on a tiny model lower the loss, write the adapter in the mlx-lm format (0700), leave the model as it was, and load back")
    func trains() async throws {
        let container = try TinyModel.container(.qwen3)
        let directory = try Self.scratch().appending(path: "adapter")
        var request = Self.request(output: directory)
        (request.layers, request.rank, request.iterations, request.learningRate) = ([1, 3], 4, 40, 5e-3)
        let pairs = TrainingTask.arithmetic(count: 4, seed: 1)
        let before = await Self.parameterNames(container)
        let run = try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
            try MLXInferenceEngine.trainAdapter(pairs, heldOut: Array(pairs.prefix(1)), request: request, directory: directory, context: context)
        }
        #expect(run.losses.count == 20)
        let first = try #require(run.losses.first)
        let last = try #require(run.losses.last)
        #expect(last < first, "\(run.losses)")
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "adapter_config.json").path))
        #expect(FileManager.default.fileExists(atPath: directory.appending(path: "adapters.safetensors").path))
        #expect(try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int == 0o700)
        #expect(await Self.parameterNames(container) == before)
        let adapter = try LoRAContainer.from(directory: directory)
        #expect(adapter.configuration.numLayers == 3 && adapter.configuration.loraParameters.rank == 4)
        let keys = adapter.parameters.flattened().map(\.0)
        #expect(!keys.isEmpty && keys.allSatisfy { $0.hasPrefix("model.layers.1.") || $0.hasPrefix("model.layers.3.") }, "\(keys)")
        try adapter.load(into: try TinyModel.make(.qwen3))
    }

    @Test("When no answer verifies nothing is trained or written: the report says so and the model is untouched")
    func nothingVerified() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await Self.parameterNames(container)
        let directory = try Self.scratch().appending(path: "never")
        let report = try await engine.train(Self.request(output: directory))
        #expect(report.collected == 4 && report.verified == 0 && report.losses.isEmpty)
        #expect(report.accuracyBefore == 0 && report.accuracyAfter == 0)
        #expect(report.adapterDirectory == nil && report.verdict.overlay == nil)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
        #expect(await Self.parameterNames(container) == before)
    }

    @Test("An output inside the model directory, per-expert LoRA and out-of-range settings are refused before anything runs; the default goes under ~/.basalt/adapters")
    func refused() async throws {
        let model = try Self.scratch()
        let engine = MLXInferenceEngine()
        await engine.attach(try TinyModel.container(.qwen3), modelDescription: model.path)
        let inside = Self.request(output: model.appending(path: "adapters/run"))
        var experts = Self.request()
        experts.experts = [0]
        var rank = Self.request()
        rank.rank = 0
        var rate = Self.request()
        rate.learningRate = .nan
        for (request, fragment) in [(inside, "next to the source weights"), (experts, "per-expert LoRA is not available"), (rank, "rank"), (rate, "learningRate")] {
            let error = await #expect(throws: BasaltError.self) { try await engine.train(request) }
            guard case .invalidRequestStructure(let details)? = error else { Issue.record("unexpected \(String(describing: error))"); continue }
            #expect(details.contains(fragment), "\(details)")
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: model.path).isEmpty)
        let fallback = try await engine.adapterDirectory(nil)
        #expect(fallback.path.hasPrefix(FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/adapters").path))
        #expect(!MLXInferenceEngine.isInside(fallback, model))
    }
}
