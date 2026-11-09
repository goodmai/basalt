import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

struct DistillSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "distill",
        abstract: "Distill a teacher behind an API into a LoRA for the local model; set BASALT_TEACHER_URL and BASALT_TEACHER_API_KEY for the key"
    )

    static let generalPrompts = ["Explain what a hash table is in two sentences.", "Write a haiku about autumn.", "What is the capital of Australia?", "Summarize why the sky looks blue."]

    @Argument(help: "Model repo id from the local cache, or a local model directory")
    var model: String

    @Option(name: .long, help: "JSONL file, one {\"prompt\": …, \"answer\"?: …} per line")
    var prompts: String

    @Option(name: .long, help: "Teacher base URL, e.g. https://openrouter.ai/api/v1 or https://api.anthropic.com/v1")
    var teacherUrl: String

    @Option(name: .long, help: "Teacher model name")
    var teacherModel: String

    @Option(name: .long, help: "Teacher response format: openai or anthropic")
    var provider: String = TeacherProvider.openai.rawValue

    @Option(name: .long, help: "Teacher reasoning effort, e.g. low (reasoning tokens are billed)")
    var reasoningEffort: String?

    @Option(name: .long, help: "Stop calling the teacher before this spend in USD; needs --output-price")
    var budgetUsd: Double?

    @Option(name: .long, help: "Teacher USD per million prompt tokens")
    var inputPrice: Double?

    @Option(name: .long, help: "Teacher USD per million completion tokens")
    var outputPrice: Double?

    @Option(name: .long, help: "verify keeps only answers matching the prompt's \"answer\"")
    var filter: String = DistillFilter.none.rawValue

    @Flag(name: .long, help: "Train on the teacher's reasoning followed by its answer")
    var includeReasoning = false

    @Option(name: .long, help: "Training iterations")
    var iterations: Int = 200

    @Option(name: .long, help: "LoRA rank")
    var rank: Int = 16

    @Option(name: .long, help: "New adapter directory (default ~/.basalt/adapters)")
    var output: String?

    mutating func run() async throws {
        guard let modelPath = FileManager.default.fileExists(atPath: model) ? model : ModelCache.resolve(repoId: model)?.path else {
            throw ValidationError("Model '\(model)' is not in the local cache; run `basalt models download \(model)` first")
        }
        guard let provider = TeacherProvider(rawValue: provider), let filter = DistillFilter(rawValue: filter) else {
            throw ValidationError("--provider takes openai or anthropic, --filter none or verify")
        }
        var teacher = TeacherSpec(url: teacherUrl, model: teacherModel)
        (teacher.provider, teacher.inputPricePerMillion, teacher.outputPricePerMillion, teacher.reasoningEffort) = (provider, inputPrice, outputPrice, reasoningEffort)
        var request = DistillRequest(prompts: prompts, teacher: teacher, gate: GateRequest(heldOut: Self.generalPrompts))
        (request.filter, request.budgetUSD, request.iterations, request.rank, request.output, request.includeReasoning) = (filter, budgetUsd, iterations, rank, output, includeReasoning)
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine)
        defer { await orchestrator.shutdown() }
        try await orchestrator.loadModel(path: modelPath)
        let report = try await engine.distill(request)
        print("teacher: \(report.teacherCalls) calls (\(report.cached) cached), \(report.teacherOutputTokens) completion tokens, $\(String(format: "%.4f", report.spentUSD)), stopped by \(report.stoppedBy.rawValue)")
        print("kept \(report.kept), held out \(report.heldOut), rejected \(report.rejected.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))")
        print("held-out teacher NLL: \(report.teacherNLLBefore.map { String(format: "%.3f", $0) } ?? "-") -> \(report.teacherNLLAfter.map { String(format: "%.3f", $0) } ?? "-")")
        print("adapter: \(report.adapterDirectory ?? "-"), gate: \(report.verdict?.overlay.map { "applied as overlay \($0.id)" } ?? "not applied")")
    }
}
