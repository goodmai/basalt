import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
import BasaltCore

struct DistillRow: Sendable, Equatable {
    var tokens: [Int]
    var promptCount: Int
}

struct DistillRun: Sendable {
    var losses: [Float]
    var nllBefore: Float
    var nllAfter: Float
    var tooLong: Int
}

extension MLXInferenceEngine {
    static let distillSamples = 1...8
    static let distillConcurrency = 1...32
    static let distillTeacherTokens = 16...32_768

    public func distill(_ request: DistillRequest) async throws(BasaltError) -> DistillReport {
        try await distill(request, session: .shared, home: FileManager.default.homeDirectoryForCurrentUser)
    }

    func distill(_ request: DistillRequest, session: URLSession, home: URL, environment: [String: String] = ProcessInfo.processInfo.environment) async throws(BasaltError) -> DistillReport {
        guard let container else { throw .modelNotLoaded }
        try Self.validate(request)
        let data: Data
        do { data = try Data(contentsOf: URL(fileURLWithPath: (request.prompts as NSString).expandingTildeInPath)) } catch {
            throw .invalidRequestStructure(details: "distill: could not read the prompts \(request.prompts): \(error.localizedDescription)")
        }
        let examples = try DistillCore.examples(jsonl: String(decoding: data, as: UTF8.self))
        let directory = try adapterDirectory(request.output)
        let teacher = try TeacherClient(request.teacher.url, model: request.teacher.model, provider: request.teacher.provider, environment: environment, session: session)
        let cache = DistillCache(directory: DistillCache.directory(for: request, prompts: data, home: home))
        var report = try await Self.collect(request, examples: examples, teacher: teacher, cache: cache)
        let (train, heldOut) = Self.pairs(request, examples: examples, entries: cache.load(), report: &report)
        guard !train.isEmpty, !heldOut.isEmpty else { return report }
        guard self.container === container else { throw .inferenceError("the model changed while the teacher answered; nothing was trained") }
        let run = try await Self.labRun(container, "distillation") { context in
            try Self.distillAdapter(train, heldOut: heldOut, request: request, directory: directory, context: context)
        }
        guard self.container === container else { throw .inferenceError("the model changed during distillation; the adapter in \(directory.path) was not applied") }
        (report.losses, report.teacherNLLBefore, report.teacherNLLAfter) = (run.losses, run.nllBefore, run.nllAfter)
        report.rejected["too_long", default: 0] += run.tooLong
        report.adapterDirectory = directory.path
        do {
            report.verdict = try await applyAdapter(.directory(directory.path), gate: request.gate)
        } catch {
            throw .inferenceError("the adapter was distilled and saved in \(directory.path) but not applied: \(Self.briefly(error))")
        }
        return report
    }

    static func validate(_ request: DistillRequest) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "distill: \(reason)") }
        try validate(request.teacher)
        if let budget = request.budgetUSD {
            guard budget.isFinite, budget > 0 else { throw refused("budgetUSD must be positive") }
            guard request.teacher.outputPricePerMillion != nil else { throw refused("budgetUSD needs teacher.outputPricePerMillion (and inputPricePerMillion) to count spend") }
        }
        guard distillSamples.contains(request.samplesPerPrompt) else { throw refused("samplesPerPrompt must be in \(distillSamples)") }
        guard request.heldOutFraction.isFinite, request.heldOutFraction > 0, request.heldOutFraction < 1 else { throw refused("heldOutFraction must be in (0, 1)") }
        guard trainingRanks.contains(request.rank) else { throw refused("rank must be in \(trainingRanks)") }
        guard trainingIterations.contains(request.iterations) else { throw refused("iterations must be in \(trainingIterations)") }
        guard (1 ... 64).contains(request.batchSize) else { throw refused("batchSize must be in 1...64") }
        guard (16 ... 32_768).contains(request.maxSeqLen) else { throw refused("maxSeqLen must be in 16...32768") }
        guard request.learningRate.isFinite, request.learningRate > 0, request.learningRate <= 1 else { throw refused("learningRate must be in (0, 1]") }
    }

    static func validate(_ teacher: TeacherSpec) throws(BasaltError) {
        func refused(_ reason: String) -> BasaltError { .invalidRequestStructure(details: "distill: \(reason)") }
        guard !teacher.model.isEmpty else { throw refused("teacher.model is required") }
        guard distillTeacherTokens.contains(teacher.maxTokens) else { throw refused("teacher.maxTokens must be in \(distillTeacherTokens)") }
        guard distillConcurrency.contains(teacher.concurrency) else { throw refused("teacher.concurrency must be in \(distillConcurrency)") }
        guard teacher.temperature.isFinite, (0 ... 2).contains(teacher.temperature) else { throw refused("teacher.temperature must be in [0, 2]") }
        guard [teacher.inputPricePerMillion, teacher.outputPricePerMillion].allSatisfy({ $0.map { $0.isFinite && $0 >= 0 } ?? true }) else { throw refused("prices must be finite and non-negative") }
    }

    static func collect(_ request: DistillRequest, examples: [DistillExample], teacher: TeacherClient, cache: DistillCache) async throws(BasaltError) -> DistillReport {
        var report = DistillReport()
        let cached = cache.load()
        (report.prompts, report.cached, report.cacheDirectory) = (examples.count, cached.count, cache.directory.path)
        var pending = DistillCore.missing(cached, prompts: examples.count, samples: request.samplesPerPrompt)[...]
        while !pending.isEmpty {
            let wave = Array(pending.prefix(request.teacher.concurrency))
            if let budget = request.budgetUSD, report.spentUSD + DistillCore.worstCase(wave.count, teacher: request.teacher) > budget {
                report.stoppedBy = .budget
                break
            }
            let entries = try await answers(wave, examples: examples, teacher: teacher, spec: request.teacher)
            do { try cache.append(entries) } catch { throw .inferenceError("distill: could not write the teacher cache \(cache.file.path): \(error.localizedDescription)") }
            for entry in entries {
                report.teacherCalls += 1
                report.teacherInputTokens += entry.completion.inputTokens
                report.teacherOutputTokens += entry.completion.outputTokens
                report.spentUSD += DistillCore.cost(entry.completion, teacher: request.teacher)
            }
            pending = pending.dropFirst(wave.count)
        }
        return report
    }

    static func answers(_ wave: [(index: Int, sample: Int)], examples: [DistillExample], teacher: TeacherClient, spec: TeacherSpec) async throws(BasaltError) -> [DistillEntry] {
        do {
            return try await withThrowingTaskGroup(of: DistillEntry.self) { group in
                for (index, sample) in wave {
                    group.addTask {
                        DistillEntry(index: index, sample: sample, completion: try await teacher.complete(examples[index].prompt, maxTokens: spec.maxTokens, temperature: spec.temperature, reasoningEffort: spec.reasoningEffort))
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }.sorted { ($0.index, $0.sample) < ($1.index, $1.sample) }
            }
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("distill: \(briefly(error))")
        }
    }

    static func pairs(_ request: DistillRequest, examples: [DistillExample], entries: [DistillEntry], report: inout DistillReport) -> (train: [TrainingTask], heldOut: [TrainingTask]) {
        let heldOutStart = examples.count - DistillCore.heldOutCount(examples.count, fraction: request.heldOutFraction)
        var seen = Set<String>()
        var (train, heldOut) = ([TrainingTask](), [TrainingTask]())
        for entry in entries where entry.index < examples.count && entry.sample < request.samplesPerPrompt {
            let example = examples[entry.index]
            if let reason = DistillCore.rejection(entry.completion, example: example, filter: request.filter, seen: &seen) {
                report.rejected[reason, default: 0] += 1
                continue
            }
            let pair = TrainingTask(prompt: example.prompt, answer: entry.completion.text.trimmingCharacters(in: .whitespacesAndNewlines))
            if entry.index >= heldOutStart { heldOut.append(pair) } else { train.append(pair) }
        }
        (report.kept, report.heldOut) = (train.count, heldOut.count)
        return (train, heldOut)
    }
}
