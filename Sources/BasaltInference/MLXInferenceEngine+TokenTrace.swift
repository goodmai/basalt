import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension MLXInferenceEngine {
    struct TokenTraceRun: Sendable {
        let table: TokenTraceTable
        let ids: [Int]
        let texts: [String]
        let promptTokens: Int
        let text: String
        let floats: Data
    }

    public func traceTokens(_ request: TokenTraceRequest) async throws(BasaltError) -> TokenTraceReport {
        guard let container else { throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first") }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else { throw .inferenceError("the model changed while the trace was prepared; retry") }
        try request.validate(against: anatomy)
        let run = try await Self.labRun(container, "the token trace") { context in try Self.tokenTrace(request, anatomy: anatomy, context: context) }
        let url: URL
        let data: Data
        do {
            data = try TokenTraceFile.encode(
                model: modelDescription, table: run.table, ids: run.ids, texts: run.texts, promptTokens: run.promptTokens, floats: run.floats)
            guard data.count <= request.maxFileBytes else {
                throw BasaltError.invalidRequestStructure(details: "the trace is \(data.count) bytes, over maxFileBytes \(request.maxFileBytes); trace fewer layers or tokens")
            }
            url = try TokenTraceFile.write(data, into: request.directory)
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .inferenceError("the trace file was not written: \(Self.briefly(error))")
        }
        var report = TokenTraceReport(
            file: url.path, bytes: data.count, promptTokens: run.promptTokens, generated: Array(run.ids.dropFirst(run.promptTokens)), text: run.text)
        report.layers = TokenTraceSummary.layers(run.table, topK: request.topK)
        report.crossings = TokenTraceSummary.crossings(run.table, directions: request.projections)
        return report
    }

    static func tokenTrace(_ request: TokenTraceRequest, anatomy: ModelAnatomy, context: ModelContext) throws -> TokenTraceRun {
        let prompt = context.tokenizer.encode(text: request.prompt, addSpecialTokens: true)
        guard !prompt.isEmpty else { throw BasaltError.invalidRequestStructure(details: "the prompt encodes to no tokens") }
        let total = prompt.count + request.maxTokens
        guard total <= TokenTraceRequest.tokenLimit else { throw BasaltError.contextLengthExceeded(maxTokens: TokenTraceRequest.tokenLimit, requested: total) }
        let estimate = request.estimatedBytes(anatomy: anatomy, tokens: total)
        guard estimate <= request.maxFileBytes else {
            throw BasaltError.invalidRequestStructure(details: "the trace would take about \(estimate) bytes, over maxFileBytes \(request.maxFileBytes); trace fewer layers or tokens")
        }
        let lens = request.lensTopK > 0 ? readout(of: context.model) : nil
        guard request.lensTopK == 0 || lens != nil else {
            throw BasaltError.invalidRequestStructure(details: "this model has no readout for the logit lens; set lensTopK to 0")
        }
        let recorder = TokenTraceRecorder(request, readout: lens)
        let generated = try tracedGreedy(prompt, maxTokens: request.maxTokens, recorder: recorder, context: context)
        guard let layout = recorder.layout, layout.floatWidth + layout.intWidth > 0, let (floats, ints) = recorder.table() else {
            throw BasaltError.invalidRequestStructure(details: "this model's family reports none of the requested signals to the forward probe")
        }
        let ids = prompt + generated
        let table = TokenTraceTable(layout: layout, tokens: ids.count, floats: floats.asArray(Float.self), ints: ints.asArray(Int32.self))
        let stops = stopTokenIds(context)
        return TokenTraceRun(
            table: table, ids: ids, texts: ids.map { context.tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false) },
            promptTokens: prompt.count, text: context.tokenizer.decode(tokenIds: generated.filter { !stops.contains($0) }, skipSpecialTokens: true),
            floats: floats.asData(access: .copy).data)
    }

    static func tracedGreedy(_ prompt: [Int], maxTokens: Int, recorder: TokenTraceRecorder, context: ModelContext) throws -> [Int] {
        let cache = try context.model.newCache(parameters: nil)
        let stops = stopTokenIds(context)
        var input = prompt
        var generated: [Int] = []
        while true {
            let logits = forward(context, probe: recorder, input, cache: cache)
            let rows = try recorder.collect(positions: input.count)
            guard generated.count < maxTokens, !(generated.last.map(stops.contains) ?? false) else {
                eval(rows)
                return generated
            }
            let next = argMax(logits[0, -1], axis: -1)
            eval(rows + [next])
            generated.append(next.item(Int.self))
            input = [generated[generated.count - 1]]
            try Task.checkCancellation()
        }
    }
}
