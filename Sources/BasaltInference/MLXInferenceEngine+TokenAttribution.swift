import BasaltCore
import Foundation
import MLX
import MLXLMCommon

struct AttributedStepCapture: Sendable {
    let tokenId: Int
    let logprob: Float
    let entropy: Float
    let margin: Float
    let topIds: [Int]
    let topLogprobs: [Float]
}

enum TokenAttributionAssembler {
    static func assemble(
        captures: [AttributedStepCapture],
        promptCount: Int,
        request: AttributedTokenTraceRequest,
        totalLayers: Int,
        table: TokenTraceTable?,
        context: ModelContext
    ) -> [OutputTokenTelemetry] {
        let tracedLayers = table?.layers ?? []
        return captures.enumerated().map { stepIdx, cap in
            let tokStr = context.tokenizer.decode(tokenIds: [cap.tokenId], skipSpecialTokens: false)
            var item = OutputTokenTelemetry(
                index: stepIdx, token: tokStr, tokenId: cap.tokenId,
                logprob: cap.logprob, entropy: cap.entropy, margin: cap.margin
            )
            item.topLogprobs = zip(cap.topIds, cap.topLogprobs).map { tid, lp in
                TokenCandidateProbability(
                    token: context.tokenizer.decode(tokenIds: [tid], skipSpecialTokens: false),
                    tokenId: tid, logprob: lp
                )
            }
            let pos = promptCount - 1 + stepIdx
            if let table, pos >= 0, pos < table.tokens {
                let rawPathways = tracedLayers.map { layer in
                    buildLayerPathway(layer: layer, pos: pos, topHeads: request.topHeads, table: table, context: context)
                }
                let pathways = HeadInfluenceCore.enrich(
                    pathways: rawPathways, sampledId: cap.tokenId, finalProbability: item.probability
                )
                item.layers = pathways
                item.onsetLayer = resolveOnsetLayer(pathways: pathways, sampledId: cap.tokenId)
                let rescored = TokenHallucinationScorer.evaluate(
                    probability: item.probability, entropy: item.entropy, margin: item.margin,
                    onsetLayer: item.onsetLayer, totalLayers: totalLayers
                )
                item.hallucinationRisk = rescored.risk
                item.riskState = rescored.state
            }
            return item
        }
    }

    private static func buildLayerPathway(
        layer: Int, pos: Int, topHeads: Int, table: TokenTraceTable, context: ModelContext
    ) -> TokenLayerPathway {
        let headNorms = table.floatRow(.heads, layer: layer, token: pos) ?? []
        let attnEnt = table.floatRow(.attentionEntropy, layer: layer, token: pos)
        let sortedHeads = headNorms.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(topHeads)
            .map { hIdx, norm in
                let ent = (attnEnt != nil && hIdx < (attnEnt?.count ?? 0)) ? attnEnt?[hIdx] : nil
                return TokenHeadActivation(head: hIdx, norm: norm, attentionEntropy: ent)
            }
        let expIds = table.row(.experts, layer: layer, token: pos) ?? []
        let expGates = table.floatRow(.gates, layer: layer, token: pos) ?? []
        let experts = zip(expIds, expGates)
            .map { TokenExpertActivation(expert: Int($0), weight: $1) }
            .sorted { $0.weight > $1.weight }
        var pathway = TokenLayerPathway(layer: layer, topHeads: sortedHeads, experts: experts)
        pathway.routerEntropy = table.floatRow(.routerEntropy, layer: layer, token: pos)?.first
        pathway.routerMargin = table.floatRow(.routerMargin, layer: layer, token: pos)?.first
        if let lensIds = table.row(.lens, layer: layer, token: pos),
           let lensProbs = table.floatRow(.lensProbabilities, layer: layer, token: pos),
           let topTid = lensIds.first, let topProb = lensProbs.first {
            let tid = Int(topTid)
            pathway.lensTopTokenId = tid
            pathway.lensTopToken = context.tokenizer.decode(tokenIds: [tid], skipSpecialTokens: false)
            pathway.lensTopProbability = topProb
        }
        return pathway
    }

    private static func resolveOnsetLayer(pathways: [TokenLayerPathway], sampledId: Int) -> Int? {
        let withLens = pathways.filter { $0.lensTopTokenId != nil }
        guard !withLens.isEmpty else { return nil }
        for idx in withLens.indices where withLens[idx...].allSatisfy({ $0.lensTopTokenId == sampledId }) {
            return withLens[idx].layer
        }
        return withLens.first(where: { $0.lensTopTokenId == sampledId })?.layer
    }
}

extension MLXInferenceEngine {
    public func attributeTokens(_ request: AttributedTokenTraceRequest) async throws(BasaltError) -> AttributedTokenTraceReport {
        guard let container else {
            throw .inferenceHardwareFailure(reason: "MLX engine not initialized — call load() first")
        }
        let epoch = modelEpoch
        let anatomy = try await self.anatomy()
        guard modelEpoch == epoch else {
            throw .inferenceError("the model changed while token attribution was prepared; retry")
        }
        try request.validate(against: anatomy)
        let templatedIds: [Int]?
        if let messages = request.messages, !messages.isEmpty {
            let genReq = GenerationRequest(prompt: ChatMessageInput.plainText(messages), messages: messages)
            do {
                let rendered = try await render(genReq, tools: nil, images: [:], container: container)
                templatedIds = rendered.input.text.tokens.asArray(Int.self)
            } catch {
                templatedIds = nil
            }
        } else {
            templatedIds = nil
        }
        let interventions = activeInterventions()
        let modelName = modelDescription
        return try await Self.labRun(container, "token attribution") { context in
            try Self.runTokenAttribution(
                request,
                templatedIds: templatedIds,
                modelName: modelName,
                anatomy: anatomy,
                interventions: interventions,
                context: context
            )
        }
    }

    private static func runTokenAttribution(
        _ request: AttributedTokenTraceRequest,
        templatedIds: [Int]?,
        modelName: String,
        anatomy: ModelAnatomy,
        interventions: [Intervention],
        context: ModelContext
    ) throws -> AttributedTokenTraceReport {
        let resolvedText = (request.messages?.isEmpty == false)
            ? ChatMessageInput.plainText(request.messages ?? []) : request.prompt
        let promptIds: [Int]
        if let templatedIds, !templatedIds.isEmpty {
            promptIds = templatedIds
        } else {
            promptIds = ModelProbe.encoded(resolvedText, context.tokenizer)
        }
        guard !promptIds.isEmpty else {
            throw BasaltError.invalidRequestStructure(details: "xray tokens: prompt encodes to no tokens")
        }
        let total = promptIds.count + request.maxTokens
        guard total <= TokenTraceRequest.tokenLimit else {
            throw BasaltError.contextLengthExceeded(maxTokens: TokenTraceRequest.tokenLimit, requested: total)
        }
        let recorder = makeAttributionRecorder(
            text: resolvedText, request: request, anatomy: anatomy, interventions: interventions, context: context
        )
        let captures = try sampleAttributedSteps(
            promptIds: promptIds, request: request, recorder: recorder, context: context
        )
        let table = extractAttributionTable(from: recorder)
        let stops = stopTokenIds(context)
        let visibleIds = captures.map(\.tokenId).filter { !stops.contains($0) }
        let fullText = context.tokenizer.decode(tokenIds: visibleIds, skipSpecialTokens: true)
        let tokens = TokenAttributionAssembler.assemble(
            captures: captures, promptCount: promptIds.count, request: request,
            totalLayers: anatomy.layers.count, table: table, context: context
        )
        return AttributedTokenTraceReport(
            model: modelName, promptTokens: promptIds.count, generatedText: fullText, tokens: tokens
        )
    }

    private static func makeAttributionRecorder(
        text: String, request: AttributedTokenTraceRequest,
        anatomy: ModelAnatomy, interventions: [Intervention], context: ModelContext
    ) -> TokenTraceRecorder {
        let lens = request.includeLogitLens ? readout(of: context.model) : nil
        var traceReq = TokenTraceRequest(prompt: text, maxTokens: request.maxTokens)
        traceReq.topK = request.topHeads
        traceReq.heads = true
        traceReq.experts = true
        traceReq.attentionEntropy = request.includeAttentionEntropy
        traceReq.lensTopK = (lens != nil) ? max(1, min(20, max(1, request.topLogprobs))) : 0
        traceReq.layers = request.layers
        let edits = interventions.isEmpty ? nil : InterventionProbe(interventions, anatomy: anatomy)
        return TokenTraceRecorder(traceReq, readout: lens, edits: edits)
    }

    private static func extractAttributionTable(from recorder: TokenTraceRecorder) -> TokenTraceTable? {
        guard let layout = recorder.layout, layout.floatWidth + layout.intWidth > 0,
              let (floats, ints) = recorder.table() else { return nil }
        return TokenTraceTable(
            layout: layout, tokens: floats.dim(0),
            floats: floats.asArray(Float.self), ints: ints.asArray(Int32.self)
        )
    }

    private static func sampleAttributedSteps(
        promptIds: [Int], request: AttributedTokenTraceRequest,
        recorder: TokenTraceRecorder, context: ModelContext
    ) throws -> [AttributedStepCapture] {
        let cache = try context.model.newCache(parameters: nil)
        let stops = stopTokenIds(context)
        var input = promptIds
        var state: LMOutput.State?
        var captures: [AttributedStepCapture] = []
        while captures.count < request.maxTokens {
            let logits = forward(context, probe: recorder, input, cache: cache, state: &state)
            let rows = try recorder.collect(positions: input.count)
            let row = logits[0, -1].asType(.float32)
            let stats = FusedLogitConfidence.statistics(logits: row)
            let logprobs = row - logSumExp(row)
            let nextArr = request.temperature <= 0 ? argMax(row, axis: -1) : MLXRandom.categorical(row / request.temperature)
            let topCount = min(max(0, request.topLogprobs), row.dim(0))
            let topIdsArr = topCount > 0 ? argSort(-logprobs)[..<topCount] : MLXArray.zeros([0], type: Int32.self)
            eval(rows + [nextArr, topIdsArr, logprobs])
            let sampledId = nextArr.item(Int.self)
            captures.append(AttributedStepCapture(
                tokenId: sampledId, logprob: logprobs[sampledId].item(Float.self),
                entropy: Float(stats?.entropy ?? 0.0), margin: Float(stats?.margin ?? 0.0),
                topIds: topCount > 0 ? topIdsArr.asArray(Int.self) : [],
                topLogprobs: topCount > 0 ? logprobs[topIdsArr].asArray(Float.self) : []
            ))
            if stops.contains(sampledId) { break }
            input = [sampledId]
            try Task.checkCancellation()
        }
        return captures
    }
}
