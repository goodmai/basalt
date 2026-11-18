import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("GQA group causal path patching and head kurtosis diagnostics (#570)", .serialized)
struct GQAGroupPatchingTests {
    static let positive = ["aaaa aaaa a", "a aaa aa aaaa", "aaaaaaaaaa a"]
    static let negative = ["zzzz zzzz z", "z zzz zz zzzz", "zzzzzzzzzz z"]

    @Test("Excess and sample kurtosis drop sharply when importance disperses from a single head across GQA head groups")
    func kurtosisDispersionAndRecommendation() throws {
        let singleHeadMHA: [Float] = [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let gqaPairDispersion: [Float] = [6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let gqaQuadDispersion: [Float] = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let kSingle = ActivationStatistics.excessKurtosis(singleHeadMHA)
        let kPair = ActivationStatistics.excessKurtosis(gqaPairDispersion)
        let kQuad = ActivationStatistics.excessKurtosis(gqaQuadDispersion)
        #expect(kSingle > kPair)
        #expect(kPair > kQuad)
        #expect(ActivationStatistics.sampleExcessKurtosis(singleHeadMHA) > ActivationStatistics.sampleExcessKurtosis(gqaQuadDispersion))
        #expect(GQAGroupPatcher.recommendMode(heads: 16, kvHeads: 16, kurtosis: kQuad) == .singleHead)
        #expect(GQAGroupPatcher.recommendMode(heads: 16, kvHeads: 4, kurtosis: kQuad, kurtosisThreshold: 10) == .gqaGroup)
        #expect(GQAGroupPatcher.recommendMode(heads: 16, kvHeads: 4, kurtosis: 120, kurtosisThreshold: 50) == .singleHead)
        #expect(try GQAGroupPatcher.queryHeads(kvGroup: 0, heads: 8, kvHeads: 2) == 0 ..< 4)
        #expect(try GQAGroupPatcher.queryHeads(kvGroup: 1, heads: 8, kvHeads: 2) == 4 ..< 8)
        #expect(throws: BasaltError.self) { try GQAGroupPatcher.queryHeads(kvGroup: 2, heads: 8, kvHeads: 2) }
        #expect(throws: BasaltError.self) { try GQAGroupPatcher.queryHeads(kvGroup: 0, heads: 7, kvHeads: 2) }
        let imp = GQAGroupPatcher.importance(baseline: 4.0, patched: 1.0, epsilon: 1e-6)
        #expect(abs(imp - (-0.75)) < 1e-5)
    }

    @Test("ContrastSearch on a GQA TinyModel reports head kurtosis, ranked KV head groups and restores every attention projection")
    func endToEndGroupAttribution() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        var request = ContrastRequest(positive: Self.positive, negative: Self.negative, heldOut: ["the cat sat", "on the mat"])
        request.topHeads = 4
        request.scales = [2, 6]
        let report = try await engine.contrast(request)
        let kurtosis = try #require(report.headKurtosis)
        #expect(kurtosis.isFinite)
        #expect(!report.gqaGroups.isEmpty)
        #expect(report.recommendedIntervention == .gqaGroup)
        for group in report.gqaGroups {
            #expect([report.bestLayer - 1, report.bestLayer].contains(group.layer))
            #expect([0, 1].contains(group.kvGroup))
            #expect(group.queryHeads == (group.kvGroup == 0 ? [0, 1] : [2, 3]))
            #expect(group.importance.isFinite && group.effect.isFinite)
        }
        for (lhs, rhs) in zip(report.gqaGroups, report.gqaGroups.dropFirst()) {
            #expect(abs(lhs.importance) >= abs(rhs.importance) - 1e-6)
        }
        let types = await container.perform { context in
            context.model.namedModules().filter { $0.0.hasSuffix(".self_attn.o_proj") }.map { String(describing: type(of: $0.1)) }
        }
        #expect(types.count == 4 && types.allSatisfy { $0 == "Linear" })
    }

    @Test("ActivationContrastReport exposes finite head kurtosis across traced attention heads")
    func activationContrastKurtosis() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = ActivationContrastRequest(a: Self.positive, b: Self.negative, maxTokens: 2, layers: [1, 2])
        let report = try await engine.activationContrast(request)
        let kurtosis = try #require(report.headKurtosis)
        #expect(kurtosis.isFinite)
    }
}

@Suite("Cognitive nullspace projection preserves protected anchor subspaces and log-probabilities (#570)", .serialized)
struct CognitiveNullspaceTests {
    static let positive = ["aaaa aaaa a", "a aaa aa aaaa", "aaaaaaaaaa a"]
    static let negative = ["zzzz zzzz z", "z zzz zz zzzz", "zzzzzzzzzz z"]
    static let mathAnchors = ["12 + 34 = 46", "7 * 8 = 56", "def fib(n): return n"]

    @Test("Extracted orthonormal basis satisfies U^T U = I and projects vectors to < 1e-5 inner-product drift on anchors")
    func nullspaceAlgebraicInvariants() throws {
        let anchors: [[Float]] = [
            [1, 2, 0, -1, 0.5, 0, 0, 0],
            [2, 4, 1, 0, -0.5, 1, 0, 0],
            [0, -1, 3, 2, 0, 0, 1, -1],
        ]
        let coreNS = try CognitiveNullspace.from(anchors: anchors)
        #expect(coreNS.rank == 3)
        #expect(coreNS.orthogonalityDefect() < 1e-5)
        let mlxMatrix = MLXArray(anchors.flatMap { $0 }, [3, 8])
        let mlxNS = try CognitiveNullspaceExtractor.extract(activations: mlxMatrix)
        #expect(mlxNS.rank == 3)
        #expect(mlxNS.orthogonalityDefect() < 1e-5)
        let rawDirection: [Float] = [0.7, -1.2, 0.9, 0.4, -0.8, 1.1, -0.3, 0.6]
        let projected = try mlxNS.project(rawDirection)
        #expect(mlxNS.maxProjectionMagnitude(projected) < 1e-5)
        #expect(mlxNS.l2ProjectionMagnitude(projected) < 1e-5)
        for anchor in anchors {
            let dot = zip(anchor, projected).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            #expect(abs(dot) < 1e-5)
        }
        let reprojected = try mlxNS.project(projected)
        let idempotenceDrift = zip(projected, reprojected).map { abs($0 - $1) }.max() ?? 0
        #expect(idempotenceDrift < 1e-6)
    }

    @Test("Projected steer_commit moves positive prompts while keeping protected math/code anchor log-probs within < 1e-5 drift")
    func commitPreservesProtectedAnchorLogProbs() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        var search = ContrastRequest(positive: Self.positive, negative: Self.negative, heldOut: ["the cat sat"])
        search.protectedPrompts = Self.mathAnchors
        search.topHeads = 0
        search.scales = [4]
        let found = try await engine.contrast(search)
        #expect(found.nullspaceRank == Self.mathAnchors.count)
        #expect(found.rawDirection != nil)
        let anchorBefore = try await Self.nextTokenLogProbs(Self.mathAnchors, in: container)
        let positiveBefore = try await Self.nextTokenLogProbs(Self.positive, in: container)
        var commit = SteeringCommit(
            layer: found.bestLayer,
            direction: found.direction,
            scale: 6,
            gate: GateRequest(heldOut: Self.mathAnchors, maxKL: 100, maxPerplexityRise: 100)
        )
        commit.positive = Self.positive
        commit.protectedPrompts = Self.mathAnchors
        let verdict = try await engine.commitSteering(commit)
        let overlay = try #require(verdict.overlay)
        let anchorAfter = try await Self.nextTokenLogProbs(Self.mathAnchors, in: container)
        let positiveAfter = try await Self.nextTokenLogProbs(Self.positive, in: container)
        var maxAnchorDrift: Float = 0
        for (before, after) in zip(anchorBefore, anchorAfter) {
            let drift = zip(before, after).map { abs($0 - $1) }.max() ?? 0
            maxAnchorDrift = max(maxAnchorDrift, drift)
        }
        var maxPositiveShift: Float = 0
        for (before, after) in zip(positiveBefore, positiveAfter) {
            let shift = zip(before, after).map { abs($0 - $1) }.max() ?? 0
            maxPositiveShift = max(maxPositiveShift, shift)
        }
        #expect(maxAnchorDrift < 1e-5)
        #expect(maxPositiveShift > 1e-3)
        try await engine.undo(overlay.id)
    }

    private static func nextTokenLogProbs(_ prompts: [String], in container: ModelContainer) async throws -> [[Float]] {
        try await container.performEvaluating(nonSendable: prompts) { context, prompts in
            try prompts.map { prompt in
                let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
                let logits = MLXInferenceEngine.forward(context, probe: nil, ids, cache: try context.model.newCache(parameters: nil))
                let last = logits[0, ids.count - 1].asType(.float32)
                let logP = last - logSumExp(last, axis: -1)
                eval(logP)
                return logP.asArray(Float.self)
            }
        }
    }
}

@Suite("Anti-steering OLS regression, sign-reversal gate and context-gated steering (#570)", .serialized)
struct AntiSteeringAndContextGateTests {
    @Test("OLS steerability regression recovers slope and detects SYS and USER_NEG polarity reversals")
    func olsAndPolaritySignReversal() {
        let ols = AntiSteeringGuard.fitOLS([(0, 0.5), (2, 3.5), (4, 6.5), (8, 12.5)])
        #expect(abs(ols.slope - 1.5) < 1e-5)
        #expect(abs(ols.intercept - 0.5) < 1e-5)
        #expect(abs(ols.rSquared - 1.0) < 1e-5)
        let baseline: [Float] = [0.2, 0.1, 0.0, -0.1]
        let steered: [Float] = [2.4, -1.2, -0.8, 1.9]
        let kinds: [PolarityProbeKind] = [.userPos, .userNeg, .system, .userPos]
        let summary = AntiSteeringGuard.summarize(
            baseline: baseline,
            trials: [(scale: 2.0, steered: steered)],
            kinds: kinds,
            referenceSlope: 1.1
        )
        #expect(abs(summary.antiSteeringRate - 0.5) < 1e-5)
        #expect(abs((summary.byKind["user_pos"] ?? -1) - 0.0) < 1e-5)
        #expect(abs((summary.byKind["user_neg"] ?? -1) - 1.0) < 1e-5)
        #expect(abs((summary.byKind["sys"] ?? -1) - 1.0) < 1e-5)
        let allReversed = AntiSteeringGuard.summarize(
            baseline: [0.0, 0.0],
            trials: [(scale: 4.0, steered: [-2.0, -3.5])],
            kinds: [.userNeg, .system],
            referenceSlope: 1.5
        )
        #expect(abs(allReversed.antiSteeringRate - 1.0) < 1e-5)
    }

    @Test("Context-gated steering applies full shift inside the cone and zeroes out-of-cone prompts for additive and geodesic modes")
    func contextGateZeroesOutOfCone() {
        let gate = ContextSteerGate(contextVector: [1, 0, 0, 0], threshold: 0.5, sharpness: 30)
        #expect(abs(gate.effectiveScale(baseScale: 4, hidden: [2, 0, 0, 0], fallbackDirection: [0, 1, 0, 0]) - 4) < 1e-4)
        #expect(abs(gate.effectiveScale(baseScale: 4, hidden: [-2, 0, 0, 0], fallbackDirection: [0, 1, 0, 0])) < 1e-6)
        let hidden = MLXArray([Float(3), 0, 0, 0, -3, 0, 0, 0], [1, 2, 4])
        let dir = MLXArray([Float(0), 1, 0, 0])
        let addStep = SteerStep(direction: dir, scale: 4, geometry: .additive, contextGate: gate)
        let geoStep = SteerStep(direction: dir, scale: 0.8, geometry: .geodesic, contextGate: gate)
        let addOut = GeodesicSteer.apply(hidden: hidden, step: addStep)
        let geoOut = GeodesicSteer.apply(hidden: hidden, step: geoStep)
        eval(addOut, geoOut)
        let addVals = addOut.asArray(Float.self)
        let geoVals = geoOut.asArray(Float.self)
        #expect(abs(addVals[1] - 4) < 1e-4)
        #expect(abs(addVals[4] - (-3)) < 1e-5 && abs(addVals[5]) < 1e-5)
        #expect(abs(geoVals[1]) > 1.0)
        #expect(abs(geoVals[4] - (-3)) < 1e-5 && abs(geoVals[5]) < 1e-5)
    }

    @Test("steer_commit rejects and rolls back an adapter whose anti-steering rate exceeds tau_anti, and accepts within threshold")
    func commitAntiSteeringGateEnforcement() async throws {
        let engine = try await TinyModel.engine(.qwen3Moe)
        let container = try #require(await engine.container)
        let pristineLogits = try await Self.probeLogits("check baseline", in: container)
        let posPrompts = ["aaaa aaaa"]
        let direction = (0 ..< 64).map { $0 % 2 == 0 ? Float(1) : Float(-0.5) }
        let reversingPrompt = try await Self.findReversingPrompt(
            layer: 3, direction: direction, positive: posPrompts, in: container)
        var rejected = SteeringCommit(
            layer: 3,
            direction: direction,
            scale: 2,
            gate: GateRequest(heldOut: ["neutral check"], maxKL: 100, maxPerplexityRise: 100)
        )
        rejected.positive = posPrompts
        rejected.polarityProbes = [PolarityProbe(kind: .userNeg, prompt: reversingPrompt)]
        rejected.maxAntiSteeringRate = 0.25
        let failVerdict = try await engine.commitSteering(rejected)
        #expect(failVerdict.overlay == nil)
        #expect(failVerdict.report?.passed == false)
        let antiCheck = try #require(failVerdict.report?.checks.first { $0.name == "anti-steering rate" })
        #expect(antiCheck.passed == false && antiCheck.value > 0.25)
        #expect(try await Self.probeLogits("check baseline", in: container) == pristineLogits)
        var accepted = rejected
        accepted.maxAntiSteeringRate = 1.0
        let passVerdict = try await engine.commitSteering(accepted)
        let overlay = try #require(passVerdict.overlay)
        #expect(passVerdict.report?.passed == true)
        try await engine.undo(overlay.id)
        #expect(try await Self.probeLogits("check baseline", in: container) == pristineLogits)
    }

    @Test("MCP steer_search, steer_commit and intervene parse Slice 2 parameters and reject out-of-range anti-steering limits")
    func mcpParsingAndValidation() throws {
        let search = try ContrastRequest(mcp: [
            "positive": ["a", "b"], "negative": ["c", "d"], "heldOut": ["e"],
            "protectedPrompts": ["1+1=2"], "nullspaceMaxRank": 2, "evaluatePolarityFlips": true,
            "contextGate": ["threshold": 0.3, "sharpness": 15.0],
            "polarityProbes": [["kind": "sys", "prompt": "system prompt"]],
        ])
        #expect(search.protectedPrompts == ["1+1=2"] && search.nullspaceMaxRank == 2 && search.evaluatePolarityFlips)
        #expect(search.polarityProbes == [PolarityProbe(kind: .system, prompt: "system prompt")])
        #expect(search.contextGate == ContextSteerGate(contextVector: nil, threshold: 0.3, sharpness: 15.0))
        let commit = try SteeringCommit(mcp: [
            "layer": 1, "direction": [1.0, 0.0], "scale": 2.0, "positive": ["a"], "negative": ["b"],
            "protectedPrompts": ["2+2=4"], "maxAntiSteeringRate": 0.2, "heldOut": ["e"],
        ])
        #expect(commit.protectedPrompts == ["2+2=4"] && commit.negative == ["b"] && commit.maxAntiSteeringRate == 0.2)
        let intervention = try Intervention(mcp: [
            "kind": "steer", "layer": 0, "vector": [1.0, 0.0], "contextGate": ["threshold": 0.1, "sharpness": 10.0],
        ])
        #expect(intervention == .steer(layer: 0, vector: [1, 0], scale: 1, geometry: .additive, contextGate: ContextSteerGate(threshold: 0.1, sharpness: 10)))
        #expect(throws: MCPFailure.self) {
            try SteeringCommit(mcp: [
                "layer": 1, "direction": [1.0], "scale": 1.0, "positive": ["a"], "heldOut": ["e"], "maxAntiSteeringRate": 1.5,
            ])
        }
    }

    private static func probeLogits(_ prompt: String, in container: ModelContainer) async throws -> [Float] {
        try await container.performEvaluating(nonSendable: prompt) { context, prompt in
            let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let logits = MLXInferenceEngine.forward(context, probe: nil, ids, cache: try context.model.newCache(parameters: nil))
            let last = logits[0, ids.count - 1].asType(.float32)
            eval(last)
            return last.asArray(Float.self)
        }
    }

    private static func findReversingPrompt(
        layer: Int,
        direction: [Float],
        positive: [String],
        in container: ModelContainer
    ) async throws -> String {
        let candidates = (0 ..< 48).map { offset -> String in
            let scalar = String(UnicodeScalar(48 + offset)!)
            return String(repeating: scalar, count: 4)
        }
        return try await container.performEvaluating(nonSendable: (positive, candidates, direction)) { context, input in
            var commit = SteeringCommit(layer: layer, direction: input.2, scale: 2, gate: GateRequest(heldOut: ["x"]))
            commit.positive = input.0
            let source = try MLXInferenceEngine.steeringSource(commit, context: context)
            guard case .rankOne(_, let aVec, _, _) = source,
                  let found = MLXInferenceEngine.decoderLayer(layer, in: context.model),
                  let target = MLXInferenceEngine.steeringTarget(in: found.layer) else {
                return input.1[0]
            }
            var bestCandidate = input.1[0]
            var minDot = Float.greatestFiniteMagnitude
            for candidate in input.1 {
                let ids = try MLXInferenceEngine.encode([candidate], tokenizer: context.tokenizer)
                let xMean = try MLXInferenceEngine.meanInput(target.key, of: found.layer, linear: target.linear, prompts: ids, context: context)
                let dot = zip(aVec, xMean).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                if dot < minDot {
                    minDot = dot
                    bestCandidate = candidate
                }
            }
            return bestCandidate
        }
    }
}

@Suite("MAO-SNMF orthogonal semi-NMF decomposition and feature pinning (#570)", .serialized)
struct MAOSNMFDecomposerTests {
    @Test("MAO-SNMF converges monotonically with W >= 0, shrinks orthogonality defect under gamma, and isolates pinned features")
    func orthogonalityConvergenceAndFeaturePinning() throws {
        let rows = 32, dim = 16, features = 4
        var wTrueFlat = Array(repeating: Float(0), count: dim * features)
        for f in 0 ..< features {
            for offset in 0 ..< 4 {
                wTrueFlat[(f * 4 + offset) * features + f] = 0.5
            }
        }
        var zTrueFlat = Array(repeating: Float(0), count: rows * features)
        for t in 0 ..< rows {
            for f in 0 ..< features {
                let phase = Float((t + 1) * (f + 2))
                zTrueFlat[t * features + f] = Foundation.sin(phase) * 2.5 + Foundation.cos(phase * 0.7)
            }
        }
        let wTrue = MLXArray(wTrueFlat, [dim, features])
        let zTrue = MLXArray(zTrueFlat, [rows, features])
        let activations = matmul(zTrue, wTrue.T)
        eval(activations)

        let unpenalized = try MAOSNMFDecomposer.decompose(
            activations, config: MAOSNMFConfig(features: features, gamma: 0.0, iterations: 80, tolerance: 1e-7))
        let orthogonal = try MAOSNMFDecomposer.decompose(
            activations, config: MAOSNMFConfig(features: features, gamma: 2.0, iterations: 120, tolerance: 1e-7))

        let wValues = orthogonal.w.asArray(Float.self)
        #expect(wValues.allSatisfy { $0 >= 0 && $0.isFinite })
        #expect(Set(orthogonal.selectedSeeds).count == features)
        for (prev, next) in zip(orthogonal.objectiveHistory, orthogonal.objectiveHistory.dropFirst()) {
            #expect(next <= prev + 1e-4)
        }
        #expect(orthogonal.orthogonalityDefect < unpenalized.orthogonalityDefect)
        #expect(orthogonal.orthogonalityDefect < 0.15)
        #expect(orthogonal.reconstructionError / Float(rows * dim).squareRoot() < 0.25)

        let wf0 = orthogonal.w[0..., 0]
        let wf1 = orthogonal.w[0..., 1]
        let pinned = try orthogonal.pin(activations, feature: 0, value: 0)
        let residual0 = MLX.abs((pinned * wf0).sum(axis: -1)).max()
        let drift1 = MLX.abs(((pinned - activations) * wf1).sum(axis: -1)).max()
        eval(residual0, drift1)
        #expect(residual0.item(Float.self) < 1e-4)
        #expect(drift1.item(Float.self) < 0.2)

        let modulated = try orthogonal.modulate(activations, feature: 1, scale: 2.0)
        let before1 = (activations * wf1).sum(axis: -1)
        let after1 = (modulated * wf1).sum(axis: -1)
        let ratioErr = MLX.abs(after1 - 2.0 * before1).max()
        eval(ratioErr)
        #expect(ratioErr.item(Float.self) < 1e-4)
    }

    @Test("Invalid MAO-SNMF shapes and hyperparameters throw BasaltError.invalidRequestStructure")
    func validationRejectsMalformedInputs() {
        let matrix = MLXArray.ones([4, 6], dtype: .float32)
        #expect(throws: BasaltError.self) {
            try MAOSNMFDecomposer.decompose(MLXArray.ones([6], dtype: .float32), config: MAOSNMFConfig(features: 2))
        }
        #expect(throws: BasaltError.self) {
            try MAOSNMFDecomposer.decompose(matrix, config: MAOSNMFConfig(features: 7))
        }
        #expect(throws: BasaltError.self) {
            try MAOSNMFDecomposer.decompose(matrix, config: MAOSNMFConfig(features: 2, gamma: -1))
        }
    }
}
