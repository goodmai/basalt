import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Direct Logit Attribution math, conservation oracle, and MCP argument parsing (#857)")
struct CausalAttributionMathTests {
    @Test("CPU oracle verifies exact additive DLA conservation within 1e-10 across embedding, layers, and bias")
    func cpuOracleAdditiveConservation() throws {
        let h0: [Double] = [0.25, -0.5, 1.25, 0.75, -0.125, 0.5]
        let deltas: [[Double]] = [
            [0.1, 0.4, -0.2, 0.3, 0.15, -0.05],
            [-0.35, 0.25, 0.8, -0.1, 0.05, 0.2],
            [0.6, -0.7, 0.1, 0.45, -0.3, 0.1],
            [-0.15, 0.2, -0.4, 0.05, 0.4, -0.25],
        ]
        let targetRow: [Double] = [0.8, -0.3, 0.5, 1.1, -0.6, 0.2]
        let foilRow: [Double] = [-0.2, 0.4, -0.1, 0.3, 0.5, -0.4]
        let biasDiff = -0.375

        let direction = CausalAttributionMath.unembedDirection(targetRow: targetRow, foilRow: foilRow)
        let dla = CausalAttributionMath.directLogitAttributions(
            embeddingState: h0, layerDeltas: deltas, direction: direction, biasDiff: biasDiff
        )
        var hFinal = h0
        for delta in deltas {
            for idx in hFinal.indices { hFinal[idx] += delta[idx] }
        }
        let oracleTarget = CausalAttributionMath.dot(hFinal.span, targetRow.span)
        let oracleFoil = CausalAttributionMath.dot(hFinal.span, foilRow.span)
        let oracleDiff = oracleTarget - oracleFoil + biasDiff
        #expect(abs(dla.totalSum - oracleDiff) < 1e-10)
        #expect(abs((dla.embeddingDLA + dla.layerDLAs.reduce(0, +) + biasDiff) - dla.totalSum) < 1e-12)
        #expect(dla.layerDLAs.count == 4)

        let scores = dla.layerDLAs.enumerated().map { idx, value in
            LayerCausalScore(layerIndex: idx, directLogitAttribution: value, patchedLogitDiff: oracleDiff, causalRecovery: Double(idx) * 0.25)
        }
        let report = CausalSweepReport(
            targetTokenID: 7, foilTokenID: 13, cleanLogitDiff: oracleDiff, corruptedLogitDiff: oracleDiff - 1.0,
            embeddingDLA: dla.embeddingDLA, layers: scores
        ).withTotalDLASum(dla.totalSum)
        #expect(report.conservationError < 1e-10)
        #expect(report.topCausalLayer == 3)
        let expectedTopDLA = scores.max { abs($0.directLogitAttribution) < abs($1.directLogitAttribution) }?.layerIndex
        #expect(report.topDLALayer == expectedTopDLA)

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(CausalSweepReport.self, from: encoded)
        #expect(decoded == report)
    }

    @Test("causalRecovery equals 0 at corrupted, 1 at clean, 0.5 at midpoint, and safely returns 0 on degenerate gaps",
          arguments: [(3.5, -1.5), (-2.0, 4.0), (0.01, -0.01)] as [(Double, Double)])
    func causalRecoveryEndpointsAndDegeneracy(cleanDiff: Double, corruptedDiff: Double) {
        let midpoint = 0.5 * (cleanDiff + corruptedDiff)
        let atCorrupted = CausalAttributionMath.causalRecovery(cleanDiff: cleanDiff, corruptedDiff: corruptedDiff, patchedDiff: corruptedDiff)
        let atClean = CausalAttributionMath.causalRecovery(cleanDiff: cleanDiff, corruptedDiff: corruptedDiff, patchedDiff: cleanDiff)
        let atMid = CausalAttributionMath.causalRecovery(cleanDiff: cleanDiff, corruptedDiff: corruptedDiff, patchedDiff: midpoint)
        #expect(abs(atCorrupted - 0.0) < 1e-12)
        #expect(abs(atClean - 1.0) < 1e-12)
        #expect(abs(atMid - 0.5) < 1e-12)

        let zeroGap = CausalAttributionMath.causalRecovery(cleanDiff: cleanDiff, corruptedDiff: cleanDiff, patchedDiff: cleanDiff + 1.0)
        let tinyGap = CausalAttributionMath.causalRecovery(cleanDiff: 1.0, corruptedDiff: 1.0 + 1e-10, patchedDiff: 2.0)
        #expect(zeroGap == 0.0)
        #expect(tinyGap == 0.0)
    }

    @Test("causal_sweep MCP descriptor schema and CausalSweepRequest argument parser validate contract")
    func mcpDescriptorAndArgumentParser() throws {
        let descriptor = MCPToolCatalog.causalSweepDescriptor
        #expect(descriptor.name == "causal_sweep")
        #expect(descriptor.description.contains("--research-tools"))
        let schema = try #require(descriptor.inputSchema.rawValue as? [String: Any])
        let required = try #require(schema["required"] as? [String])
        let properties = try #require(schema["properties"] as? [String: Any])
        #expect(Set(required) == Set(["cleanPrompt", "corruptedPrompt"]))
        for key in ["cleanPrompt", "corruptedPrompt", "targetToken", "foilToken", "targetTokenID", "foilTokenID", "layers", "includePatching"] {
            #expect(properties.keys.contains(key))
        }
        let toolDef = MCPToolCatalog.causalSweepToolDefinition
        #expect(toolDef.name == "causal_sweep" && toolDef.parametersJSON != nil)

        let byToken = try CausalSweepRequest(mcp: [
            "cleanPrompt": "The capital of France is", "corruptedPrompt": "The capital of Poland is",
            "targetToken": " Paris", "foilToken": " Warsaw",
        ])
        #expect(byToken.targetToken == " Paris" && byToken.foilToken == " Warsaw" && byToken.includePatching)

        let byID = try CausalSweepRequest(mcp: [
            "cleanPrompt": "clean", "corruptedPrompt": "corrupt",
            "targetTokenID": 5, "foilTokenID": 9, "layers": [0, 2], "includePatching": false,
        ])
        #expect(byID.targetTokenID == 5 && byID.foilTokenID == 9 && byID.layers == [0, 2] && !byID.includePatching)

        let badPayloads: [[String: Any]] = [
            [:],
            ["cleanPrompt": "a", "corruptedPrompt": "b", "targetToken": "x"],
            ["cleanPrompt": "a", "corruptedPrompt": "b", "targetTokenID": 4, "foilTokenID": 4],
            ["cleanPrompt": "a", "corruptedPrompt": "b", "targetToken": "x", "targetTokenID": 4, "foilTokenID": 5],
            ["cleanPrompt": "a", "corruptedPrompt": "b", "targetTokenID": 4, "foilTokenID": 5, "layers": [Int]()],
            ["cleanPrompt": "a", "corruptedPrompt": "b", "targetTokenID": -1, "foilTokenID": 5],
        ]
        for payload in badPayloads {
            let err = #expect(throws: MCPFailure.self) { try CausalSweepRequest(mcp: payload) }
            #expect(err?.code == -32602)
        }
    }
}

@Suite("TinyModel live DLA conservation, residual patching sweep, and zero weight mutation (#857)", .serialized)
struct CausalSweepLiveModelTests {
    @Test("TinyModel.make(.qwen3) achieves < 1e-3 DLA conservation, finite recovery, ~1.0 final-layer recovery, and bit-identical weights")
    func qwen3LiveSweepAndZeroMutation() throws {
        let model = try TinyModel.make(.qwen3)
        let checksumBefore = Self.parameterChecksum(model)
        let clean = MLXArray([Int32(12), 45, 78, 91, 64]).reshaped(1, -1)
        let corrupted = MLXArray([Int32(12), 99, 33, 21, 88]).reshaped(1, -1)
        let (targetID, foilID) = try Self.extremeTokenPair(model: model, tokens: clean)

        let report = try MLXInferenceEngine.runCausalSweep(
            model: model, cleanTokens: clean, corruptedTokens: corrupted,
            targetTokenID: targetID, foilTokenID: foilID
        )
        #expect(report.targetTokenID == targetID && report.foilTokenID == foilID)
        #expect(report.conservationError < 1e-3)
        #expect(report.layers.map(\.layerIndex) == [0, 1, 2, 3])
        #expect(report.topDLALayer != nil && report.topCausalLayer != nil)
        #expect(abs(report.cleanLogitDiff - report.corruptedLogitDiff) > 1e-4)
        for score in report.layers {
            #expect(score.directLogitAttribution.isFinite)
            let patched = try #require(score.patchedLogitDiff)
            let recovery = try #require(score.causalRecovery)
            #expect(patched.isFinite && recovery.isFinite)
        }
        let lastRecovery = try #require(report.layers.last?.causalRecovery)
        #expect(abs(lastRecovery - 1.0) < 1e-4)
        #expect(Self.parameterChecksum(model) == checksumBefore)
    }

    @Test("Patching clean into clean preserves cleanLogitDiff at every layer, and subset DLA-only sweep preserves conservation")
    func cleanIntoCleanAndSubsetDLAOnly() throws {
        let model = try TinyModel.make(.qwen3)
        let checksumBefore = Self.parameterChecksum(model)
        let clean = MLXArray([Int32(14), 52, 83, 19, 77]).reshaped(1, -1)
        let (targetID, foilID) = try Self.extremeTokenPair(model: model, tokens: clean)

        let identitySweep = try MLXInferenceEngine.runCausalSweep(
            model: model, cleanTokens: clean, corruptedTokens: clean,
            targetTokenID: targetID, foilTokenID: foilID
        )
        #expect(identitySweep.conservationError < 1e-3)
        #expect(abs(identitySweep.cleanLogitDiff - identitySweep.corruptedLogitDiff) < 1e-6)
        for score in identitySweep.layers {
            let patched = try #require(score.patchedLogitDiff)
            #expect(abs(patched - identitySweep.cleanLogitDiff) < 1e-4)
            #expect(score.causalRecovery == 0.0)
            let normalized = CausalAttributionMath.causalRecovery(
                cleanDiff: identitySweep.cleanLogitDiff, corruptedDiff: identitySweep.cleanLogitDiff - 1.0, patchedDiff: patched
            )
            #expect(abs(normalized - 1.0) < 1e-4)
        }

        let subset = try MLXInferenceEngine.runCausalSweep(
            model: model, cleanTokens: clean, corruptedTokens: MLXArray([Int32(14), 10, 20, 30, 40]).reshaped(1, -1),
            targetTokenID: targetID, foilTokenID: foilID,
            options: CausalSweepOptions(layers: [3, 1, 3], includePatching: false)
        )
        #expect(subset.conservationError < 1e-3)
        #expect(subset.layers.map(\.layerIndex) == [1, 3])
        #expect(subset.topCausalLayer == nil)
        #expect(subset.layers.allSatisfy { $0.patchedLogitDiff == nil && $0.causalRecovery == nil })
        #expect(Self.parameterChecksum(model) == checksumBefore)
    }

    @Test("Quantized lm_head and tied embeddings both satisfy < 1e-3 DLA conservation with zero weight mutation")
    func quantizedAndTiedUnembeddingConservation() throws {
        let clean = MLXArray([Int32(12), 45, 78, 91, 64]).reshaped(1, -1)
        let corrupted = MLXArray([Int32(12), 99, 33, 21, 88]).reshaped(1, -1)

        let quantized = try TinyModel.make(.qwen3)
        MLXNN.quantize(model: quantized, groupSize: 32, bits: 4, mode: .affine)
        eval(quantized)
        let qChecksum = Self.parameterChecksum(quantized)
        let (qTarget, qFoil) = try Self.extremeTokenPair(model: quantized, tokens: clean)
        let qReport = try MLXInferenceEngine.runCausalSweep(
            model: quantized, cleanTokens: clean, corruptedTokens: corrupted,
            targetTokenID: qTarget, foilTokenID: qFoil
        )
        #expect(qReport.conservationError < 1e-3)
        let qLastRecovery = try #require(qReport.layers.last?.causalRecovery)
        #expect(abs(qLastRecovery - 1.0) < 1e-3)
        #expect(Self.parameterChecksum(quantized) == qChecksum)

        let tied = try TinyModel.make(.nemotronH)
        let tChecksum = Self.parameterChecksum(tied)
        let (tTarget, tFoil) = try Self.extremeTokenPair(model: tied, tokens: clean)
        let tReport = try MLXInferenceEngine.runCausalSweep(
            model: tied, cleanTokens: clean, corruptedTokens: corrupted,
            targetTokenID: tTarget, foilTokenID: tFoil
        )
        #expect(tReport.conservationError < 1e-3)
        let tLastRecovery = try #require(tReport.layers.last?.causalRecovery)
        #expect(abs(tLastRecovery - 1.0) < 1e-4)
        #expect(Self.parameterChecksum(tied) == tChecksum)
    }

    @Test("MLXInferenceEngine.runCausalSweep resolves tokens end-to-end, leaves no overlays, and rejects invalid requests")
    func engineEndToEndAndValidation() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let request = CausalSweepRequest(
            cleanPrompt: "hello world", corruptedPrompt: "zzzzz qqqqq", targetToken: "a", foilToken: "z"
        )
        let report = try await engine.runCausalSweep(request)
        #expect(report.conservationError < 1e-3)
        #expect(report.layers.count == 4)
        let expectedTarget = try #require(ByteTokenizer().encode(text: "a", addSpecialTokens: false).first)
        let expectedFoil = try #require(ByteTokenizer().encode(text: "z", addSpecialTokens: false).first)
        #expect(report.targetTokenID == expectedTarget)
        #expect(report.foilTokenID == expectedFoil)
        #expect(await engine.overlays().isEmpty)

        let invalid: [CausalSweepRequest] = [
            CausalSweepRequest(cleanPrompt: "", corruptedPrompt: "b", targetTokenID: 1, foilTokenID: 2),
            CausalSweepRequest(cleanPrompt: "a", corruptedPrompt: "b", targetTokenID: 1, foilTokenID: 1),
            CausalSweepRequest(cleanPrompt: "a", corruptedPrompt: "b", targetToken: "ab", foilToken: "z"),
            CausalSweepRequest(cleanPrompt: "a", corruptedPrompt: "b", targetTokenID: 1, foilTokenID: 2, layers: [99]),
        ]
        for bad in invalid {
            await #expect(throws: BasaltError.self) { try await engine.runCausalSweep(bad) }
        }
    }

    @Test("TinyModel softcapped families (.gemma4, .gemma4Unified, .inkling2Loop) resolve softcap and achieve < 1e-3 DLA conservation on non-looped families")
    func softcappedFamiliesDLAConservation() throws {
        let clean = MLXArray([Int32(1), 4, 7, 11]).reshaped(1, 4)
        let corrupted = MLXArray([Int32(1), 5, 8, 11]).reshaped(1, 4)
        for family: TinyModel.Family in [.gemma4, .gemma4Unified] {
            let model = try TinyModel.make(family)
            let softcap = try #require(MLXInferenceEngine.resolveFinalLogitSoftcap(of: model))
            #expect(softcap > 0)
            let (target, foil) = try Self.extremeTokenPair(model: model, tokens: clean)
            let report = try MLXInferenceEngine.runCausalSweep(
                model: model, cleanTokens: clean, corruptedTokens: corrupted,
                targetTokenID: target, foilTokenID: foil
            )
            #expect(report.conservationError < 1e-3)
            #expect(report.layers.allSatisfy { $0.directLogitAttribution.isFinite })
            let finalRecovery = try #require(report.layers.last?.causalRecovery)
            #expect(abs(finalRecovery - 1.0) < 1e-3)
        }
        let inkling = try TinyModel.make(.inkling2Loop)
        let inklingCap = try #require(MLXInferenceEngine.resolveFinalLogitSoftcap(of: inkling))
        #expect(inklingCap > 0)
        let (iTarget, iFoil) = try Self.extremeTokenPair(model: inkling, tokens: clean)
        let inklingReport = try MLXInferenceEngine.runCausalSweep(
            model: inkling, cleanTokens: clean, corruptedTokens: corrupted,
            targetTokenID: iTarget, foilTokenID: iFoil
        )
        #expect(inklingReport.layers.allSatisfy { $0.directLogitAttribution.isFinite })
    }

    private static func extremeTokenPair(model: any LanguageModel, tokens: MLXArray) throws -> (target: Int, foil: Int) {
        let cache = try model.newCache(parameters: nil)
        let logits = model(LMInput.Text(tokens: tokens), cache: cache, state: nil).logits
        let row = logits[0, tokens.dim(1) - 1].asType(.float32)
        eval(row)
        let values = row.asArray(Float.self)
        let maxIdx = values.indices.max { values[$0] < values[$1] } ?? 0
        let minIdx = values.indices.min { values[$0] < values[$1] } ?? (maxIdx == 0 ? 1 : 0)
        return (maxIdx, minIdx == maxIdx ? (maxIdx + 1) % max(values.count, 2) : minIdx)
    }

    private static func parameterChecksum(_ model: any LanguageModel) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        let sorted = model.parameters().flattened().sorted { $0.0 < $1.0 }
        for (name, array) in sorted {
            for byte in name.utf8 {
                hash = (hash ^ UInt64(byte)) &* 1_099_511_628_211
            }
            if array.dtype == .uint32 {
                for word in array.asArray(UInt32.self) {
                    hash = (hash ^ UInt64(word)) &* 1_099_511_628_211
                }
            } else {
                for value in array.asType(.float32).asArray(Float.self) {
                    hash = (hash ^ UInt64(value.bitPattern)) &* 1_099_511_628_211
                }
            }
        }
        return hash
    }
}
