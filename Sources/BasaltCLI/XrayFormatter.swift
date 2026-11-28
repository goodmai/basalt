import ArgumentParser
import BasaltCore
import Foundation

enum XrayFormatter {
    static func json<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    static func parseLayers(_ spec: String?) throws -> [Int]? {
        guard let spec, !spec.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        var result: [Int] = []
        for raw in spec.split(separator: ",") {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard let index = Int(trimmed), index >= 0 else {
                throw ValidationError("Invalid layer index '\(trimmed)' in --layers (expected non-negative comma-separated integers)")
            }
            result.append(index)
        }
        return result.isEmpty ? nil : result
    }

    static func formatInspect(_ anatomy: ModelAnatomy, tensors: [TensorInfo]) -> String {
        var lines: [String] = []
        let sizeMB = Double(anatomy.bytes.rawValue) / 1_048_576.0
        lines.append("Model X-Ray Inspect")
        lines.append("  architecture : \(anatomy.architecture) (\(anatomy.modelType))")
        lines.append("  layers       : \(anatomy.layers.count)  hidden=\(anatomy.hiddenSize)  vocab=\(anatomy.vocabulary)")
        lines.append(String(format: "  parameters   : %d (%.1f MB)  quant=%@", anatomy.parameterCount, sizeMB, anatomy.quantization ?? "dense"))
        let kinds = Dictionary(grouping: anatomy.layers, by: \.kind).mapValues(\.count)
        let kindSummary = kinds.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
        if !kindSummary.isEmpty { lines.append("  layer kinds  : \(kindSummary)") }
        if let moe = anatomy.layers.first(where: { ($0.experts ?? 0) > 0 }),
           let experts = moe.experts {
            let active = moe.expertsPerToken ?? 0
            lines.append("  moe geometry : \(experts) experts, top-\(active) active")
        }
        if !tensors.isEmpty {
            let dtypes = Dictionary(grouping: tensors, by: \.dtype).mapValues(\.count)
            let dtypeSummary = dtypes.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            let quantizedCount = tensors.filter(\.quantized).count
            lines.append("  tensors      : \(tensors.count) total (\(quantizedCount) quantized) [\(dtypeSummary)]")
        }
        return lines.joined(separator: "\n")
    }

    static func formatDepth(_ report: DepthDiagnosticsReport) -> String {
        var lines: [String] = []
        lines.append("Activation Depth Diagnostics (tokens=\(report.promptTokenCount), layers=\(report.layerCount))")
        lines.append(String(format: "  max S_max=%.3f (layer %d)  mean growth=%.3f", report.maxSMaxOverall, report.maxSMaxLayer, report.meanResidualGrowth))
        lines.append("  layer | norm_in | norm_out | growth | s_max  | softcap_sat | eff_rank")
        for layer in report.layers {
            let satStr = layer.softcapSaturationRate.map { String(format: "%10.2f%%", $0 * 100.0) } ?? "          -"
            let rankStr = layer.effectiveRank.map { String(format: "%.2f", $0) } ?? "-"
            lines.append(String(
                format: "  %5d | %7.3f | %8.3f | %6.3f | %6.3f | %@ | %@",
                layer.layerIndex, layer.residualNormIn, layer.residualNormOut, layer.residualNormGrowth, layer.maxPreSoftmaxLogit, satStr, rankStr
            ))
        }
        return lines.joined(separator: "\n")
    }

    static func formatOutliers(_ report: OutlierProfileReport) -> String {
        var lines: [String] = []
        lines.append("Outlier & Attention Sink Profile (prompts=\(report.promptCount), tokens=\(report.totalTokens), sensitive=\(report.sensitiveLayers.count))")
        lines.append("  layer | kurtosis | crest  | dead_pct | sink_mass | tier")
        for layer in report.layers {
            let deadStr = layer.deadNeuronFraction.map { String(format: "%7.2f%%", $0 * 100.0) } ?? "       -"
            let sinkStr = layer.meanAttentionSinkMass.map { String(format: "%9.3f", $0) } ?? "        -"
            lines.append(String(
                format: "  %5d | %8.2f | %6.2f | %@ | %@ | %@",
                layer.layerIndex, layer.excessKurtosis, layer.crestFactor, deadStr, sinkStr, layer.riskTier.rawValue
            ))
        }
        return lines.joined(separator: "\n")
    }

    static func formatCausal(_ report: CausalSweepReport) -> String {
        var lines: [String] = []
        lines.append(String(
            format: "Causal Sweep (target=%d foil=%d clean=%.4f corrupted=%.4f conservation_err=%.2e)",
            report.targetTokenID, report.foilTokenID, report.cleanLogitDiff, report.corruptedLogitDiff, report.conservationError
        ))
        lines.append("  layer |     dla | patched | recovery")
        for layer in report.layers {
            let patched = layer.patchedLogitDiff.map { String(format: "%7.3f", $0) } ?? "      -"
            let recovery = layer.causalRecovery.map { String(format: "%8.3f", $0) } ?? "       -"
            lines.append(String(format: "  %5d | %7.3f | %@ | %@", layer.layerIndex, layer.directLogitAttribution, patched, recovery))
        }
        return lines.joined(separator: "\n")
    }
}
