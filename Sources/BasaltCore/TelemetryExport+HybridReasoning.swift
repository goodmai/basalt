import Foundation

extension TelemetryExport {
    public static func renderRecurrentStateHealth(_ health: RecurrentStateHealthSnapshot) -> String {
        var lines: [String] = []
        appendRecurrentStateHealth(health, into: &lines)
        return lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
    }

    public static func renderReasoningTelemetry(_ reasoning: ReasoningTelemetrySnapshot) -> String {
        var lines: [String] = []
        appendReasoningTelemetry(reasoning, into: &lines)
        return lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
    }

    static func appendRecurrentStateHealth(_ health: RecurrentStateHealthSnapshot, into lines: inout [String]) {
        let normSamples = recurrentSamples(health, layerValue: \.frobeniusNorm, fallback: \.maxFrobeniusNorm)
        appendGauge(
            "recurrent_state_frobenius_norm",
            "Frobenius norm of recurrent state tensors per layer.",
            normSamples,
            into: &lines
        )
        let maxAbsSamples = recurrentSamples(health, layerValue: \.maxAbsEntry, fallback: \.maxAbsEntry)
        appendGauge(
            "recurrent_state_max_abs",
            "Maximum absolute entry in recurrent state tensors per layer.",
            maxAbsSamples,
            into: &lines
        )
        appendGauge(
            "recurrent_state_non_finite_total",
            "Total non-finite entries across recurrent state tensors.",
            [("", Double(health.totalNonFiniteCount))],
            into: &lines
        )
        appendGauge(
            "recurrent_state_saturated_layers",
            "Recurrent layers exceeding saturation threshold or containing non-finite entries.",
            [("", Double(health.saturatedLayerCount))],
            into: &lines
        )
        appendOptionalRecurrentGauges(health, into: &lines)
    }

    static func appendReasoningTelemetry(_ reasoning: ReasoningTelemetrySnapshot, into lines: inout [String]) {
        appendGauge(
            "reasoning_thinking_to_answer_ratio",
            "Ratio of thinking tokens to answer tokens across recent reasoning turns.",
            [("", reasoning.thinkingToAnswerRatio)],
            into: &lines
        )
        appendGauge(
            "reasoning_ngram_loop_rate",
            "Mean repeated n-gram fraction inside thinking traces.",
            [("", reasoning.meanNGramLoopRate)],
            into: &lines
        )
        appendGauge(
            "reasoning_budget_truncation_rate",
            "Fraction of reasoning turns hitting the token budget before closing the thinking block.",
            [("", reasoning.budgetTruncationRate)],
            into: &lines
        )
        if let entropyDrop = reasoning.meanPostThinkEntropyDrop {
            appendGauge(
                "reasoning_post_think_entropy_drop",
                "Mean next-token entropy reduction from thinking span to final answer span.",
                [("", entropyDrop)],
                into: &lines
            )
        }
    }

    private static func appendOptionalRecurrentGauges(
        _ health: RecurrentStateHealthSnapshot,
        into lines: inout [String]
    ) {
        let betaSamples = health.layers.compactMap { layer in
            layer.betaGuardTriggerRate.map { (layerLabels(layer), $0) }
        }
        if !betaSamples.isEmpty {
            appendGauge(
                "recurrent_state_beta_guard_rate",
                "Fraction of Gated DeltaNet-2 steps triggering beta_guard stabilization.",
                betaSamples,
                into: &lines
            )
        } else if let meanBeta = health.meanBetaGuardRate {
            appendGauge(
                "recurrent_state_beta_guard_rate",
                "Mean fraction of Gated DeltaNet-2 steps triggering beta_guard stabilization.",
                [("", meanBeta)],
                into: &lines
            )
        }
        let growthSamples = health.layers.compactMap { layer in
            layer.relativeGrowthVsBaseline.map { (layerLabels(layer), $0) }
        }
        appendGauge(
            "recurrent_state_relative_growth",
            "Recurrent state Frobenius norm relative to baseline norm per layer.",
            growthSamples,
            into: &lines
        )
    }

    private static func recurrentSamples(
        _ health: RecurrentStateHealthSnapshot,
        layerValue: (LayerStateHealth) -> Double,
        fallback: (RecurrentStateHealthSnapshot) -> Double
    ) -> [(labels: String, value: Double)] {
        if !health.layers.isEmpty {
            return health.layers.map { (layerLabels($0), layerValue($0)) }
        }
        return health.recurrentLayerCount > 0 ? [("", fallback(health))] : []
    }

    private static func layerLabels(_ layer: LayerStateHealth) -> String {
        "layer=\"\(layer.layerIndex)\",kind=\"\(escape(layer.cacheKind))\""
    }

    private static func appendGauge(
        _ name: String,
        _ help: String,
        _ samples: [(labels: String, value: Double)],
        into lines: inout [String]
    ) {
        let finite = samples.filter { $0.value.isFinite }
        guard !finite.isEmpty else { return }
        lines += ["# TYPE basalt_\(name) gauge", "# HELP basalt_\(name) \(help)"]
        lines += finite.map {
            "basalt_\(name)\($0.labels.isEmpty ? "" : "{\($0.labels)}") \(value($0.value))"
        }
    }
}
