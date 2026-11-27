import Foundation

extension TelemetryExport {
    public static func renderMoELoadBalance(_ layers: [(layer: Int, metrics: MoELoadBalanceMetrics)]) -> String {
        var lines: [String] = []
        appendMoELoadBalance(layers, into: &lines)
        guard !lines.isEmpty else { return "" }
        return lines.joined(separator: "\n") + "\n"
    }

    public static func renderMoELoadBalance(
        _ layers: [ExpertLayerRouting],
        metricsByLayer: [Int: MoELoadBalanceMetrics]
    ) -> String {
        let paired = layers.compactMap { entry -> (layer: Int, metrics: MoELoadBalanceMetrics)? in
            guard let metrics = metricsByLayer[entry.layer] else { return nil }
            return (layer: entry.layer, metrics: metrics)
        }
        return renderMoELoadBalance(paired)
    }

    static func appendMoELoadBalance(
        _ layers: [(layer: Int, metrics: MoELoadBalanceMetrics)],
        into lines: inout [String]
    ) {
        let ordered = layers.sorted { $0.layer < $1.layer }
        guard !ordered.isEmpty else { return }
        appendMoEGauge(
            "expert_layer_router_entropy",
            "Shannon entropy in nats of routed expert dispatches per MoE layer.",
            ordered.map { ($0.layer, $0.metrics.routerEntropy) },
            into: &lines
        )
        appendMoEGauge(
            "expert_layer_cv_squared",
            "Squared coefficient of variation of routed expert counts per MoE layer.",
            ordered.map { ($0.layer, $0.metrics.cvSquared) },
            into: &lines
        )
        appendMoEGauge(
            "expert_layer_gini",
            "Gini inequality coefficient of routed expert counts per MoE layer.",
            ordered.map { ($0.layer, $0.metrics.gini) },
            into: &lines
        )
        appendMoEGauge(
            "expert_layer_max_to_mean",
            "Ratio of the most used expert's dispatches to the layer mean.",
            ordered.map { ($0.layer, $0.metrics.maxToMeanRatio) },
            into: &lines
        )
        appendMoEGauge(
            "expert_layer_dead_experts",
            "Number of experts with zero dispatches in the MoE layer.",
            ordered.map { ($0.layer, Double($0.metrics.deadExpertCount)) },
            into: &lines
        )
    }

    private static func appendMoEGauge(
        _ name: String,
        _ help: String,
        _ samples: [(layer: Int, value: Double)],
        into lines: inout [String]
    ) {
        let finite = samples.filter { $0.value.isFinite }
        guard !finite.isEmpty else { return }
        lines += ["# TYPE basalt_\(name) gauge", "# HELP basalt_\(name) \(help)"]
        lines += finite.map { "basalt_\(name){layer=\"\($0.layer)\"} \(Self.value($0.value))" }
    }
}
