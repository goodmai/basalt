extension TelemetryExport {
    public static func renderServingSLO(_ slo: ServingSLOSnapshot) -> String {
        var lines: [String] = []
        appendServingSLO(slo, into: &lines)
        return lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
    }

    static func appendServingSLO(_ slo: ServingSLOSnapshot, into lines: inout [String]) {
        appendSLOMetric("slo_ttft_seconds", "Time to first token by quantile over the serving SLO window.", [
            ("quantile=\"0.5\"", slo.ttftP50Seconds),
            ("quantile=\"0.95\"", slo.ttftP95Seconds),
            ("quantile=\"0.99\"", slo.ttftP99Seconds),
        ], into: &lines)
        appendSLOMetric("itl_seconds", "Inter-token latency by quantile over the serving SLO window.", [
            ("quantile=\"0.5\"", slo.itlP50Seconds),
            ("quantile=\"0.95\"", slo.itlP95Seconds),
            ("quantile=\"0.99\"", slo.itlP99Seconds),
        ], into: &lines)
        appendSLOMetric("tpot_seconds", "Mean time per output token over the serving SLO window.", [
            ("", slo.tpotMeanSeconds),
        ], into: &lines)
        appendSLOMetric("prefill_tokens_per_second", "Prefill throughput in tokens per second over the serving SLO window.", [
            ("", slo.prefillTokensPerSecond),
        ], into: &lines)
        appendSLOMetric("decode_tokens_per_second", "Decode throughput in tokens per second over the serving SLO window.", [
            ("", slo.decodeTokensPerSecond),
        ], into: &lines)
        appendSLOMetric("queue_wait_seconds", "Admission queue wait time by quantile over the serving SLO window.", [
            ("quantile=\"0.5\"", slo.queueWaitP50Seconds),
            ("quantile=\"0.95\"", slo.queueWaitP95Seconds),
        ], into: &lines)
        appendSLOMetric("memory_bandwidth_gbps", "Achieved unified memory bandwidth in GB/s during decode.", [
            ("", slo.achievedMemoryBandwidthGBps),
        ], into: &lines)
        appendSLOMetric("memory_bandwidth_peak_gbps", "Published peak unified memory bandwidth in GB/s for the active chip.", [
            ("", slo.peakMemoryBandwidthGBps),
        ], into: &lines)
        appendSLOMetric("memory_bandwidth_utilization", "Ratio of achieved to peak unified memory bandwidth.", [
            ("", slo.memoryBandwidthUtilization),
        ], into: &lines)
        appendSLOMetric(
            "serving_regime",
            "Active serving compute regime (1 for current regime, 0 otherwise).",
            ServingComputeRegime.allCases.map { ("regime=\"\($0.rawValue)\"", $0 == slo.regime ? 1.0 : 0.0) },
            into: &lines)
    }

    private static func appendSLOMetric(
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
