import Foundation

public enum TelemetryExport {
    public static let contentType = "application/openmetrics-text; version=1.0.0; charset=utf-8"

    public static func openMetrics(_ snapshot: ServerTelemetrySnapshot) -> String {
        var lines: [String] = []
        appendCoreAndWindows(snapshot, into: &lines)
        appendDecisions(snapshot, into: &lines)
        appendConfidenceAndExperts(snapshot, into: &lines)
        appendCacheAndCalibration(snapshot, into: &lines)
        return (lines + ["# EOF"]).joined(separator: "\n") + "\n"
    }

    private static func appendMetric(
        _ name: String, _ type: String, _ help: String, _ samples: [(labels: String, value: Double)], into lines: inout [String]
    ) {
        let finite = samples.filter { $0.value.isFinite }
        guard !finite.isEmpty else { return }
        lines += ["# TYPE basalt_\(name) \(type)", "# HELP basalt_\(name) \(help)"]
        lines += finite.map {
            "basalt_\(name)\($0.labels.isEmpty ? "" : "{\($0.labels)}") \(value($0.value))"
        }
    }

    private static func appendCoreAndWindows(_ snapshot: ServerTelemetrySnapshot, into lines: inout [String]) {
        appendMetric("requests", "counter", "Generations served since start.", [("", Double(snapshot.totalRequests))], into: &lines)
        appendMetric("generated_tokens", "counter", "Completion tokens generated since start.", [("", Double(snapshot.totalTokensGenerated))], into: &lines)
        appendMetric("active_requests", "gauge", "Generations running now.", [("", Double(snapshot.activeRequests))], into: &lines)
        appendMetric("gpu_active_bytes", "gauge", "GPU memory in use.", [("", Double(snapshot.gpuActiveBytes))], into: &lines)
        appendMetric("gpu_peak_bytes", "gauge", "Peak GPU memory.", [("", Double(snapshot.gpuPeakBytes))], into: &lines)
        appendMetric("model_bytes", "gauge", "Loaded model weights.", [("", Double(snapshot.modelRAMMB) * 1_048_576)], into: &lines)
        let windows = snapshot.windows ?? []
        appendMetric("window_requests", "gauge", "Generations in the trailing window.", windows.map { ("window=\"\($0.seconds)s\"", Double($0.requests)) }, into: &lines)
        appendMetric("window_tokens", "gauge", "Completion tokens in the trailing window.", windows.map { ("window=\"\($0.seconds)s\"", Double($0.tokens)) }, into: &lines)
        appendMetric("ttft_seconds", "gauge", "Time to first token by quantile over the trailing window.", quantiles(windows, \.ttftP50, \.ttftP95), into: &lines)
        appendMetric("tokens_per_second", "gauge", "Decode speed by quantile over the trailing window.", quantiles(windows, \.tokensPerSecondP50, \.tokensPerSecondP95), into: &lines)
        appendMetric("prompt_cache_hit_ratio", "gauge", "Cached prompt tokens over prompt tokens in the trailing window.", windows.compactMap { w in w.promptCacheHitRatio.map { ("window=\"\(w.seconds)s\"", $0) } }, into: &lines)
        appendMetric("draft_acceptance_ratio", "gauge", "Accepted over proposed draft tokens in the trailing window.", windows.compactMap { w in w.draftAcceptance.map { ("window=\"\(w.seconds)s\"", $0) } }, into: &lines)
    }

    private static func appendDecisions(_ snapshot: ServerTelemetrySnapshot, into lines: inout [String]) {
        let decisions = (snapshot.decisions ?? [:]).sorted { $0.key < $1.key }
        appendMetric("decision_requests", "counter", "System One requests per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.requests)) }, into: &lines)
        appendMetric("decision_answers", "counter", "Answered questions per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.answers)) }, into: &lines)
        appendMetric("decision_abstains", "counter", "Abstained questions per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.abstains)) }, into: &lines)
        appendMetric("decision_escalations", "counter", "Requests escalated past a stage per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.escalations)) }, into: &lines)
        appendMetric("decision_cost_usd", "counter", "Cloud cost of decisions per model.", decisions.map { ("model=\"\(escape($0.key))\"", $0.value.cost) }, into: &lines)
        appendMetric("decision_confidence", "gauge", "Answers per confidence decile per decision model.", decisions.flatMap { name, item in
            item.confidence.enumerated().map { ("model=\"\(escape(name))\",decile=\"\($0.offset)\"", Double($0.element)) }
        }, into: &lines)
    }

    private static func appendConfidenceAndExperts(_ snapshot: ServerTelemetrySnapshot, into lines: inout [String]) {
        if let confidence = snapshot.confidence {
            appendMetric("confidence_responses", "counter", "Generations whose per-token confidence was measured.", [("", Double(confidence.responses))], into: &lines)
            appendMetric("confidence_min_probability", "gauge", "Measured generations per decile of their lowest top-token probability.", bins(confidence.minProbability, "decile"), into: &lines)
            appendMetric("confidence_mean_entropy", "gauge", "Measured generations per half-nat bin of their mean next-token entropy (the last bin is open).", bins(confidence.meanEntropy, "bin"), into: &lines)
            appendMetric("confidence_min_margin", "gauge", "Measured generations per decile of their smallest gap between the top two token probabilities.", bins(confidence.minMargin, "decile"), into: &lines)
            appendMetric(
                "confidence_belnap_total",
                "counter",
                "Measured generations per four-valued Belnap evidence state.",
                BelnapState.allCases.map { ("state=\"\($0.rawValue)\"", Double(confidence.belnap[$0])) },
                into: &lines
            )
        }
        if let experts = snapshot.experts {
            appendMetric("expert_cache_hits", "counter", "Streamed expert cache hits.", [("", Double(experts.cache.hits))], into: &lines)
            appendMetric("expert_cache_misses", "counter", "Streamed expert cache misses (loads from SSD).", [("", Double(experts.cache.misses))], into: &lines)
            appendMetric("expert_cache_evictions", "counter", "Streamed experts evicted from the cache.", [("", Double(experts.cache.evictions))], into: &lines)
            appendMetric("expert_bytes_read", "counter", "Bytes of expert weights read from SSD.", [("", Double(experts.cache.bytesRead.rawValue))], into: &lines)
            appendMetric("expert_cache_capacity_bytes", "gauge", "Size of the streamed expert cache.", [("", Double(experts.capacityBytes))], into: &lines)
            appendMetric("expert_layer_distinct", "gauge", "Distinct experts routed per MoE layer since load.", experts.layers.map { ("layer=\"\($0.layer)\"", Double($0.distinct)) }, into: &lines)
            appendMetric("expert_layer_top_share", "gauge", "Share of routings that went to the layer's eight most used experts.", experts.layers.map { ("layer=\"\($0.layer)\"", $0.topShare) }, into: &lines)
        }
    }

    private static func appendCacheAndCalibration(_ snapshot: ServerTelemetrySnapshot, into lines: inout [String]) {
        if let cache = snapshot.cache {
            appendMetric("prompt_cache_hits_total", "counter", "Prompt cache prefix hits since start.", [("", Double(cache.hits))], into: &lines)
            appendMetric("prompt_cache_misses_total", "counter", "Prompt cache prefix misses since start.", [("", Double(cache.misses))], into: &lines)
            appendMetric("prompt_cache_evictions_total", "counter", "Prompt cache evictions since start.", [("", Double(cache.evictions))], into: &lines)
            appendMetric("prompt_cache_entries", "gauge", "Entries stored in the prompt cache tree.", [("", Double(cache.cachedEntries))], into: &lines)
            appendMetric("prompt_cache_tokens", "gauge", "Tokens stored in the prompt cache tree.", [("", Double(cache.cachedTokens))], into: &lines)
            appendMetric("prompt_cache_bytes", "gauge", "Bytes stored in the prompt cache tree.", [("", Double(cache.cachedBytes))], into: &lines)
        }
        if let calibration = snapshot.calibration {
            appendMetric("calibration_samples_total", "counter", "Decisions evaluated against a teacher or verifier.", [("", Double(calibration.samples))], into: &lines)
            appendMetric("calibration_agreement_total", "counter", "Decisions agreeing with the teacher or verifier.", [("", Double(calibration.agreements))], into: &lines)
            appendMetric("calibration_ece", "gauge", "Expected calibration error across reliability bins.", [("", calibration.ece)], into: &lines)
        }
    }

    static func quantiles(_ windows: [TelemetryWindow], _ p50: KeyPath<TelemetryWindow, Double?>, _ p95: KeyPath<TelemetryWindow, Double?>) -> [(labels: String, value: Double)] {
        windows.flatMap { window in
            [("0.5", window[keyPath: p50]), ("0.95", window[keyPath: p95])].compactMap { quantile, item in
                item.map { ("window=\"\(window.seconds)s\",quantile=\"\(quantile)\"", $0) }
            }
        }
    }

    static func bins(_ counts: [Int], _ label: String) -> [(labels: String, value: Double)] {
        counts.enumerated().map { ("\(label)=\"\($0.offset)\"", Double($0.element)) }
    }

    static func value(_ number: Double) -> String { number.rounded() == number && abs(number) < 1e15 ? String(Int64(number)) : String(number) }

    static func escape(_ label: String) -> String {
        label.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n")
    }
}
