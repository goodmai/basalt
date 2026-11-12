import Foundation

public enum TelemetryExport {
    public static let contentType = "application/openmetrics-text; version=1.0.0; charset=utf-8"

    public static func openMetrics(_ snapshot: ServerTelemetrySnapshot) -> String {
        var lines: [String] = []
        func metric(_ name: String, _ type: String, _ help: String, _ samples: [(labels: String, value: Double)]) {
            guard !samples.isEmpty else { return }
            lines += ["# TYPE basalt_\(name) \(type)", "# HELP basalt_\(name) \(help)"]
            lines += samples.filter { $0.value.isFinite }.map { "basalt_\(name)\($0.labels.isEmpty ? "" : "{\($0.labels)}") \(value($0.value))" }
        }
        metric("requests", "counter", "Generations served since start.", [("", Double(snapshot.totalRequests))])
        metric("generated_tokens", "counter", "Completion tokens generated since start.", [("", Double(snapshot.totalTokensGenerated))])
        metric("active_requests", "gauge", "Generations running now.", [("", Double(snapshot.activeRequests))])
        metric("gpu_active_bytes", "gauge", "GPU memory in use.", [("", Double(snapshot.gpuActiveBytes))])
        metric("gpu_peak_bytes", "gauge", "Peak GPU memory.", [("", Double(snapshot.gpuPeakBytes))])
        metric("model_bytes", "gauge", "Loaded model weights.", [("", Double(snapshot.modelRAMMB) * 1_048_576)])
        let windows = snapshot.windows ?? []
        metric("window_requests", "gauge", "Generations in the trailing window.", windows.map { ("window=\"\($0.seconds)s\"", Double($0.requests)) })
        metric("window_tokens", "gauge", "Completion tokens in the trailing window.", windows.map { ("window=\"\($0.seconds)s\"", Double($0.tokens)) })
        metric("ttft_seconds", "gauge", "Time to first token by quantile over the trailing window.", quantiles(windows, \.ttftP50, \.ttftP95))
        metric("tokens_per_second", "gauge", "Decode speed by quantile over the trailing window.", quantiles(windows, \.tokensPerSecondP50, \.tokensPerSecondP95))
        metric("prompt_cache_hit_ratio", "gauge", "Cached prompt tokens over prompt tokens in the trailing window.", windows.compactMap { w in w.promptCacheHitRatio.map { ("window=\"\(w.seconds)s\"", $0) } })
        metric("draft_acceptance_ratio", "gauge", "Accepted over proposed draft tokens in the trailing window.", windows.compactMap { w in w.draftAcceptance.map { ("window=\"\(w.seconds)s\"", $0) } })
        let decisions = (snapshot.decisions ?? [:]).sorted { $0.key < $1.key }
        metric("decision_requests", "counter", "System One requests per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.requests)) })
        metric("decision_answers", "counter", "Answered questions per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.answers)) })
        metric("decision_abstains", "counter", "Abstained questions per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.abstains)) })
        metric("decision_escalations", "counter", "Requests escalated past a stage per decision model.", decisions.map { ("model=\"\(escape($0.key))\"", Double($0.value.escalations)) })
        metric("decision_cost_usd", "counter", "Cloud cost of decisions per model.", decisions.map { ("model=\"\(escape($0.key))\"", $0.value.cost) })
        metric("decision_confidence", "gauge", "Answers per confidence decile per decision model.", decisions.flatMap { name, value in
            value.confidence.enumerated().map { ("model=\"\(escape(name))\",decile=\"\($0.offset)\"", Double($0.element)) }
        })
        return (lines + ["# EOF"]).joined(separator: "\n") + "\n"
    }

    static func quantiles(_ windows: [TelemetryWindow], _ p50: KeyPath<TelemetryWindow, Double?>, _ p95: KeyPath<TelemetryWindow, Double?>) -> [(labels: String, value: Double)] {
        windows.flatMap { window in
            [("0.5", window[keyPath: p50]), ("0.95", window[keyPath: p95])].compactMap { quantile, value in
                value.map { ("window=\"\(window.seconds)s\",quantile=\"\(quantile)\"", $0) }
            }
        }
    }

    static func value(_ number: Double) -> String { number.rounded() == number && abs(number) < 1e15 ? String(Int64(number)) : String(number) }

    static func escape(_ label: String) -> String {
        label.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n")
    }
}
