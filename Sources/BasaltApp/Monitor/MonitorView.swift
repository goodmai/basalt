import BasaltAppCore
import BasaltCore
import Charts
import SwiftUI

public struct MonitorView: View {
    let store: MonitorStore

    public init(store: MonitorStore) { self.store = store }

    public var body: some View {
        Form {
            if let problem = store.state.problem {
                Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            if let snapshot = store.state.snapshot {
                Section("Generation") {
                    LabeledContent("Tokens per second", value: String(format: "%.1f", snapshot.tokensPerSecond))
                    LabeledContent("Time to first token", value: String(format: "%.2f s", snapshot.timeToFirstToken))
                    LabeledContent("Requests", value: "\(snapshot.totalRequests) total, \(snapshot.activeRequests) active")
                    LabeledContent("Tokens generated", value: "\(snapshot.totalTokensGenerated)")
                }
                Section("Memory") {
                    LabeledContent("Model", value: "\(snapshot.modelRAMMB) MB")
                    LabeledContent("GPU active", value: ByteCountFormatter.string(fromByteCount: snapshot.gpuActiveBytes, countStyle: .memory))
                    LabeledContent("GPU peak", value: ByteCountFormatter.string(fromByteCount: snapshot.gpuPeakBytes, countStyle: .memory))
                    LabeledContent("Available", value: String(format: "%.1f of %d GB", snapshot.availableRAMGB, snapshot.totalRAMGB))
                }
                Section("Hardware") {
                    LabeledContent("Chip", value: snapshot.chipModel)
                    LabeledContent("GPU", value: snapshot.gpuName)
                }
                if let windows = snapshot.windows, !windows.isEmpty { WindowsSection(windows: windows) }
                if let decisions = snapshot.decisions, !decisions.isEmpty { DecisionsSection(decisions: decisions) }
                if let confidence = snapshot.confidence { ConfidenceSection(confidence: confidence) }
                if let compose = store.state.compose { ComposeSection(compose: compose) }
            } else if store.state.problem == nil {
                ProgressView().accessibilityLabel("Waiting for the engine's statistics")
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 360, minHeight: 420)
        .onAppear { store.send(.appeared) }
        .onDisappear { store.send(.disappeared) }
    }
}

struct ComposeSection: View {
    let compose: ComposeInfo

    var body: some View {
        Section("Compose system \(compose.name)") {
            LabeledContent("Pipelines", value: compose.pipelines.joined(separator: ", "))
            LabeledContent("Cloud spend today", value: String(format: "$%.4f", compose.spentToday))
            ForEach(compose.models, id: \.name) { model in
                LabeledContent {
                    Text(Self.detail(model)).foregroundStyle(model.state == "failed" || model.state == "no-key" ? .red : .secondary)
                } label: {
                    Label(model.name, systemImage: model.isCloud ? "cloud" : "desktopcomputer")
                }
            }
        }
    }

    static func detail(_ model: ComposeInfo.Model) -> String {
        var parts = [model.state, model.keep]
        if let spent = model.spentToday { parts.append(String(format: "$%.4f", spent) + (model.limit.map { String(format: " of $%.2f", $0) } ?? "")) }
        if let refusal = model.refusal { parts.append(refusal) }
        return parts.joined(separator: " · ")
    }
}

struct WindowsSection: View {
    let windows: [TelemetryWindow]

    var body: some View {
        Section("Recent traffic") {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                GridRow {
                    Text("Window")
                    Text("Requests")
                    Text("TTFT p50 / p95")
                    Text("tok/s p50 / p95")
                    Text("Cache hits")
                }
                .font(.caption).foregroundStyle(.secondary)
                ForEach(windows, id: \.seconds) { window in
                    GridRow {
                        Text(Self.span(window.seconds))
                        Text("\(window.requests)")
                        Text(Self.pair(window.ttftP50, window.ttftP95, format: "%.2f s"))
                        Text(Self.pair(window.tokensPerSecondP50, window.tokensPerSecondP95, format: "%.1f"))
                        Text(window.promptCacheHitRatio.map { String(format: "%.0f%%", $0 * 100) } ?? "–")
                    }
                    .monospacedDigit()
                }
            }
        }
    }

    static func span(_ seconds: Int) -> String { seconds >= 3600 ? "\(seconds / 3600) h" : "\(seconds / 60) min" }

    static func pair(_ low: Double?, _ high: Double?, format: String) -> String {
        guard let low, let high else { return "–" }
        return String(format: format, low) + " / " + String(format: format, high)
    }
}

struct DecisionsSection: View {
    let decisions: [String: DecisionTelemetry]

    var body: some View {
        Section("Decisions") {
            ForEach(decisions.keys.sorted(), id: \.self) { model in
                if let value = decisions[model] {
                    LabeledContent(model) {
                        Text("\(value.requests) requests · \(value.answers) answered · \(value.abstains) abstained" + (value.cost > 0 ? String(format: " · $%.4f", value.cost) : ""))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}

struct ConfidenceSection: View {
    let confidence: ConfidenceTelemetry

    var body: some View {
        Section("Confidence of \(confidence.responses) measured answers") {
            Chart(Array(confidence.minProbability.enumerated()), id: \.offset) { bin in
                BarMark(x: .value("Lowest top-token probability", Double(bin.offset) / 10), y: .value("Answers", bin.element), width: .ratio(0.8))
            }
            .chartXAxisLabel("Lowest top-token probability (decile)")
            .frame(height: 120)
            .accessibilityLabel("Answers per decile of their lowest top-token probability")
        }
    }
}
