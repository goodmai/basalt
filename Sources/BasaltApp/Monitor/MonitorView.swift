import BasaltAppCore
import BasaltCore
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
                if let compose = store.state.compose { ComposeSection(compose: compose) }
            } else if store.state.problem == nil {
                ProgressView()
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
