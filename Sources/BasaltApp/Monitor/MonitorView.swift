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
