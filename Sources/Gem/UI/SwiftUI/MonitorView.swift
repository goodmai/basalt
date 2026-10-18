import SwiftUI

@available(macOS 14.0, *)
public struct MonitorView: View {
    @State private var snapshot: ServerTelemetrySnapshot?
    @State private var isRefreshing: Bool = false
    private let timer = Timer.publish(every: 2.0, on: .main, in: .common).autoconnect()

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("System & Inference Monitor")
                            .font(.title2)
                            .bold()
                        if let s = snapshot {
                            Text("Chip: \(s.chipModel) • GPU: \(s.gpuName)")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    Button(action: {
                        Task { await refreshMetrics() }
                    }) {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isRefreshing)
                }
                .padding(.bottom, 8)

                if let s = snapshot {
                    // Metric Cards Grid
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                        metricCard(
                            title: "Tokens / Second",
                            value: String(format: "%.1f", s.tokensPerSecond),
                            unit: "tok/s",
                            color: .green,
                            systemImage: "bolt.fill"
                        )
                        metricCard(
                            title: "Time to First Token",
                            value: String(format: "%.2f", s.timeToFirstToken),
                            unit: "seconds",
                            color: .blue,
                            systemImage: "stopwatch.fill"
                        )
                        metricCard(
                            title: "KV Cache Size",
                            value: s.kvCacheSizeFormatted,
                            unit: "\(s.kvCacheSizeBytes) bytes",
                            color: .purple,
                            systemImage: "memorychip.fill"
                        )
                    }

                    // Memory Overview
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Unified Memory Telemetry")
                            .font(.headline)

                        let totalGB = Double(s.totalRAMGB)
                        let availGB = s.availableRAMGB
                        let usedGB = max(0, totalGB - availGB)
                        let usedFraction = totalGB > 0 ? usedGB / totalGB : 0.0

                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Unified Memory Usage")
                                    .font(.subheadline)
                                Spacer()
                                Text(String(format: "%.1f GB / %d GB (%.0f%%)", usedGB, s.totalRAMGB, usedFraction * 100))
                                    .font(.subheadline)
                                    .bold()
                            }
                            ProgressView(value: usedFraction)
                                .tint(usedFraction > 0.85 ? .red : (usedFraction > 0.70 ? .yellow : .blue))
                        }

                        HStack(spacing: 24) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Model Footprint")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("\(s.modelRAMMB) MB")
                                    .font(.title3)
                                    .bold()
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Available RAM")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f GB", s.availableRAMGB))
                                    .font(.title3)
                                    .bold()
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Active Requests")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("\(s.activeRequests)")
                                    .font(.title3)
                                    .bold()
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Parked Conversations")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("\(s.parkedConversations)")
                                    .font(.title3)
                                    .bold()
                            }
                        }
                        .padding(.top, 8)
                    }
                    .padding()
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(10)

                    // Operational Statistics
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Cumulative Inference Statistics")
                            .font(.headline)

                        HStack(spacing: 32) {
                            statItem(title: "Total Requests", value: "\(s.totalRequests)")
                            statItem(title: "Tokens Generated", value: "\(s.totalTokensGenerated)")
                            if let spec = s.speculativeAcceptanceRate {
                                statItem(title: "Speculative Acceptance", value: String(format: "%.1f%%", spec * 100))
                            }
                        }
                    }
                    .padding()
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(10)

                } else {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading telemetry metrics...")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 200)
                }
            }
            .padding(24)
        }
        .onAppear {
            Task { await refreshMetrics() }
        }
        .onReceive(timer) { _ in
            Task { await refreshMetrics() }
        }
    }

    private func refreshMetrics() async {
        isRefreshing = true
        let snap = await TelemetryTracker.shared.snapshot()
        self.snapshot = snap
        isRefreshing = false
    }

    private func metricCard(title: String, value: String, unit: String, color: Color, systemImage: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Image(systemName: systemImage)
                    .foregroundColor(color)
            }
            Text(value)
                .font(.title)
                .bold()
            Text(unit)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(10)
    }

    private func statItem(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.title3)
                .bold()
        }
    }
}
