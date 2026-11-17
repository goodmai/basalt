import BasaltAppCore
import BasaltCore
import SwiftUI

struct ComposeProfileRow: View {
    let profile: ComposeBundledProfile
    let detectedRAMGB: Int
    let isActive: Bool
    let activate: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: profile.archetype.symbolName)
                .font(.title3)
                .foregroundStyle(profile.isPrimary ? Color.accentColor : Color.secondary)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(LocalizedStringKey(profile.archetype.titleKey))
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                    if profile.isPrimary {
                        Image(systemName: "star.fill")
                            .font(.caption2)
                            .foregroundStyle(.yellow)
                    }
                }
                Text(LocalizedStringKey(profile.archetype.summaryKey))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack(spacing: 4) {
                    Text(verbatim: profile.tier.displayTitle)
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.14), in: .capsule)
                    ForEach(profile.capabilityBadges, id: \.self) { badge in
                        Text(LocalizedStringKey(badge))
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.12), in: .capsule)
                    }
                }
            }
            Spacer(minLength: 6)
            VStack(alignment: .trailing, spacing: 5) {
                fitPill
                if isActive {
                    Text("Active")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                } else {
                    Button("Activate", action: activate)
                        .controlSize(.small)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var fitPill: some View {
        let fits = profile.fits(ramGB: detectedRAMGB)
        let label = String(format: "~%.1f GB", profile.estimatedResidentGB)
        return HStack(spacing: 4) {
            Circle()
                .fill(fits ? Color.green : Color.orange)
                .frame(width: 6, height: 6)
            Text(verbatim: label)
                .font(.caption2.monospacedDigit())
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background((fits ? Color.green : Color.orange).opacity(0.14), in: .capsule)
    }
}

struct ComposeProfileDetailPane: View {
    let profile: ComposeBundledProfile?
    let detectedRAMGB: Int
    let activeCompose: String
    let applyingID: String?
    let activate: (String) -> Void

    var body: some View {
        if let profile {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    headerCard(profile)
                    metadataGrid(profile)
                    yamlSection(profile)
                }
                .padding(16)
            }
        } else {
            ContentUnavailableView(
                "Select a Compose Profile",
                systemImage: "square.stack.3d.up",
                description: Text("Choose any hardware-tuned multi-model profile on the left.")
            )
        }
    }

    private func headerCard(_ profile: ComposeBundledProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Image(systemName: profile.archetype.symbolName)
                            .foregroundStyle(Color.accentColor)
                        Text(LocalizedStringKey(profile.archetype.titleKey))
                            .font(.title3.weight(.semibold))
                    }
                    Text(verbatim: profile.id)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                actionButton(profile)
            }
            Text(LocalizedStringKey(profile.archetype.summaryKey))
                .font(.callout)
                .foregroundStyle(.secondary)
            if profile.tier == ComposeMemoryTier.detect(ramGB: detectedRAMGB) && profile.isPrimary {
                HStack(spacing: 5) {
                    Image(systemName: "star.circle.fill")
                        .foregroundStyle(.yellow)
                    Text("Recommended Primary Profile for this Mac")
                        .foregroundStyle(.primary)
                }
                .font(.caption.weight(.semibold))
            } else if !profile.fits(ramGB: detectedRAMGB) {
                Label("Targets \(profile.tier.displayTitle) Macs; may exceed this Mac's memory", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(14)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 14))
    }

    @ViewBuilder
    private func actionButton(_ profile: ComposeBundledProfile) -> some View {
        if activeCompose == profile.id {
            Label("Active on Engine", systemImage: "checkmark.seal.fill")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.green)
        } else if applyingID == profile.id {
            ProgressView()
                .controlSize(.small)
        } else {
            Button(action: { activate(profile.id) }) {
                Text("Activate Profile")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.accentColor, in: .capsule)
            }
            .buttonStyle(.plain)
        }
    }

    private func metadataGrid(_ profile: ComposeBundledProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            LabeledContent("Memory Tier", value: profile.tier.displayTitle)
            LabeledContent("Estimated Resident RAM", value: String(format: "%.1f GB", profile.estimatedResidentGB))
            LabeledContent("Primary Model", value: profile.primaryModel)
            LabeledContent("CLI Command", value: "basalt compose up \(profile.id)")
                .font(.caption.monospaced())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.secondary.opacity(0.08), in: .rect(cornerRadius: 12))
    }

    private func yamlSection(_ profile: ComposeBundledProfile) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Compose Specification (YAML)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(verbatim: profile.yaml)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color.secondary.opacity(0.08), in: .rect(cornerRadius: 12))
        }
    }
}
