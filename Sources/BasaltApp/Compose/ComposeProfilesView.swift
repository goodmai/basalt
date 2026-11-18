import BasaltAppCore
import BasaltCore
import SwiftUI

public typealias ComposeProfilesStore = Store<ComposeProfilesState, ComposeProfilesAction, ComposeProfilesEffect>

public struct ComposeProfilesView: View {
    let store: ComposeProfilesStore
    @Environment(\.glassSurfacePolicy) private var policy

    public init(store: ComposeProfilesStore) {
        self.store = store
    }

    public var body: some View {
        VStack(spacing: 0) {
            topChromeBar
            Divider()
            HSplitView {
                profileList
                    .frame(minWidth: 380, idealWidth: 440)
                ComposeProfileDetailPane(
                    profile: store.state.selectedProfile,
                    detectedRAMGB: store.state.detectedRAMGB,
                    activeCompose: store.state.activeCompose,
                    applyingID: store.state.applyingID
                ) { id in
                    store.send(.activate(id))
                }
                .frame(minWidth: 380, idealWidth: 460)
            }
        }
        .frame(minWidth: 820, minHeight: 540)
    }

    private var topChromeBar: some View {
        GlassEffectContainer(spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    Label("Detected RAM: \(store.state.detectedRAMGB) GB", systemImage: "memorychip")
                        .font(.callout.weight(.semibold))
                    Text("Recommended tier: \(store.state.detectedTier.displayTitle)")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.18), in: .capsule)
                    Spacer()
                    searchBox
                }
                tierFilterStrip
                if let problem = store.state.problem {
                    Label(problem, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
        }
    }

    private var searchBox: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField(
                "Filter profiles or models",
                text: Binding(
                    get: { store.state.query },
                    set: { store.send(.queryChanged($0)) }
                )
            )
            .textFieldStyle(.plain)
            .frame(width: 210)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .adaptiveGlass(.chromeRegular, in: .capsule)
    }

    private var tierFilterStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                filterChip(
                    title: LocalizedStringKey("Recommended"),
                    systemImage: "star.fill",
                    isSelected: store.state.filter == .recommended
                ) {
                    store.send(.filterChanged(.recommended))
                }
                ForEach(ComposeMemoryTier.allCases) { tier in
                    let isHost = tier == store.state.detectedTier
                    filterChip(
                        title: LocalizedStringKey(tier.displayTitle),
                        systemImage: isHost ? "checkmark.circle.fill" : nil,
                        isSelected: store.state.filter == .tier(tier)
                    ) {
                        store.send(.filterChanged(.tier(tier)))
                    }
                }
                filterChip(
                    title: LocalizedStringKey("All 91"),
                    systemImage: "square.grid.2x2",
                    isSelected: store.state.filter == .all
                ) {
                    store.send(.filterChanged(.all))
                }
            }
        }
    }

    private func filterChip(
        title: LocalizedStringKey,
        systemImage: String?,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.caption2)
                }
                Text(title)
                    .font(.caption.weight(isSelected ? .semibold : .regular))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                isSelected ? Color.accentColor.opacity(0.24) : Color.secondary.opacity(0.10),
                in: .capsule
            )
        }
        .buttonStyle(.plain)
    }
}

extension ComposeProfilesView {
    @ViewBuilder
    fileprivate var profileList: some View {
        List(
            selection: Binding(
                get: { store.state.selectedProfile?.id },
                set: { if let id = $0 { store.send(.select(id)) } }
            )
        ) {
            switch store.state.filter {
            case .recommended:
                Section("Recommended for this Mac") {
                    ForEach(store.state.recommendedPrimary) { profile in
                        ComposeProfileRow(
                            profile: profile,
                            detectedRAMGB: store.state.detectedRAMGB,
                            isActive: store.state.activeCompose == profile.id
                        ) {
                            store.send(.activate(profile.id))
                        }
                        .tag(profile.id)
                    }
                }
                Section("Specialized Profiles for this RAM") {
                    ForEach(store.state.recommendedSecondary) { profile in
                        ComposeProfileRow(
                            profile: profile,
                            detectedRAMGB: store.state.detectedRAMGB,
                            isActive: store.state.activeCompose == profile.id
                        ) {
                            store.send(.activate(profile.id))
                        }
                        .tag(profile.id)
                    }
                }
            case .tier(let tier):
                let primary = store.state.filteredProfiles.filter(\.isPrimary)
                let secondary = store.state.filteredProfiles.filter { !$0.isPrimary }
                Section("Primary Profiles (\(tier.displayTitle))") {
                    ForEach(primary) { profile in
                        ComposeProfileRow(
                            profile: profile,
                            detectedRAMGB: store.state.detectedRAMGB,
                            isActive: store.state.activeCompose == profile.id
                        ) {
                            store.send(.activate(profile.id))
                        }
                        .tag(profile.id)
                    }
                }
                Section("Specialized Profiles (\(tier.displayTitle))") {
                    ForEach(secondary) { profile in
                        ComposeProfileRow(
                            profile: profile,
                            detectedRAMGB: store.state.detectedRAMGB,
                            isActive: store.state.activeCompose == profile.id
                        ) {
                            store.send(.activate(profile.id))
                        }
                        .tag(profile.id)
                    }
                }
            case .all:
                ForEach(ComposeMemoryTier.allCases) { tier in
                    let items = store.state.filteredProfiles.filter { $0.tier == tier }
                    if !items.isEmpty {
                        Section(LocalizedStringKey(tier.displayTitle)) {
                            ForEach(items) { profile in
                                ComposeProfileRow(
                                    profile: profile,
                                    detectedRAMGB: store.state.detectedRAMGB,
                                    isActive: store.state.activeCompose == profile.id
                                ) {
                                    store.send(.activate(profile.id))
                                }
                                .tag(profile.id)
                            }
                        }
                    }
                }
            }
        }
        .scrollEdgeEffectStyle(policy.usesLiquidGlass ? .automatic : .hard, for: .top)
    }
}
