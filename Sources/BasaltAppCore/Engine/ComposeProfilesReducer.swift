import BasaltCore
import Foundation

public enum ComposeTierFilter: Sendable, Equatable, Hashable {
    case recommended
    case tier(ComposeMemoryTier)
    case all
}

public struct ComposeProfilesState: Sendable, Equatable {
    public var detectedRAMGB: Int
    public var filter: ComposeTierFilter
    public var query: String
    public var activeCompose: String
    public var selectedID: String?
    public var applyingID: String?
    public var problem: String?

    public init(
        detectedRAMGB: Int = Int((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0).rounded()),
        filter: ComposeTierFilter = .recommended,
        query: String = "",
        activeCompose: String = "",
        selectedID: String? = nil
    ) {
        self.detectedRAMGB = max(8, detectedRAMGB)
        self.filter = filter
        self.query = query
        self.activeCompose = activeCompose
        let defaultTier = ComposeMemoryTier.detect(ramGB: max(8, detectedRAMGB))
        self.selectedID = selectedID ?? "\(defaultTier.rawValue)-\(ComposeProfileArchetype.dailyChat.rawValue)"
    }

    public var detectedTier: ComposeMemoryTier {
        ComposeMemoryTier.detect(ramGB: detectedRAMGB)
    }

    public var recommendedPrimary: [ComposeBundledProfile] {
        applySearch(ComposeProfileCatalog.recommended(for: detectedRAMGB, primaryOnly: true))
    }

    public var recommendedSecondary: [ComposeBundledProfile] {
        applySearch(ComposeProfileCatalog.profiles(for: detectedTier).filter { !$0.isPrimary })
    }

    public var filteredProfiles: [ComposeBundledProfile] {
        let base: [ComposeBundledProfile] = switch filter {
        case .recommended: ComposeProfileCatalog.profiles(for: detectedTier)
        case .tier(let tier): ComposeProfileCatalog.profiles(for: tier)
        case .all: ComposeProfileCatalog.profiles
        }
        return applySearch(base)
    }

    public var selectedProfile: ComposeBundledProfile? {
        if let selectedID, let found = ComposeProfileCatalog.resolve(id: selectedID) {
            return found
        }
        return filteredProfiles.first ?? ComposeProfileCatalog.profiles.first
    }

    private func applySearch(_ items: [ComposeBundledProfile]) -> [ComposeBundledProfile] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return items }
        return items.filter { profile in
            profile.id.localizedStandardContains(trimmed)
                || profile.archetype.titleKey.localizedStandardContains(trimmed)
                || profile.archetype.summaryKey.localizedStandardContains(trimmed)
                || profile.primaryModel.localizedStandardContains(trimmed)
                || profile.capabilityBadges.contains { $0.localizedStandardContains(trimmed) }
        }
    }
}

public enum ComposeProfilesAction: Sendable, Equatable {
    case filterChanged(ComposeTierFilter)
    case queryChanged(String)
    case select(String)
    case activate(String)
    case activated(String)
    case failed(String)
    case syncActive(String)
}

public enum ComposeProfilesEffect: Sendable, Equatable {
    case applyProfile(String)
}

public enum ComposeProfilesReducer {
    public static func reduce(
        _ state: inout ComposeProfilesState,
        _ action: ComposeProfilesAction
    ) -> [ComposeProfilesEffect] {
        switch action {
        case .filterChanged(let filter):
            state.filter = filter
            if let current = state.selectedID,
               !state.filteredProfiles.contains(where: { $0.id == current }),
               let first = state.filteredProfiles.first {
                state.selectedID = first.id
            }
            return []
        case .queryChanged(let query):
            state.query = query
            if let current = state.selectedID,
               !state.filteredProfiles.contains(where: { $0.id == current }),
               let first = state.filteredProfiles.first {
                state.selectedID = first.id
            }
            return []
        case .select(let id):
            state.selectedID = id
            return []
        case .activate(let id):
            guard ComposeProfileCatalog.resolve(id: id) != nil else {
                state.problem = "Unknown compose profile: \(id)"
                return []
            }
            state.applyingID = id
            state.selectedID = id
            state.problem = nil
            return [.applyProfile(id)]
        case .activated(let id):
            state.applyingID = nil
            state.activeCompose = id
            state.problem = nil
            return []
        case .failed(let message):
            state.applyingID = nil
            state.problem = message
            return []
        case .syncActive(let active):
            state.activeCompose = active
            return []
        }
    }
}
