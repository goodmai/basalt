import Foundation

public enum GlassElevation: String, Sendable, Equatable, Hashable, CaseIterable {
    case chromeRegular
    case chromeInteractive
    case subtleClear
    case prominentAction
}

public struct GlassSurfacePolicy: Sendable, Equatable, Hashable {
    public let reduceTransparency: Bool
    public let reduceMotion: Bool
    public let increaseContrast: Bool

    public init(
        reduceTransparency: Bool,
        reduceMotion: Bool,
        increaseContrast: Bool
    ) {
        self.reduceTransparency = reduceTransparency
        self.reduceMotion = reduceMotion
        self.increaseContrast = increaseContrast
    }

    public static let standard = GlassSurfacePolicy(
        reduceTransparency: false,
        reduceMotion: false,
        increaseContrast: false
    )

    public var usesLiquidGlass: Bool { !reduceTransparency }

    public var usesMorphAnimation: Bool { !reduceMotion }

    public var strokeOpacity: Double {
        switch (reduceTransparency, increaseContrast) {
        case (true, true): 0.55
        case (false, true): 0.36
        case (true, false): 0.24
        case (false, false): 0.12
        }
    }
}

public enum ScrollPinState: Sendable, Equatable {
    case pinned
    case detached(unreadDeltas: Int)

    public var isPinned: Bool {
        if case .pinned = self { true } else { false }
    }

    public var unreadDeltas: Int {
        if case .detached(let count) = self { count } else { 0 }
    }

    public var showsJumpControl: Bool {
        if case .detached = self { true } else { false }
    }

    public mutating func applyUserScroll(atBottom: Bool) {
        if atBottom {
            self = .pinned
        } else if case .pinned = self {
            self = .detached(unreadDeltas: 0)
        }
    }

    public mutating func recordStreamDelta() {
        if case .detached(let count) = self {
            self = .detached(unreadDeltas: count + 1)
        }
    }

    public mutating func jumpToBottom() {
        self = .pinned
    }
}
