import Foundation

public struct VisualSceneID: RawRepresentable, Hashable, Codable, Sendable, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ raw: String) { self.rawValue = raw }

    public var description: String { rawValue }

    public static let rootEmpty = Self("root.empty")
    public static let rootBannerFailed = Self("root.banner.failed")
    public static let chatConversationRich = Self("chat.conversation.rich")
    public static let chatInspectorOpen = Self("chat.inspector.open")
    public static let quickAskCompact = Self("quickask.compact")
    public static let quickAskStreaming = Self("quickask.streaming")
    public static let menuBarPopover = Self("menubar.popover")
    public static let libraryMixed = Self("library.mixed")
    public static let monitorCompose = Self("monitor.compose")
    public static let settingsTabs = Self("settings.tabs")
}

public enum AppearanceScheme: String, CaseIterable, Codable, Sendable {
    case light
    case dark
}

public struct TraitVariant: Hashable, Codable, Sendable, CustomStringConvertible {
    public var scheme: AppearanceScheme
    public var reduceTransparency: Bool
    public var increaseContrast: Bool
    public var reduceMotion: Bool
    public var localeIdentifier: String

    public init(
        scheme: AppearanceScheme,
        reduceTransparency: Bool = false,
        increaseContrast: Bool = false,
        reduceMotion: Bool = false,
        localeIdentifier: String = "en_US"
    ) {
        self.scheme = scheme
        self.reduceTransparency = reduceTransparency
        self.increaseContrast = increaseContrast
        self.reduceMotion = reduceMotion
        self.localeIdentifier = localeIdentifier
    }

    public static let lightStandard = Self(scheme: .light)
    public static let darkStandard = Self(scheme: .dark)
    public static let darkReducedTransparency = Self(
        scheme: .dark, reduceTransparency: true, increaseContrast: true, reduceMotion: true
    )
    public static let darkRussian = Self(scheme: .dark, localeIdentifier: "ru_RU")

    public var slug: String {
        var parts = [scheme.rawValue]
        if reduceTransparency { parts.append("opaque") }
        if increaseContrast { parts.append("hc") }
        if reduceMotion { parts.append("nomotion") }
        if localeIdentifier != "en_US" { parts.append(localeIdentifier) }
        return parts.joined(separator: "-")
    }

    public var description: String { slug }
}

public struct DiffTolerance: Equatable, Hashable, Codable, Sendable {
    public var pixelDeltaThreshold: Double
    public var maxChangedRatio: Double

    public init(pixelDeltaThreshold: Double, maxChangedRatio: Double) {
        self.pixelDeltaThreshold = pixelDeltaThreshold
        self.maxChangedRatio = maxChangedRatio
    }

    public static let exact = Self(pixelDeltaThreshold: 0.001, maxChangedRatio: 0.0)
    public static let crispUI = Self(pixelDeltaThreshold: 0.008, maxChangedRatio: 0.0005)
    public static let liveGlass = Self(pixelDeltaThreshold: 0.02, maxChangedRatio: 0.005)
}

public struct PixelBounds: Equatable, Codable, Sendable {
    public var minX: Int
    public var minY: Int
    public var maxX: Int
    public var maxY: Int

    public init(minX: Int, minY: Int, maxX: Int, maxY: Int) {
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
    }
}

public enum DiffVerdict: Equatable, Codable, Sendable {
    case identical
    case withinTolerance
    case mismatched
    case dimensionMismatch(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
}

public struct DiffResult: Equatable, Codable, Sendable {
    public var totalPixels: Int
    public var changedPixels: Int
    public var changedRatio: Double
    public var maxPixelDelta: Double
    public var bounds: PixelBounds?
    public var verdict: DiffVerdict

    public init(
        totalPixels: Int,
        changedPixels: Int,
        changedRatio: Double,
        maxPixelDelta: Double,
        bounds: PixelBounds?,
        verdict: DiffVerdict
    ) {
        self.totalPixels = totalPixels
        self.changedPixels = changedPixels
        self.changedRatio = changedRatio
        self.maxPixelDelta = maxPixelDelta
        self.bounds = bounds
        self.verdict = verdict
    }
}

public struct VisualSceneSpec: Hashable, Codable, Sendable {
    public var id: VisualSceneID
    public var width: Int
    public var height: Int
    public var scale: Int
    public var tolerance: DiffTolerance

    public init(
        id: VisualSceneID,
        width: Int,
        height: Int,
        scale: Int = 2,
        tolerance: DiffTolerance = .crispUI
    ) {
        self.id = id
        self.width = width
        self.height = height
        self.scale = scale
        self.tolerance = tolerance
    }

    public func goldenFilename(for variant: TraitVariant) -> String {
        "\(id.rawValue)__\(variant.slug).png"
    }
}

public enum VisualTestError: Error, Equatable, Sendable {
    case renderFailed(VisualSceneID)
    case baselineMissing(path: String)
    case invalidImageData(String)
    case windowNotFound(title: String)
    case screenCaptureUnavailable(String)
    case recordingFailed(String)
    case ioFailed(String)
}
