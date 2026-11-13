import AppKit
import BasaltApp
import BasaltAppCore
import CoreGraphics
import Foundation
import QuartzCore
import SwiftUI

public struct FrameCaptureClient: Sendable {
    public typealias SceneCapture = @MainActor @Sendable (VisualSceneID, TraitVariant) async throws(VisualTestError) -> RenderedBitmap
    public typealias WindowCapture = @Sendable (CGWindowID, Int) async throws(VisualTestError) -> RenderedBitmap

    public var captureScene: SceneCapture
    public var captureWindow: WindowCapture

    public init(captureScene: @escaping SceneCapture, captureWindow: @escaping WindowCapture) {
        self.captureScene = captureScene
        self.captureWindow = captureWindow
    }
}

@MainActor
public enum OffscreenSceneRenderer {
    public static func isolatedDefaults(suiteName: String) -> UserDefaults {
        let defaults = UserDefaults(suiteName: suiteName) ?? .standard
        defaults.removePersistentDomain(forName: suiteName)
        defaults.set("mlx-community/gemma-4-12B-it-qat-4bit", forKey: "model")
        defaults.set(true, forKey: "quickAsk")
        defaults.set(true, forKey: "checkUpdates")
        defaults.set("OPENROUTER_API_KEY\nANTHROPIC_API_KEY", forKey: "cloudKeys")
        return defaults
    }

    public static func appearance(for variant: TraitVariant) -> NSAppearance? {
        switch (variant.scheme, variant.increaseContrast) {
        case (.light, false): NSAppearance(named: .aqua)
        case (.light, true): NSAppearance(named: .accessibilityHighContrastAqua)
        case (.dark, false): NSAppearance(named: .darkAqua)
        case (.dark, true): NSAppearance(named: .accessibilityHighContrastDarkAqua)
        }
    }

    public static func render<V: View>(
        _ rootView: V,
        spec: VisualSceneSpec,
        variant: TraitVariant,
        desktopBackdrop: Bool = false
    ) throws(VisualTestError) -> RenderedBitmap {
        let suiteName = "app.basalt.visual.\(UUID().uuidString)"
        let defaults = isolatedDefaults(suiteName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let colorScheme: ColorScheme = variant.scheme == .dark ? .dark : .light
        let policy = GlassSurfacePolicy(
            reduceTransparency: variant.reduceTransparency,
            reduceMotion: variant.reduceMotion,
            increaseContrast: variant.increaseContrast
        )
        let wrapped = BackdropContainer(
            width: CGFloat(spec.width),
            height: CGFloat(spec.height),
            scheme: variant.scheme,
            desktopBackdrop: desktopBackdrop,
            content: rootView
        )
        .environment(\.colorScheme, colorScheme)
        .environment(\.locale, Locale(identifier: variant.localeIdentifier))
        .environment(\.glassPolicyOverride, policy)
        .defaultAppStorage(defaults)
        .transaction { $0.disablesAnimations = true }

        let targetAppearance = appearance(for: variant)
        let savedAppAppearance = NSApp?.appearance
        NSApp?.appearance = targetAppearance
        defer { NSApp?.appearance = savedAppAppearance }

        let window = makeOffscreenWindow(spec: spec, variant: variant, appearance: targetAppearance)
        defer { window.close() }

        let host = NSHostingView(rootView: wrapped)
        host.frame = NSRect(x: 0, y: 0, width: spec.width, height: spec.height)
        host.appearance = targetAppearance
        window.contentView = host

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.current.duration = 0
        host.layoutSubtreeIfNeeded()
        window.makeFirstResponder(nil)
        host.layoutSubtreeIfNeeded()
        NSAnimationContext.endGrouping()
        CATransaction.commit()
        CATransaction.flush()
        return try captureBitmap(from: host, spec: spec, appearance: targetAppearance)
    }

    private static func makeOffscreenWindow(
        spec: VisualSceneSpec,
        variant: TraitVariant,
        appearance: NSAppearance?
    ) -> NSWindow {
        let rect = NSRect(x: -16_000, y: -16_000, width: spec.width, height: spec.height)
        let window = NSWindow(contentRect: rect, styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.colorSpace = .sRGB
        window.appearance = appearance
        window.backgroundColor = variant.scheme == .dark
            ? NSColor(srgbRed: 0.11, green: 0.11, blue: 0.13, alpha: 1.0)
            : NSColor(srgbRed: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        return window
    }

    private static func captureBitmap(
        from host: NSView,
        spec: VisualSceneSpec,
        appearance: NSAppearance?
    ) throws(VisualTestError) -> RenderedBitmap {
        let pixelWidth = spec.width * max(1, spec.scale)
        let pixelHeight = spec.height * max(1, spec.scale)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: pixelWidth,
            pixelsHigh: pixelHeight,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .calibratedRGB,
            bytesPerRow: pixelWidth * 4,
            bitsPerPixel: 32
        ) else {
            throw .renderFailed(spec.id)
        }
        rep.size = NSSize(width: spec.width, height: spec.height)
        if let appearance {
            appearance.performAsCurrentDrawingAppearance {
                host.cacheDisplay(in: host.bounds, to: rep)
            }
        } else {
            host.cacheDisplay(in: host.bounds, to: rep)
        }
        guard let cgImage = rep.cgImage else {
            throw .renderFailed(spec.id)
        }
        return try BitmapCodec.fromCGImage(cgImage)
    }
}

private struct BackdropContainer<Content: View>: View {
    let width: CGFloat
    let height: CGFloat
    let scheme: AppearanceScheme
    let desktopBackdrop: Bool
    let content: Content

    var body: some View {
        ZStack {
            if desktopBackdrop {
                LinearGradient(
                    colors: scheme == .dark
                        ? [Color(red: 0.08, green: 0.12, blue: 0.22), Color(red: 0.18, green: 0.10, blue: 0.24)]
                        : [Color(red: 0.84, green: 0.90, blue: 0.98), Color(red: 0.94, green: 0.88, blue: 0.96)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            } else {
                Color(nsColor: .windowBackgroundColor)
            }
            content
        }
        .frame(width: width, height: height)
        .clipped()
    }
}
