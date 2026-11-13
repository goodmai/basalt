import AppKit
import BasaltAppCore
import SwiftUI

public enum ComposerGlassID: Hashable, Sendable {
    case attach
    case field
    case submitOrStop
    case attachmentChip(UUID)
}

public enum QuickAskGlassID: Hashable, Sendable {
    case promptBar
    case answerBody
    case statusFooter
}

private struct GlassPolicyOverrideKey: EnvironmentKey {
    static let defaultValue: GlassSurfacePolicy? = nil
}

public extension EnvironmentValues {
    var glassPolicyOverride: GlassSurfacePolicy? {
        get { self[GlassPolicyOverrideKey.self] }
        set { self[GlassPolicyOverrideKey.self] = newValue }
    }

    var glassSurfacePolicy: GlassSurfacePolicy {
        if let override = glassPolicyOverride {
            return override
        }
        return GlassSurfacePolicy(
            reduceTransparency: accessibilityReduceTransparency,
            reduceMotion: accessibilityReduceMotion,
            increaseContrast: colorSchemeContrast == .increased
        )
    }
}

public struct AdaptiveGlassModifier<S: InsettableShape>: ViewModifier {
    let elevation: GlassElevation
    let shape: S
    @Environment(\.glassSurfacePolicy) private var policy
    @Environment(\.colorScheme) private var colorScheme

    public init(elevation: GlassElevation, shape: S) {
        self.elevation = elevation
        self.shape = shape
    }

    public func body(content: Content) -> some View {
        let labelColor = colorScheme == .dark ? Color.white.opacity(0.94) : Color.black.opacity(0.88)
        if policy.usesLiquidGlass {
            content
                .foregroundStyle(labelColor)
                .background {
                    shape.fill(colorScheme == .dark ? Color.white.opacity(0.06) : Color.white.opacity(0.55))
                }
                .glassEffect(glassSpec, in: shape)
                .overlay {
                    shape.strokeBorder(
                        labelColor.opacity(policy.strokeOpacity),
                        lineWidth: policy.increaseContrast ? 1.25 : 0.75
                    )
                }
        } else {
            content
                .foregroundStyle(labelColor)
                .background {
                    shape.fill(opaqueFill)
                }
                .overlay {
                    shape.strokeBorder(
                        labelColor.opacity(policy.strokeOpacity),
                        lineWidth: policy.increaseContrast ? 1.5 : 1.0
                    )
                }
        }
    }

    private var glassSpec: Glass {
        switch elevation {
        case .chromeRegular: .regular
        case .chromeInteractive: .regular.interactive()
        case .subtleClear: .clear
        case .prominentAction: .regular.tint(.accentColor).interactive()
        }
    }

    private var opaqueFill: AnyShapeStyle {
        switch elevation {
        case .prominentAction:
            AnyShapeStyle(Color.accentColor)
        case .chromeRegular, .chromeInteractive:
            AnyShapeStyle(Color(nsColor: .controlBackgroundColor))
        case .subtleClear:
            AnyShapeStyle(Color(nsColor: .windowBackgroundColor))
        }
    }
}

public struct AdaptiveGlassButtonModifier: ViewModifier {
    let prominent: Bool
    @Environment(\.glassSurfacePolicy) private var policy
    @Environment(\.colorScheme) private var colorScheme

    public func body(content: Content) -> some View {
        let labelColor = colorScheme == .dark ? Color.white.opacity(0.94) : Color.black.opacity(0.88)
        if policy.usesLiquidGlass {
            if prominent {
                content.buttonStyle(.glassProminent)
            } else {
                content.foregroundStyle(labelColor).buttonStyle(.glass)
            }
        } else {
            if prominent {
                content.buttonStyle(.borderedProminent)
            } else {
                content.foregroundStyle(labelColor).buttonStyle(.bordered)
            }
        }
    }
}

public extension View {
    func adaptiveGlass<S: InsettableShape>(_ elevation: GlassElevation, in shape: S) -> some View {
        modifier(AdaptiveGlassModifier(elevation: elevation, shape: shape))
    }

    func adaptiveGlassButton(prominent: Bool = false) -> some View {
        modifier(AdaptiveGlassButtonModifier(prominent: prominent))
    }
}
