import Foundation

public struct IQBlockDescriptor: Sendable, Equatable, Hashable {
    public enum Family: String, Sendable, Equatable, Hashable, CaseIterable {
        case iq
        case kQuant
        case blockQuant
    }

    public let ggmlType: GGMLType
    public let family: Family
    public let elementsPerBlock: Int
    public let bytesPerBlock: Int
    public let bitsPerWeight: Double
    public let hasImportanceMatrixSupport: Bool
    public let lutSizeBytes: Int
    public let distortionFactor: Double

    public init(
        ggmlType: GGMLType,
        family: Family,
        hasImportanceMatrixSupport: Bool,
        lutSizeBytes: Int,
        distortionFactor: Double
    ) {
        let (elements, bytes) = ggmlType.block
        self.ggmlType = ggmlType
        self.family = family
        self.elementsPerBlock = elements
        self.bytesPerBlock = bytes
        self.bitsPerWeight = Double(bytes * 8) / Double(elements)
        self.hasImportanceMatrixSupport = hasImportanceMatrixSupport
        self.lutSizeBytes = lutSizeBytes
        self.distortionFactor = distortionFactor
    }

    public var isIQ: Bool { family == .iq }

    public var unitDistortion: Double {
        distortionFactor * pow(2.0, -2.0 * bitsPerWeight)
    }

    public static let iq1S = IQBlockDescriptor(
        ggmlType: .iq1S, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 16_384, distortionFactor: 0.90)

    public static let iq1M = IQBlockDescriptor(
        ggmlType: .iq1M, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 16_384, distortionFactor: 0.86)

    public static let iq2XXS = IQBlockDescriptor(
        ggmlType: .iq2XXS, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 2_176, distortionFactor: 0.85)

    public static let iq2XS = IQBlockDescriptor(
        ggmlType: .iq2XS, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 4_224, distortionFactor: 0.84)

    public static let iq2S = IQBlockDescriptor(
        ggmlType: .iq2S, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 8_192, distortionFactor: 0.83)

    public static let iq3XXS = IQBlockDescriptor(
        ggmlType: .iq3XXS, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 1_152, distortionFactor: 0.83)

    public static let iq3S = IQBlockDescriptor(
        ggmlType: .iq3S, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 2_048, distortionFactor: 0.82)

    public static let iq4XS = IQBlockDescriptor(
        ggmlType: .iq4XS, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 64, distortionFactor: 0.82)

    public static let iq4NL = IQBlockDescriptor(
        ggmlType: .iq4NL, family: .iq, hasImportanceMatrixSupport: true, lutSizeBytes: 64, distortionFactor: 0.83)

    public static let q2K = IQBlockDescriptor(
        ggmlType: .q2K, family: .kQuant, hasImportanceMatrixSupport: true, lutSizeBytes: 0, distortionFactor: 1.04)

    public static let q3K = IQBlockDescriptor(
        ggmlType: .q3K, family: .kQuant, hasImportanceMatrixSupport: true, lutSizeBytes: 0, distortionFactor: 1.00)

    public static let q4K = IQBlockDescriptor(
        ggmlType: .q4K, family: .kQuant, hasImportanceMatrixSupport: true, lutSizeBytes: 0, distortionFactor: 0.96)

    public static let q5K = IQBlockDescriptor(
        ggmlType: .q5K, family: .kQuant, hasImportanceMatrixSupport: true, lutSizeBytes: 0, distortionFactor: 0.95)

    public static let q6K = IQBlockDescriptor(
        ggmlType: .q6K, family: .kQuant, hasImportanceMatrixSupport: true, lutSizeBytes: 0, distortionFactor: 0.94)

    public static let q8_0 = IQBlockDescriptor(
        ggmlType: .q8_0, family: .blockQuant, hasImportanceMatrixSupport: false, lutSizeBytes: 0, distortionFactor: 0.92)

    public static let iqFormats: [IQBlockDescriptor] = [
        .iq1S, .iq1M, .iq2XXS, .iq2XS, .iq2S, .iq3XXS, .iq3S, .iq4XS, .iq4NL,
    ]

    public static let kQuantAnchors: [IQBlockDescriptor] = [
        .q2K, .q3K, .q4K, .q5K, .q6K, .q8_0,
    ]

    public static let all: [IQBlockDescriptor] = [
        .iq1S, .iq1M, .iq2XXS, .iq2XS, .iq2S, .q2K, .iq3XXS, .iq3S, .q3K, .iq4XS, .iq4NL, .q4K, .q5K, .q6K, .q8_0,
    ]

    private static let byType: [GGMLType: IQBlockDescriptor] = Dictionary(
        uniqueKeysWithValues: all.map { ($0.ggmlType, $0) }
    )

    public static func descriptor(for type: GGMLType) -> IQBlockDescriptor? {
        byType[type]
    }
}
