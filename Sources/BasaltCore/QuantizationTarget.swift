import Foundation

public enum PVTunePreset: String, Sendable, CaseIterable, Codable {
    case fast
    case balanced
    case thorough

    public var calibrationBatches: Int {
        switch self {
        case .fast: 16
        case .balanced: 40
        case .thorough: 64
        }
    }

    public var globalContinuousSteps: Int {
        switch self {
        case .fast: 8
        case .balanced: 24
        case .thorough: 32
        }
    }

    public var maxLayerGroups: Int? {
        switch self {
        case .fast: 2
        case .balanced: 3
        case .thorough: nil
        }
    }
}

public enum QuantizationTarget: Sendable, Equatable, Codable {
    case denseBF16
    case denseFP16
    case affine(bits: Int, groupSize: Int)
    case mixed(lowBits: Int, highBits: Int, groupSize: Int)
    case mxfp4
    case nvfp4
    case nf4(groupSize: Int)
    case macaqAffine(bits: Int, groupSize: Int)
    case macaqMXFP4
    case macaqMoEBudget(bpw: Double)
    case flatQuant(bits: Int, groupSize: Int)
    case lirQuant(bits: Int, groupSize: Int, subspaceDim: Int)
    case qtip(bpw: Double)
    case qtipBudget(bpw: Double)
    case gsqRCO(bpw: Double)

    public var specString: String {
        switch self {
        case .denseBF16: "dense:bf16"
        case .denseFP16: "dense:f16"
        case .affine(let b, let g): "affine:\(b):\(g)"
        case .mixed(let lo, let hi, let g): "mixed:\(lo):\(hi):\(g)"
        case .mxfp4: "mxfp4"
        case .nvfp4: "nvfp4"
        case .nf4(let g): "nf4:\(g)"
        case .macaqAffine(let b, let g): "macaq:affine:\(b):\(g)"
        case .macaqMXFP4: "macaq:mxfp4"
        case .macaqMoEBudget(let bpw): "macaq:moe:\(Self.formatBPW(bpw))"
        case .flatQuant(let b, let g): "flatquant:\(b):\(g)"
        case .lirQuant(let b, let g, let k): "lirquant:\(b):\(g):\(k)"
        case .qtip(let bpw): String(format: "qtip:%.1f", bpw)
        case .qtipBudget(let bpw): String(format: "qtip-budget:%.2f", bpw)
        case .gsqRCO(let bpw): String(format: "gsq-rco:%.2f", bpw)
        }
    }

    public var suggestedTag: String {
        switch self {
        case .denseBF16: "BF16"
        case .denseFP16: "FP16"
        case .affine(let b, _): "\(b)bit"
        case .mixed(let lo, let hi, _): "Mixed-\(lo)-\(hi)bit"
        case .mxfp4: "MXFP4"
        case .nvfp4: "NVFP4"
        case .nf4(let g): "NF4-g\(g)"
        case .macaqAffine(let b, _): "MACaQ-\(b)bit"
        case .macaqMXFP4: "MACaQ-MXFP4"
        case .macaqMoEBudget(let bpw): "MACaQ-MoE-\(Self.formatBPW(bpw))bpw"
        case .flatQuant(let b, _): "FlatQuant-\(b)bit"
        case .lirQuant(let b, _, let k): "LIRQuant-\(b)bit-k\(k)"
        case .qtip(let bpw): String(format: "QTIP-%.1fbpw", bpw)
        case .qtipBudget(let bpw): String(format: "QTIP-%.2fbpw", bpw)
        case .gsqRCO(let bpw): String(format: "GSQ-%.2fbpw", bpw)
        }
    }

    public var isDense: Bool {
        switch self {
        case .denseBF16, .denseFP16: true
        default: false
        }
    }

    public var supportsPVTune: Bool {
        switch self {
        case .denseBF16, .denseFP16, .gsqRCO: false
        default: true
        }
    }

    public var effectiveBitsPerWeight: Double {
        switch self {
        case .denseBF16, .denseFP16: 16.0
        case .affine(let b, let g): Double(b) + 32.0 / Double(g)
        case .mixed(let lo, let hi, let g): Double(lo + hi) * 0.5 + 32.0 / Double(g)
        case .mxfp4: 4.25
        case .nvfp4: 4.5
        case .nf4(let g): 4.0 + 16.0 / Double(g)
        case .macaqAffine(let b, let g): Double(b) + 32.0 / Double(g) + 0.02
        case .macaqMXFP4: 4.27
        case .flatQuant(let b, let g): Double(b) + 32.0 / Double(g) + 0.02
        case .lirQuant(let b, let g, _): Double(b) + 32.0 / Double(g) + 0.03
        case .macaqMoEBudget(let bpw), .qtip(let bpw), .qtipBudget(let bpw), .gsqRCO(let bpw): bpw
        }
    }

    public static var standardCatalogSpecs: [String] {
        [
            "affine:2:64", "affine:3:64", "affine:4:64", "affine:6:64", "affine:8:64",
            "mixed:3:4:64", "mxfp4", "nvfp4", "nf4:64",
            "macaq:affine:2:64", "macaq:affine:3:64", "macaq:affine:4:64", "macaq:mxfp4", "macaq:moe:2.8",
            "flatquant:3:64", "flatquant:4:64", "lirquant:3:64:128", "lirquant:4:64:128",
            "qtip:2.0", "qtip:2.5", "qtip:3.0", "qtip-budget:2.75", "gsq-rco:3.0",
            "dense:bf16", "dense:f16",
        ]
    }
}

extension QuantizationTarget {
    private static func formatBPW(_ bpw: Double) -> String {
        abs(bpw * 10.0 - (bpw * 10.0).rounded()) < 1e-6
            ? String(format: "%.1f", bpw)
            : String(format: "%.2f", bpw)
    }
    public static func parse(_ raw: String) throws(BasaltError) -> QuantizationTarget {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let parts = trimmed.split(separator: ":").map(String.init)
        guard let head = parts.first else {
            throw .invalidRequestStructure(details: "convert: target specification must not be empty")
        }
        if let parsed = parseStandard(head: head, parts: parts)
            ?? parseBasisTransform(head: head, parts: parts)
            ?? parseCalibrated(head: head, parts: parts) {
            return parsed
        }
        throw .invalidRequestStructure(details: "convert: unsupported target '\(raw)'")
    }

    private static func parseBasisTransform(head: String, parts: [String]) -> QuantizationTarget? {
        switch head {
        case "flatquant" where parts.count == 3:
            guard let b = Int(parts[1]), let g = Int(parts[2]),
                  [2, 3, 4, 8].contains(b), [32, 64, 128].contains(g) else { return nil }
            return .flatQuant(bits: b, groupSize: g)
        case "lirquant" where parts.count == 3 || parts.count == 4:
            guard let b = Int(parts[1]), let g = Int(parts[2]),
                  [2, 3, 4, 8].contains(b), [32, 64, 128].contains(g) else { return nil }
            let k = parts.count == 4 ? Int(parts[3]) : 128
            guard let k, [32, 64, 128, 256].contains(k) else { return nil }
            return .lirQuant(bits: b, groupSize: g, subspaceDim: k)
        default:
            return nil
        }
    }

    private static func parseStandard(head: String, parts: [String]) -> QuantizationTarget? {
        switch head {
        case "dense" where parts.count == 2:
            if parts[1] == "bf16" || parts[1] == "bfloat16" { return .denseBF16 }
            if parts[1] == "f16" || parts[1] == "fp16" || parts[1] == "float16" { return .denseFP16 }
            return nil
        case "bf16" where parts.count == 1:
            return .denseBF16
        case "f16" where parts.count == 1, "fp16" where parts.count == 1:
            return .denseFP16
        case "mxfp4" where parts.count == 1:
            return .mxfp4
        case "nvfp4" where parts.count == 1:
            return .nvfp4
        case "nf4" where parts.count == 2:
            guard let g = Int(parts[1]), [32, 64, 128].contains(g) else { return nil }
            return .nf4(groupSize: g)
        case "affine" where parts.count == 3:
            guard let b = Int(parts[1]), let g = Int(parts[2]),
                  [2, 3, 4, 5, 6, 8].contains(b), [32, 64, 128].contains(g) else { return nil }
            return .affine(bits: b, groupSize: g)
        default:
            return nil
        }
    }

    private static func parseCalibrated(head: String, parts: [String]) -> QuantizationTarget? {
        switch head {
        case "mixed" where parts.count == 4:
            guard let lo = Int(parts[1]), let hi = Int(parts[2]), let g = Int(parts[3]),
                  [2, 3, 4, 5, 6, 8].contains(lo), [2, 3, 4, 5, 6, 8].contains(hi),
                  lo < hi, [32, 64, 128].contains(g) else { return nil }
            return .mixed(lowBits: lo, highBits: hi, groupSize: g)
        case "macaq":
            if parts.count == 2, parts[1] == "mxfp4" { return .macaqMXFP4 }
            if parts.count == 3, parts[1] == "moe",
               let bpw = Double(parts[2]), bpw.isFinite, (2.0 ... 4.5).contains(bpw) {
                return .macaqMoEBudget(bpw: bpw)
            }
            guard parts.count == 4, parts[1] == "affine",
                  let b = Int(parts[2]), let g = Int(parts[3]),
                  [2, 3, 4].contains(b), [32, 64, 128].contains(g) else { return nil }
            return .macaqAffine(bits: b, groupSize: g)
        case "qtip" where parts.count == 2:
            guard let bpw = Double(parts[1]), [2.0, 2.5, 3.0].contains(bpw) else { return nil }
            return .qtip(bpw: bpw)
        case "qtip-budget" where parts.count == 2:
            guard let bpw = Double(parts[1]), bpw.isFinite, (2.0 ... 4.0).contains(bpw) else { return nil }
            return .qtipBudget(bpw: bpw)
        case "gsq-rco" where parts.count == 2:
            guard let bpw = Double(parts[1]), bpw.isFinite, (1.5625 ... 8.5).contains(bpw) else { return nil }
            return .gsqRCO(bpw: bpw)
        default:
            return nil
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        self = try Self.parse(raw)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(specString)
    }
}
