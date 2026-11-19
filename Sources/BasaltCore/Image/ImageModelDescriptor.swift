import Foundation

public enum ImageArchitectureFamily: String, Sendable, Codable, Equatable, CaseIterable {
    case fluxKlein = "flux_klein"
    case flux3Klein = "flux3_klein"
    case s3DiT = "s3_dit"
    case pixelDiT = "pixel_dit"
    case linearSana = "linear_sana"
    case ternaryBitNet = "ternary_bitnet"
    case fiboPerceiver = "fibo_perceiver"
    case step1XReasonEdit = "step1x_reason_edit"
    case unifiedMoT = "unified_mot"
    case audioToImage = "audio_to_image"
    case seedVR2Upscaler = "seedvr2_upscaler"
}

public struct ImageMemoryFootprint: Sendable, Codable, Equatable {
    public var generatorBytes: Int64
    public var textEncoderBytes: Int64
    public var vaeBytes: Int64
    public var workingLatentBytes: Int64
    public var sharedTextEncoderReused: Bool
    public var sequentialEncoderOffload: Bool
    public var qualityWarning: String?

    public init(
        generatorBytes: Int64,
        textEncoderBytes: Int64,
        vaeBytes: Int64 = 0,
        workingLatentBytes: Int64 = 0
    ) {
        self.generatorBytes = generatorBytes
        self.textEncoderBytes = textEncoderBytes
        self.vaeBytes = vaeBytes
        self.workingLatentBytes = workingLatentBytes
        self.sharedTextEncoderReused = false
        self.sequentialEncoderOffload = false
        self.qualityWarning = nil
    }

    public var peakResidentBytes: Int64 {
        let effectiveTE: Int64 = sharedTextEncoderReused ? 0 : textEncoderBytes
        if sequentialEncoderOffload {
            return max(effectiveTE, generatorBytes + vaeBytes + workingLatentBytes)
        }
        return generatorBytes + effectiveTE + vaeBytes + workingLatentBytes
    }
}

public struct ImageModelDescriptor: Sendable, Codable, Equatable {
    public var id: String
    public var family: ImageArchitectureFamily
    public var licenseName: String
    public var licenseTier: ImageLicenseTier
    public var generatorBillionParams: Double
    public var textEncoderFamily: String
    public var textEncoderBillionParams: Double
    public var usesVAE: Bool
    public var supportsNativeRGBA: Bool
    public var supportsMultiReferenceEdit: Bool
    public var supportsReferenceKVCache: Bool
    public var maxReferences: Int
    public var supportsBBoxLayout: Bool
    public var supportsAudioConditioning: Bool
    public var defaultSteps: Int
    public var minimumRecommendedBits: Int
    public var defaultCompute: ImageComputeBackend

    public init(
        id: String,
        family: ImageArchitectureFamily,
        licenseName: String,
        licenseTier: ImageLicenseTier,
        generatorBillionParams: Double,
        textEncoderFamily: String = "qwen3"
    ) {
        self.id = id
        self.family = family
        self.licenseName = licenseName
        self.licenseTier = licenseTier
        self.generatorBillionParams = generatorBillionParams
        self.textEncoderFamily = textEncoderFamily
        self.textEncoderBillionParams = 4.0
        self.usesVAE = true
        self.supportsNativeRGBA = false
        self.supportsMultiReferenceEdit = false
        self.supportsReferenceKVCache = false
        self.maxReferences = 0
        self.supportsBBoxLayout = false
        self.supportsAudioConditioning = false
        self.defaultSteps = 8
        self.minimumRecommendedBits = 4
        self.defaultCompute = .mlx
    }
}

extension ImageModelDescriptor {
    private struct DetectionRule: Sendable {
        let allOf: [String]
        let anyOf: [String]
        let build: @Sendable (String) -> ImageModelDescriptor
    }

    private static let detectionRules: [DetectionRule] = [
        DetectionRule(allOf: ["base"], anyOf: ["flux-3", "flux.3", "flux3", "flux3transformer"], build: flux3KleinBase(id:)),
        DetectionRule(allOf: [], anyOf: ["flux-3", "flux.3", "flux3", "flux3transformer"], build: flux3Klein(id:)),
        DetectionRule(allOf: ["klein", "9b", "kv"], anyOf: [], build: flux2Klein9BKV(id:)),
        DetectionRule(allOf: ["klein", "9b"], anyOf: [], build: flux2Klein9BNonCommercial(id:)),
        DetectionRule(allOf: [], anyOf: ["klein", "flux.2", "flux2"], build: flux2Klein4B(id:)),
        DetectionRule(allOf: [], anyOf: ["seedvr2", "seed_vr2"], build: seedVR2_3B(id:)),
        DetectionRule(allOf: [], anyOf: ["iris-3b", "iris_3b", "pixel_dit"], build: iris3BTurbo(id:)),
        DetectionRule(allOf: [], anyOf: ["hidream"], build: hiDreamO1(id:)),
        DetectionRule(allOf: [], anyOf: ["sana"], build: sana15(id:)),
        DetectionRule(allOf: [], anyOf: ["bonsai"], build: bonsaiTernary4B(id:)),
        DetectionRule(allOf: [], anyOf: ["moss-image", "moss_image"], build: mossImage(id:)),
        DetectionRule(allOf: [], anyOf: ["ming-image"], build: mingImageDesign(id:)),
        DetectionRule(allOf: [], anyOf: ["qwen-image-2.1", "qwen_image_2_1"], build: qwenImage21(id:)),
        DetectionRule(allOf: [], anyOf: ["fibo"], build: fibo2(id:)),
        DetectionRule(allOf: [], anyOf: ["step1x-edit", "reasonedit"], build: step1XEdit(id:)),
        DetectionRule(allOf: [], anyOf: ["sensenova", "lance"], build: unifiedMoT(id:)),
        DetectionRule(allOf: [], anyOf: ["sonicdiffusion", "audiocanvas", "ming-lite-omni"], build: audioToImage(id:)),
    ]

    public static func detect(identifier: String, configJSON: [String: Any]? = nil) -> ImageModelDescriptor {
        let arch = (configJSON?["model_type"] as? String)?.lowercased() ?? ""
        let cls = (configJSON?["_class_name"] as? String)?.lowercased() ?? ""
        let key = "\(identifier.lowercased()) \(arch) \(cls)"
        for rule in detectionRules {
            let matchesAll = rule.allOf.allSatisfy { key.contains($0) }
            let matchesAny = rule.anyOf.isEmpty || rule.anyOf.contains { key.contains($0) }
            if matchesAll && matchesAny {
                return rule.build(identifier)
            }
        }
        return zImageTurbo(id: identifier)
    }

    public func estimateFootprint(
        quantizationBits: Int = 8,
        dimensions: ImageDimensions = ImageDimensions(),
        sharedTextEncoderReused: Bool = false,
        sequentialEncoderOffload: Bool = false
    ) -> ImageMemoryFootprint {
        let effectiveBits = family == .ternaryBitNet ? 2 : max(2, quantizationBits)
        let bytesPerParam = Double(effectiveBits) / 8.0 * 1.08
        let genBytes = Int64(generatorBillionParams * 1_000_000_000.0 * bytesPerParam)
        let teBytes = Int64(textEncoderBillionParams * 1_000_000_000.0 * (Double(max(4, quantizationBits)) / 8.0) * 1.06)
        let vaeBytes: Int64 = usesVAE ? 360_000_000 : 0
        let pixels = Int64(dimensions.width) * Int64(dimensions.height)
        let latentBytes = usesVAE ? pixels * 8 : pixels * 24
        var footprint = ImageMemoryFootprint(
            generatorBytes: genBytes, textEncoderBytes: teBytes, vaeBytes: vaeBytes, workingLatentBytes: latentBytes
        )
        footprint.sharedTextEncoderReused = sharedTextEncoderReused
        footprint.sequentialEncoderOffload = sequentialEncoderOffload
        if quantizationBits < minimumRecommendedBits {
            footprint.qualityWarning = "\(id) recommends at least \(minimumRecommendedBits)-bit weights; \(quantizationBits)-bit degrades fine skin/text detail"
        }
        return footprint
    }
}

extension ImageModelDescriptor {
    public static func flux2Klein4B(id: String = "black-forest-labs/FLUX.2-klein-4B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .fluxKlein, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 4.0, textEncoderFamily: "qwen3"
        )
        d.supportsMultiReferenceEdit = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 6
        d.defaultSteps = 4
        return d
    }

    public static func flux2Klein9BNonCommercial(id: String = "black-forest-labs/FLUX.2-klein-9B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .fluxKlein, licenseName: "FLUX Non-Commercial",
            licenseTier: .nonCommercialResearch, generatorBillionParams: 9.0, textEncoderFamily: "qwen3"
        )
        d.supportsMultiReferenceEdit = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 8
        return d
    }

    public static func flux2Klein9BKV(id: String = "black-forest-labs/FLUX.2-klein-9B-kv") -> ImageModelDescriptor {
        var d = flux2Klein9BNonCommercial(id: id)
        d.supportsNativeRGBA = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 10
        d.defaultSteps = 4
        return d
    }

    public static func flux3Klein(id: String = "black-forest-labs/FLUX.3-Image-Klein") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .flux3Klein, licenseName: "FLUX Kommunity (<$5M/yr)",
            licenseTier: .revenueCappedCommercial, generatorBillionParams: 5.55, textEncoderFamily: "qwen3_vl"
        )
        d.textEncoderBillionParams = 4.4
        d.supportsNativeRGBA = true
        d.supportsMultiReferenceEdit = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 10
        d.supportsBBoxLayout = true
        d.defaultSteps = 16
        let lower = id.lowercased()
        d.minimumRecommendedBits = (lower.contains("macaq") || lower.contains("mxfp4")) ? 4 : 8
        return d
    }

    public static func flux3KleinBase(id: String = "black-forest-labs/flux-3-image-klein-base") -> ImageModelDescriptor {
        var d = flux3Klein(id: id)
        d.defaultSteps = 50
        return d
    }

    public static func zImageTurbo(id: String = "Tongyi-MAI/Z-Image-Turbo") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .s3DiT, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 6.0, textEncoderFamily: "qwen3"
        )
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 4
        return d
    }

    public static func iris3BTurbo(id: String = "Speridlabs/Iris-3B-Turbo") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .pixelDiT, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 3.0, textEncoderFamily: "qwen3_vl"
        )
        d.usesVAE = false
        d.defaultSteps = 4
        return d
    }

    public static func hiDreamO1(id: String = "HiDream-ai/HiDream-O1-Image") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .pixelDiT, licenseName: "MIT",
            licenseTier: .permissiveCommercial, generatorBillionParams: 8.0, textEncoderFamily: "qwen3_vl"
        )
        d.usesVAE = false
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 4
        d.supportsBBoxLayout = true
        d.defaultSteps = 12
        return d
    }

    public static func sana15(id: String = "Efficient-Large-Model/SANA-1.5-1.6B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .linearSana, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 1.6, textEncoderFamily: "gemma"
        )
        d.textEncoderBillionParams = 2.0
        d.defaultSteps = 4
        d.defaultCompute = .mlxANEHybrid
        return d
    }

    public static func bonsaiTernary4B(id: String = "Bonsai/Bonsai-Image-4B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .ternaryBitNet, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 4.0, textEncoderFamily: "qwen3"
        )
        d.textEncoderBillionParams = 1.7
        d.defaultSteps = 4
        d.minimumRecommendedBits = 2
        return d
    }

    public static func mossImage(id: String = "OpenMOSS/MOSS-Image-4.9B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .s3DiT, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 4.9, textEncoderFamily: "qwen3_5"
        )
        d.textEncoderBillionParams = 9.0
        d.supportsBBoxLayout = true
        return d
    }

    public static func mingImageDesign(id: String = "inclusionAI/Ming-Image-0.1-Design") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .s3DiT, licenseName: "MIT",
            licenseTier: .permissiveCommercial, generatorBillionParams: 6.0, textEncoderFamily: "qwen3_vl"
        )
        d.supportsNativeRGBA = true
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 4
        d.supportsBBoxLayout = true
        d.defaultSteps = 12
        d.minimumRecommendedBits = 6
        return d
    }
}

extension ImageModelDescriptor {
    public static func qwenImage21(id: String = "Qwen/Qwen-Image-2.1-Turbo") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .s3DiT, licenseName: "Qwen Research Non-Commercial",
            licenseTier: .nonCommercialResearch, generatorBillionParams: 7.1, textEncoderFamily: "qwen3_vl"
        )
        d.textEncoderBillionParams = 8.0
        d.supportsNativeRGBA = true
        d.supportsMultiReferenceEdit = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 10
        d.supportsBBoxLayout = true
        return d
    }

    public static func fibo2(id: String = "briaai/FIBO-2.0-Edit-Turbo") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .fiboPerceiver, licenseName: "CC BY-NC 4.0",
            licenseTier: .nonCommercialResearch, generatorBillionParams: 8.1, textEncoderFamily: "qwen3_vl"
        )
        d.supportsNativeRGBA = true
        d.supportsMultiReferenceEdit = true
        d.supportsReferenceKVCache = true
        d.maxReferences = 10
        d.supportsBBoxLayout = true
        d.defaultSteps = 4
        d.minimumRecommendedBits = 8
        return d
    }

    public static func seedVR2_3B(id: String = "ByteDance-Seed/SeedVR2-3B") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .seedVR2Upscaler, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 3.0, textEncoderFamily: "none"
        )
        d.textEncoderBillionParams = 0.0
        d.supportsNativeRGBA = true
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 1
        d.defaultSteps = 1
        d.minimumRecommendedBits = 8
        return d
    }

    public static func step1XEdit(id: String = "stepfun-ai/Step1X-Edit-v1p2") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .step1XReasonEdit, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 8.0, textEncoderFamily: "qwen3_vl"
        )
        d.textEncoderBillionParams = 7.0
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 4
        d.defaultSteps = 20
        d.minimumRecommendedBits = 8
        return d
    }

    public static func unifiedMoT(id: String = "SenseTime/SenseNova-U1.5-8B-MoT") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .unifiedMoT, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 3.0, textEncoderFamily: "mot_endogenous"
        )
        d.textEncoderBillionParams = 0.0
        d.usesVAE = false
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 4
        d.supportsBBoxLayout = true
        return d
    }

    public static func audioToImage(id: String = "SonicDiffusion/SonicDiffusion-SD3") -> ImageModelDescriptor {
        var d = ImageModelDescriptor(
            id: id, family: .audioToImage, licenseName: "Apache-2.0",
            licenseTier: .permissiveCommercial, generatorBillionParams: 2.8, textEncoderFamily: "clap_omni"
        )
        d.textEncoderBillionParams = 0.3
        d.supportsMultiReferenceEdit = true
        d.maxReferences = 2
        d.supportsAudioConditioning = true
        return d
    }
}
