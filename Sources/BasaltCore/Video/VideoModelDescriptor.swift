import Foundation

public enum VideoArchitectureFamily: String, Sendable, Codable, Equatable, CaseIterable {
    case ltx25AV = "ltx_2_5_av"
    case kandinskyCrossDiT = "kandinsky_cross_dit"
    case moeVideoDiT = "moe_video_dit"
}

public enum VideoExecutionStrategy24GB: String, Sendable, Codable, Equatable {
    case resident4Bit = "resident_4bit"
    case expertStreamingSSD = "expert_streaming_ssd"
    case cascadeLiteToCloudPro = "cascade_lite_to_cloud_pro"
}

public struct VideoModelDescriptor: Sendable, Codable, Equatable {
    public var id: String
    public var family: VideoArchitectureFamily
    public var licenseName: String
    public var activeBillionParams: Double
    public var totalBillionParams: Double
    public var supportsJointAudio: Bool
    public var supportsExpertStreaming: Bool
    public var defaultSteps: Int
    public var defaultGuidanceScale: Double
    public var defaultFPS: Int
    public var strategy24GB: VideoExecutionStrategy24GB

    public init(
        id: String,
        family: VideoArchitectureFamily,
        licenseName: String,
        activeBillionParams: Double,
        totalBillionParams: Double? = nil
    ) {
        self.id = id
        self.family = family
        self.licenseName = licenseName
        self.activeBillionParams = activeBillionParams
        self.totalBillionParams = totalBillionParams ?? activeBillionParams
        self.supportsJointAudio = true
        self.supportsExpertStreaming = false
        self.defaultSteps = 8
        self.defaultGuidanceScale = 1.0
        self.defaultFPS = 24
        self.strategy24GB = .resident4Bit
    }
}

extension VideoModelDescriptor {
    public static func ltx25Distilled(id: String = "Lightricks/LTX-2.5-22B-Distilled") -> VideoModelDescriptor {
        var d = VideoModelDescriptor(
            id: id,
            family: .ltx25AV,
            licenseName: "LTX Open",
            activeBillionParams: 22.0
        )
        d.supportsJointAudio = true
        d.defaultSteps = 8
        d.defaultGuidanceScale = 1.0
        d.strategy24GB = .resident4Bit
        return d
    }

    public static func kandinsky60Lite3B(id: String = "ai-forever/Kandinsky-6.0-3B-Lite") -> VideoModelDescriptor {
        var d = VideoModelDescriptor(
            id: id,
            family: .kandinskyCrossDiT,
            licenseName: "MIT",
            activeBillionParams: 3.0
        )
        d.supportsJointAudio = true
        d.defaultSteps = 12
        d.defaultGuidanceScale = 3.5
        d.strategy24GB = .resident4Bit
        return d
    }

    public static func kandinsky60Pro29B(id: String = "ai-forever/Kandinsky-6.0-29B-Pro") -> VideoModelDescriptor {
        var d = VideoModelDescriptor(
            id: id,
            family: .kandinskyCrossDiT,
            licenseName: "MIT",
            activeBillionParams: 29.0
        )
        d.supportsJointAudio = true
        d.defaultSteps = 20
        d.defaultGuidanceScale = 4.0
        d.strategy24GB = .cascadeLiteToCloudPro
        return d
    }

    public static func lingBotVideo30BA3B(id: String = "inclusionAI/LingBot-Video-30B-A3B") -> VideoModelDescriptor {
        var d = VideoModelDescriptor(
            id: id,
            family: .moeVideoDiT,
            licenseName: "Apache-2.0",
            activeBillionParams: 3.0,
            totalBillionParams: 30.0
        )
        d.supportsJointAudio = true
        d.supportsExpertStreaming = true
        d.defaultSteps = 16
        d.defaultGuidanceScale = 3.0
        d.strategy24GB = .expertStreamingSSD
        return d
    }

    public static func wan22A14B(id: String = "Wan-Video/Wan2.2-A14B") -> VideoModelDescriptor {
        var d = VideoModelDescriptor(
            id: id,
            family: .moeVideoDiT,
            licenseName: "Apache-2.0",
            activeBillionParams: 14.0,
            totalBillionParams: 27.0
        )
        d.supportsJointAudio = false
        d.supportsExpertStreaming = true
        d.defaultSteps = 20
        d.defaultGuidanceScale = 3.5
        d.strategy24GB = .expertStreamingSSD
        return d
    }

    public static func detect(identifier: String, configJSON: [String: Any]? = nil) -> VideoModelDescriptor {
        let modelType = (configJSON?["model_type"] as? String)?.lowercased() ?? ""
        let lower = "\(identifier.lowercased()) \(modelType)"
        if lower.contains("kandinsky") {
            return (lower.contains("29b") || lower.contains("pro"))
                ? kandinsky60Pro29B(id: identifier)
                : kandinsky60Lite3B(id: identifier)
        }
        if lower.contains("lingbot") || lower.contains("moe_video") {
            return lingBotVideo30BA3B(id: identifier)
        }
        if lower.contains("wan") {
            return wan22A14B(id: identifier)
        }
        return ltx25Distilled(id: identifier)
    }

    public static func selectKandinskyCascade(availableRAMBytes: Int64) -> VideoModelDescriptor {
        availableRAMBytes >= 42_000_000_000 ? kandinsky60Pro29B() : kandinsky60Lite3B()
    }
}
