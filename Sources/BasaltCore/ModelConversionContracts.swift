import Foundation

public struct ModelConversionRequest: Sendable, Equatable, Codable {
    public var source: String
    public var outputId: String
    public var target: QuantizationTarget
    public var recipe: String? = nil
    public var hadamardBlock: Int? = nil
    public var alpha: Float? = nil
    public var budgetGB: Double? = nil
    public var adapter: String? = nil
    public var macaqRefine: Bool = false
    public var pvTune: Bool = false
    public var pvPreset: PVTunePreset = .balanced
    public var pvTeacher: String? = nil
    public var calibrationText: String? = nil
    public var clusterPeers: [String] = []
    public var qatRefine: Bool = false
    public var auditSuitePath: String? = nil
    public var stream: Bool = true

    public init(
        source: String,
        outputId: String,
        target: QuantizationTarget,
        recipe: String? = nil,
        pvTune: Bool = false,
        stream: Bool = true
    ) {
        self.source = source
        self.outputId = outputId
        self.target = target
        self.recipe = recipe
        self.pvTune = pvTune
        self.stream = stream
    }

    enum CodingKeys: String, CodingKey {
        case source
        case outputId = "output_id"
        case target, recipe, alpha, adapter, stream
        case hadamardBlock = "hadamard_block"
        case budgetGB = "budget_gb"
        case macaqRefine = "macaq_refine"
        case pvTune = "pv_tune"
        case pvPreset = "pv_preset"
        case pvTeacher = "pv_teacher"
        case calibrationText = "calibration_text"
        case clusterPeers = "cluster_peers"
        case qatRefine = "qat_refine"
        case auditSuitePath = "audit_suite_path"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        source = try c.decode(String.self, forKey: .source)
        outputId = try c.decode(String.self, forKey: .outputId)
        target = try c.decode(QuantizationTarget.self, forKey: .target)
        recipe = try c.decodeIfPresent(String.self, forKey: .recipe)
        hadamardBlock = try c.decodeIfPresent(Int.self, forKey: .hadamardBlock)
        alpha = try c.decodeIfPresent(Float.self, forKey: .alpha)
        budgetGB = try c.decodeIfPresent(Double.self, forKey: .budgetGB)
        adapter = try c.decodeIfPresent(String.self, forKey: .adapter)
        macaqRefine = try c.decodeIfPresent(Bool.self, forKey: .macaqRefine) ?? false
        pvTune = try c.decodeIfPresent(Bool.self, forKey: .pvTune) ?? false
        pvPreset = try c.decodeIfPresent(PVTunePreset.self, forKey: .pvPreset) ?? .balanced
        pvTeacher = try c.decodeIfPresent(String.self, forKey: .pvTeacher)
        calibrationText = try c.decodeIfPresent(String.self, forKey: .calibrationText)
        clusterPeers = try c.decodeIfPresent([String].self, forKey: .clusterPeers) ?? []
        qatRefine = try c.decodeIfPresent(Bool.self, forKey: .qatRefine) ?? false
        auditSuitePath = try c.decodeIfPresent(String.self, forKey: .auditSuitePath)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream) ?? true
    }
}

public struct ModelConversionPlan: Sendable, Equatable, Codable {
    public struct MemoryFootprint: Sendable, Equatable {
        public var sourceBytes: Int64
        public var estimatedOutputBytes: Int64
        public var peakConversionRAMBytes: Int64
        public var availableRAMBytes: Int64
        public var fitsConversionOnThisMac: Bool
        public var outputFitsInference: Bool

        public init(
            sourceBytes: Int64 = 0,
            estimatedOutputBytes: Int64 = 0,
            peakConversionRAMBytes: Int64 = 0,
            availableRAMBytes: Int64 = 0,
            fitsConversionOnThisMac: Bool = true,
            outputFitsInference: Bool = true
        ) {
            self.sourceBytes = sourceBytes
            self.estimatedOutputBytes = estimatedOutputBytes
            self.peakConversionRAMBytes = peakConversionRAMBytes
            self.availableRAMBytes = availableRAMBytes
            self.fitsConversionOnThisMac = fitsConversionOnThisMac
            self.outputFitsInference = outputFitsInference
        }
    }

    public var sourceId: String
    public var sourceFormat: String
    public var sourceBytes: Int64
    public var target: String
    public var estimatedOutputBytes: Int64
    public var peakConversionRAMBytes: Int64
    public var availableRAMBytes: Int64
    public var fitsConversionOnThisMac: Bool
    public var outputFitsInference: Bool
    public var supportsPVTune: Bool
    public var supportedTargets: [String]
    public var warnings: [String]

    public init(
        sourceId: String,
        sourceFormat: String,
        target: String,
        supportsPVTune: Bool,
        footprint: MemoryFootprint = .init(),
        warnings: [String] = []
    ) {
        self.sourceId = sourceId
        self.sourceFormat = sourceFormat
        self.sourceBytes = footprint.sourceBytes
        self.target = target
        self.estimatedOutputBytes = footprint.estimatedOutputBytes
        self.peakConversionRAMBytes = footprint.peakConversionRAMBytes
        self.availableRAMBytes = footprint.availableRAMBytes
        self.fitsConversionOnThisMac = footprint.fitsConversionOnThisMac
        self.outputFitsInference = footprint.outputFitsInference
        self.supportsPVTune = supportsPVTune
        self.supportedTargets = QuantizationTarget.standardCatalogSpecs
        self.warnings = warnings
    }

    enum CodingKeys: String, CodingKey {
        case sourceId = "source_id"
        case sourceFormat = "source_format"
        case sourceBytes = "source_bytes"
        case target
        case estimatedOutputBytes = "estimated_output_bytes"
        case peakConversionRAMBytes = "peak_conversion_ram_bytes"
        case availableRAMBytes = "available_ram_bytes"
        case fitsConversionOnThisMac = "fits_conversion_on_this_mac"
        case outputFitsInference = "output_fits_inference"
        case supportsPVTune = "supports_pv_tune"
        case supportedTargets = "supported_targets"
        case warnings
    }
}

public struct ModelConversionProgress: Sendable, Equatable, Codable {
    public var phase: String
    public var completed: Int
    public var total: Int
    public var message: String
    public var valKLBefore: Double?
    public var valKLCurrent: Double?

    public init(
        phase: String,
        completed: Int,
        total: Int,
        message: String,
        valKLBefore: Double? = nil,
        valKLCurrent: Double? = nil
    ) {
        self.phase = phase
        self.completed = completed
        self.total = total
        self.message = message
        self.valKLBefore = valKLBefore
        self.valKLCurrent = valKLCurrent
    }

    enum CodingKeys: String, CodingKey {
        case phase, completed, total, message
        case valKLBefore = "val_kl_before"
        case valKLCurrent = "val_kl_current"
    }
}

public struct ModelConversionPVSummary: Sendable, Equatable, Codable {
    public var valKLBefore: Double
    public var valKLAfter: Double
    public var updatedCodes: Int

    public init(valKLBefore: Double, valKLAfter: Double, updatedCodes: Int) {
        self.valKLBefore = valKLBefore
        self.valKLAfter = valKLAfter
        self.updatedCodes = updatedCodes
    }

    enum CodingKeys: String, CodingKey {
        case valKLBefore = "val_kl_before"
        case valKLAfter = "val_kl_after"
        case updatedCodes = "updated_codes"
    }
}

public struct ModelConversionResult: Sendable, Equatable, Codable {
    public var outputId: String
    public var bytesBefore: Int64
    public var bytesAfter: Int64
    public var shards: Int
    public var pvTune: ModelConversionPVSummary?
    public var audit: QuantizationAuditVerdict?

    public init(
        outputId: String,
        bytesBefore: Int64,
        bytesAfter: Int64,
        shards: Int,
        pvTune: ModelConversionPVSummary? = nil,
        audit: QuantizationAuditVerdict? = nil
    ) {
        self.outputId = outputId
        self.bytesBefore = bytesBefore
        self.bytesAfter = bytesAfter
        self.shards = shards
        self.pvTune = pvTune
        self.audit = audit
    }

    enum CodingKeys: String, CodingKey {
        case outputId = "output_id"
        case bytesBefore = "bytes_before"
        case bytesAfter = "bytes_after"
        case shards, audit
        case pvTune = "pv_tune"
    }
}

public protocol ModelConversionService: Sendable {
    func plan(
        source: String,
        target: QuantizationTarget,
        budgetGB: Double?,
        pvTune: Bool,
        cacheRoot: URL
    ) async throws -> ModelConversionPlan

    func convert(
        _ request: ModelConversionRequest,
        cacheRoot: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionResult
}
