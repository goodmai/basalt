import BasaltCore
import Foundation
import MLX

package struct UniversalModelConverter: ModelConversionService, Sendable {
    package init() {}

    package static func resolveDirectory(_ spec: String, cacheRoot: URL = ModelCache.root) throws(BasaltError) -> URL {
        let expanded = (spec as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: expanded) {
            return URL(fileURLWithPath: expanded)
        }
        if !spec.hasPrefix("/"), !spec.hasPrefix("."), !spec.hasPrefix("~") {
            let folder = cacheRoot.appending(path: "models--" + spec.replacingOccurrences(of: "/", with: "--"))
            let snaps = folder.appending(component: "snapshots")
            if let names = try? FileManager.default.contentsOfDirectory(atPath: snaps.path).filter({ !$0.hasPrefix(".") }).sorted(),
               let last = names.last {
                return snaps.appending(component: last)
            }
            if let resolved = ModelCache.resolve(repoId: spec) {
                return resolved
            }
        }
        throw .modelNotFound(identifier: spec)
    }

    package static func cacheSnapshotDirectory(for repoId: String, cacheRoot: URL = ModelCache.root) throws(BasaltError) -> URL {
        let parts = repoId.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let valid = parts.count == 2 && parts.allSatisfy { part in
            !part.isEmpty && part != "." && part != ".." && part.unicodeScalars.allSatisfy(allowed.contains)
        }
        guard valid else {
            throw .invalidRequestStructure(details: "convert: output_id '\(repoId)' must be org/name")
        }
        return cacheRoot
            .appending(path: "models--" + repoId.replacingOccurrences(of: "/", with: "--"))
            .appending(component: "snapshots")
            .appending(component: "local")
    }

    package func plan(
        source: String,
        target: QuantizationTarget,
        budgetGB: Double?,
        pvTune: Bool,
        cacheRoot: URL = ModelCache.root
    ) async throws -> ModelConversionPlan {
        let srcURL = try Self.resolveDirectory(source, cacheRoot: cacheRoot)
        let detected = try UniversalSourceReader.detect(at: srcURL)
        let srcBPW = max(1.0, detected.effectiveBPW)
        let estBytes: Int64
        if case .mixed = target, let budgetGB, budgetGB > 0 {
            estBytes = Int64(budgetGB * 1_000_000_000)
        } else {
            let ratio = target.effectiveBitsPerWeight / srcBPW
            estBytes = max(1_048_576, Int64((Double(detected.totalBytes) * ratio).rounded()))
        }
        let resources = await SystemProfiler().detectResources()
        let availBytes = resources.availableRAM
        let peakFactor: Double = pvTune ? 2.1 : (detected.requiresCanonicalDecode ? 1.7 : 1.35)
        let peakRAM = max(estBytes, Int64(Double(max(detected.totalBytes, estBytes)) * peakFactor))
        let fitsConv = peakRAM <= Int64(Double(resources.totalRAM) * 0.88)
        let fitsInf = estBytes <= Int64(Double(resources.totalRAM) * 0.80)
        var warnings: [String] = []
        if pvTune && !target.supportsPVTune {
            warnings.append("PV-Tuning is skipped for target \(target.specString)")
        }
        if !fitsConv {
            warnings.append("Peak conversion RAM exceeds safe Unified Memory headroom on this Mac")
        }
        let footprint = ModelConversionPlan.MemoryFootprint(
            sourceBytes: detected.totalBytes,
            estimatedOutputBytes: estBytes,
            peakConversionRAMBytes: peakRAM,
            availableRAMBytes: availBytes,
            fitsConversionOnThisMac: fitsConv,
            outputFitsInference: fitsInf)
        return ModelConversionPlan(
            sourceId: source,
            sourceFormat: detected.format,
            target: target.specString,
            supportsPVTune: target.supportsPVTune,
            footprint: footprint,
            warnings: warnings)
    }

    package func convert(
        _ request: ModelConversionRequest,
        cacheRoot: URL = ModelCache.root,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionResult {
        let srcURL = try Self.resolveDirectory(request.source, cacheRoot: cacheRoot)
        let outURL = try Self.cacheSnapshotDirectory(for: request.outputId, cacheRoot: cacheRoot)
        if FileManager.default.fileExists(atPath: outURL.path) {
            throw BasaltError.invalidRequestStructure(details: "convert: output '\(request.outputId)' already exists")
        }
        try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let refsDir = outURL.deletingLastPathComponent().deletingLastPathComponent().appending(component: "refs")
        let result = try await Self.convertToDirectory(
            request: request, sourceURL: srcURL, outputURL: outURL, cacheRoot: cacheRoot, onProgress: onProgress)
        try? FileManager.default.createDirectory(at: refsDir, withIntermediateDirectories: true)
        try? Data("local\n".utf8).write(to: refsDir.appending(component: "main"))
        return result
    }
}

extension UniversalModelConverter {
    package static func convertToDirectory(
        request: ModelConversionRequest,
        sourceURL: URL,
        outputURL: URL,
        cacheRoot: URL = ModelCache.root,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionResult {
        let detected = try UniversalSourceReader.detect(at: sourceURL)
        let validatedPeers = try validateClusterPeers(request.clusterPeers)
        if !validatedPeers.isEmpty {
            onProgress(.init(phase: "cluster", completed: 0, total: 4, message: "Validated \(validatedPeers.count) cluster peer(s) for distributed calibration"))
        }
        let teacherURL = try request.pvTeacher.map { try resolveDirectory($0, cacheRoot: cacheRoot) } ?? sourceURL
        let existedBefore = FileManager.default.fileExists(atPath: outputURL.path)
        var stagingURL: URL?
        defer {
            if let stagingURL {
                try? FileManager.default.removeItem(at: stagingURL)
            }
        }

        do {
            if request.target.isDense {
                return try await exportDenseTarget(
                    request: request, detected: detected, sourceURL: sourceURL, outputURL: outputURL, onProgress: onProgress)
            }
            return try await runQuantizedPipeline(
                request: request, detected: detected, sourceURL: sourceURL, outputURL: outputURL,
                teacherURL: teacherURL, stagingURL: &stagingURL, onProgress: onProgress)
        } catch {
            if !existedBefore, outputURL != sourceURL {
                try? FileManager.default.removeItem(at: outputURL)
            }
            throw error
        }
    }

    private static func exportDenseTarget(
        request: ModelConversionRequest,
        detected: DetectedSourceCheckpoint,
        sourceURL: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionResult {
        onProgress(.init(phase: "decoding", completed: 1, total: 2, message: "Decoding \(detected.format) to \(request.target.specString)"))
        let dt: DType = request.target == .denseFP16 ? .float16 : .bfloat16
        let summary = try await UniversalSourceReader.exportCanonicalDense(from: sourceURL, to: outputURL, dtype: dt) { msg in
            onProgress(.init(phase: "writing", completed: 1, total: 2, message: msg))
        }
        onProgress(.init(phase: "writing", completed: 2, total: 2, message: "Completed dense export"))
        return ModelConversionResult(
            outputId: request.outputId,
            bytesBefore: Int64(summary.bytesBefore),
            bytesAfter: Int64(summary.bytesAfter),
            shards: summary.shards)
    }

    private static func runQuantizedPipeline(
        request: ModelConversionRequest,
        detected: DetectedSourceCheckpoint,
        sourceURL: URL,
        outputURL: URL,
        teacherURL: URL,
        stagingURL: inout URL?,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionResult {
        let workingSource: URL
        if needsCanonicalStaging(detected: detected, target: request.target) {
            onProgress(.init(phase: "decoding", completed: 1, total: 4, message: "Unrotating & decoding \(detected.format) source"))
            let tmp = FileManager.default.temporaryDirectory.appending(path: "basalt-canon-\(UUID().uuidString)")
            stagingURL = tmp
            _ = try await UniversalSourceReader.exportCanonicalDense(from: sourceURL, to: tmp, dtype: .bfloat16)
            workingSource = tmp
        } else {
            workingSource = sourceURL
        }
        onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: "Quantizing to \(request.target.specString)"))
        let encoded = try await encodeTarget(
            request: request, workingSource: workingSource, outputURL: outputURL, onProgress: onProgress)
        let pvOut = try await applyPostRefinements(
            request: request, outputURL: outputURL, teacherURL: teacherURL, onProgress: onProgress)
        let finalBytes = UniversalSourceReader.directoryWeightBytes(outputURL)
        onProgress(.init(
            phase: "writing", completed: 4, total: 4, message: "Finished \(request.outputId)",
            valKLBefore: pvOut?.valKLBefore, valKLCurrent: pvOut?.valKLAfter))
        return ModelConversionResult(
            outputId: request.outputId,
            bytesBefore: detected.totalBytes,
            bytesAfter: finalBytes > 0 ? finalBytes : encoded.bytesAfter,
            shards: encoded.shards,
            pvTune: pvOut)
    }

    private static func applyPostRefinements(
        request: ModelConversionRequest,
        outputURL: URL,
        teacherURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> ModelConversionPVSummary? {
        if VideoDiTCheckpointCompat.isVideoDiTDirectory(teacherURL) {
            guard request.pvTune, request.target.supportsPVTune else { return nil }
            return try VideoDiTCheckpointCompat.pvTuneDirectory(output: outputURL, teacher: teacherURL, onProgress: onProgress)
        }
        if DiTBlockQuantizer.isDiTCheckpoint(outputURL) { return nil }
        if request.macaqRefine {
            onProgress(.init(phase: "macaq_refine", completed: 3, total: 4, message: "Running MACaQ-R Hessian flip-refinement"))
            _ = try await MACaQRefiner.refineConvertedDirectory(
                output: outputURL, teacher: teacherURL, calibrationText: request.calibrationText
            ) { step in
                onProgress(.init(phase: "macaq_refine", completed: 3, total: 4, message: step))
            }
        }
        guard request.pvTune, request.target.supportsPVTune else { return nil }
        onProgress(.init(phase: "pv_tune", completed: 3, total: 4, message: "Running PV-Tuning (\(request.pvPreset.rawValue))"))
        let pvCfg = PVTuneConfig(preset: request.pvPreset)
        let pv = try await PVTuner.tuneConvertedDirectory(
            output: outputURL, teacher: teacherURL, calibrationText: request.calibrationText, config: pvCfg
        ) { step in
            onProgress(.init(phase: "pv_tune", completed: 3, total: 4, message: step))
        }
        return ModelConversionPVSummary(
            valKLBefore: Double(pv.valKLBefore),
            valKLAfter: Double(pv.valKLAfter),
            updatedCodes: pv.totalUpdatedCodes)
    }
}

extension UniversalModelConverter {
    fileprivate static func validateClusterPeers(_ specs: [String]) throws(BasaltError) -> [ClusterPeerEndpoint] {
        guard !specs.isEmpty else { return [] }
        do {
            return try ClusterPeerEndpoint.parseList(specs)
        } catch {
            throw .invalidRequestStructure(details: "convert: \(error.localizedDescription)")
        }
    }

    fileprivate static func needsCanonicalStaging(detected: DetectedSourceCheckpoint, target: QuantizationTarget) -> Bool {
        if detected.requiresCanonicalDecode { return true }
        switch target {
        case .nf4:
            return detected.format.hasPrefix("affine:") && !detected.format.hasPrefix("affine:4:")
        default:
            return false
        }
    }

    private static func encodeTarget(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        switch request.target {
        case .denseBF16, .denseFP16:
            return (1, 0)
        case .affine, .mixed, .mxfp4, .nf4:
            return try await encodeSINQTarget(request: request, workingSource: workingSource, outputURL: outputURL)
        case .macaqAffine, .macaqMXFP4, .macaqMoEBudget, .flatQuant, .lirQuant, .qtip, .qtipBudget, .gsqRCO:
            return try await encodeCalibratedTarget(
                request: request, workingSource: workingSource, outputURL: outputURL, onProgress: onProgress)
        }
    }

    private static func encodeSINQTarget(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        let rawRecipe = SINQRecipe(rawValue: (request.recipe ?? "rtn").lowercased()) ?? .rtn
        let isDiT = DiTBlockQuantizer.isDiTCheckpoint(workingSource)
        let recipe: SINQRecipe = (isDiT && rawRecipe == .asinq) ? .sinq : rawRecipe
        let sinqSource: URL = if isDiT,
            !FileManager.default.fileExists(atPath: workingSource.appending(component: "config.json").path),
            FileManager.default.fileExists(atPath: workingSource.appending(path: "transformer/config.json").path) {
            workingSource.appending(component: "transformer")
        } else {
            workingSource
        }
        let adapterURL = request.adapter.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let prec: WeightConverter.Precision
        switch request.target {
        case .affine(let b, let g):
            prec = .affine(bits: b, groupSize: g)
        case .mixed(let lo, let hi, let g):
            guard let gb = request.budgetGB, gb.isFinite, gb > 0 else {
                throw BasaltError.invalidRequestStructure(details: "convert: mixed target requires a positive budget_gb")
            }
            prec = try WeightConverter.precision(quantize: "mixed:\(lo):\(hi):\(g)", dequantize: nil, format: .mlx, budgetGB: gb)
        case .mxfp4:
            prec = .mxfp4
        case .nf4(let g):
            prec = .nf4(groupSize: g)
        default:
            return (1, 0)
        }
        let conv = WeightConverter(precision: prec, adapter: adapterURL, recipe: recipe)
        let sum = try await conv.runWithSINQ(source: sinqSource, output: outputURL, calibrationText: request.calibrationText)
        return (sum.shards, Int64(sum.bytesAfter))
    }

    private static func encodeCalibratedTarget(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        if VideoDiTCheckpointCompat.isVideoDiTDirectory(workingSource) {
            return try VideoDiTCheckpointCompat.convertCalibrated(
                request: request, workingSource: workingSource, outputURL: outputURL, onProgress: onProgress
            )
        }
        switch request.target {
        case .macaqAffine(let b, let g):
            return try await encodeMACaQTarget(
                request: request, workingSource: workingSource, outputURL: outputURL,
                targetMode: .affine(bits: b, groupSize: g), onProgress: onProgress)
        case .macaqMXFP4:
            return try await encodeMACaQTarget(
                request: request, workingSource: workingSource, outputURL: outputURL,
                targetMode: .mxfp4, onProgress: onProgress)
        case .flatQuant, .lirQuant:
            let res = try await BasisTransformConverter.convertDirectory(
                source: workingSource, output: outputURL, target: request.target,
                calibrationText: request.calibrationText, alpha: request.alpha
            ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
            return (res.shards, Int64(res.bytesAfter))
        case .macaqMoEBudget(let bpw):
            return try await encodeMoERWBudgetDirectory(
                request: request, workingSource: workingSource, outputURL: outputURL,
                targetBPW: bpw, onProgress: onProgress)
        default:
            return try await encodeTrellisOrGSQTarget(
                request: request, workingSource: workingSource, outputURL: outputURL, onProgress: onProgress)
        }
    }
}

extension UniversalModelConverter {
    private static func encodeMACaQTarget(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        targetMode: MACaQTargetMode,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        var cfg = MACaQConfig()
        if let blk = request.hadamardBlock { cfg.hadamardBlock = blk }
        if let alpha = request.alpha { cfg.alpha = alpha }
        if DiTBlockQuantizer.isDiTCheckpoint(workingSource) {
            return try DiTBlockQuantizer.convertMACaQDirectory(
                source: workingSource, output: outputURL,
                targetMode: targetMode,
                hadamardBlock: request.hadamardBlock ?? 32,
                alpha: cfg.alpha
            ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
        }
        if request.recipe?.lowercased() == SINQRecipe.moeRW.rawValue {
            return try await encodeMACaQWithMoERWDirectory(
                request: request, workingSource: workingSource, outputURL: outputURL,
                targetMode: targetMode, onProgress: onProgress)
        }
        let res = try await MACaQRefiner.convertDirectory(
            source: workingSource, output: outputURL, targetMode: targetMode,
            calibrationText: request.calibrationText, config: cfg
        ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
        return (res.shards, Int64(res.bytesAfter))
    }

    private static func encodeTrellisOrGSQTarget(
        request: ModelConversionRequest,
        workingSource: URL,
        outputURL: URL,
        onProgress: @escaping @Sendable (ModelConversionProgress) -> Void
    ) async throws -> (shards: Int, bytesAfter: Int64) {
        switch request.target {
        case .qtip(let bpw):
            guard let rate = QTIPRate(bpw: bpw) else {
                throw BasaltError.invalidRequestStructure(details: "convert: invalid QTIP bpw \(bpw)")
            }
            let res = try await QTIPNativeLoader.convertDirectory(
                source: workingSource, output: outputURL, rate: rate, calibrationText: request.calibrationText
            ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
            return (res.shards, Int64(res.bytesAfter))
        case .qtipBudget(let bpw):
            let res = try await GSQRCOConverter.convertQTIPBudgetDirectory(
                source: workingSource, output: outputURL, targetBPW: bpw, calibrationText: request.calibrationText
            ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
            return (res.shards, Int64(res.bytesAfter))
        case .gsqRCO(let bpw):
            let res = try await GSQRCOConverter.convertDirectory(
                source: workingSource, output: outputURL, targetBPW: bpw, calibrationText: request.calibrationText
            ) { onProgress(.init(phase: "quantizing", completed: 2, total: 4, message: $0)) }
            return (res.shards, Int64(res.bytesAfter))
        default:
            return (1, 0)
        }
    }
}
