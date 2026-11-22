import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension ConvertSubcommand {
    var hasBasisTransformFlag: Bool {
        experimentalFlatquant != nil || experimentalLirquant != nil
    }

    func basisTransformCLITarget() throws -> QuantizationTarget? {
        if let raw = experimentalFlatquant {
            return try Self.parseFlatQuantFlag(raw)
        }
        if let raw = experimentalLirquant {
            return try Self.parseLIRQuantFlag(raw)
        }
        return nil
    }

    func validateBasisTransformOptions(isGSQRCO: Bool) throws {
        if experimentalFlatquant != nil && experimentalLirquant != nil {
            throw ValidationError("pass at most one of --experimental-flatquant or --experimental-lirquant")
        }
        guard let target = try basisTransformCLITarget() else { return }
        _ = target
        let conflict = [
            hasDirectSourceConflict,
            experimentalMacaq != nil,
            experimentalQtip != nil,
            experimentalQtipBudget != nil,
            isGSQRCO,
        ].contains(true)
        if conflict {
            throw ValidationError("--experimental-flatquant and --experimental-lirquant convert an MLX source directly without --quantize, --dequantize, --adapter or --format hf")
        }
    }

    static func parseFlatQuantFlag(_ raw: String) throws -> QuantizationTarget {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normalized: String
        if trimmed.hasPrefix("flatquant:") {
            normalized = trimmed
        } else if trimmed.hasPrefix("affine:") {
            normalized = "flatquant:" + trimmed.dropFirst("affine:".count)
        } else {
            normalized = "flatquant:" + trimmed
        }
        do {
            return try QuantizationTarget.parse(normalized)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    static func parseLIRQuantFlag(_ raw: String) throws -> QuantizationTarget {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normalized = trimmed.hasPrefix("lirquant:") ? trimmed : ("lirquant:" + trimmed)
        do {
            return try QuantizationTarget.parse(normalized)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    func convertBasisTransform(target: QuantizationTarget) async throws {
        let sourceURL = try Self.resolveModelDirectory(source)
        let calibrationText = try readCalibrationText()
        let outputId = registerCache ?? URL(fileURLWithPath: (out as NSString).expandingTildeInPath).lastPathComponent
        var req = ModelConversionRequest(
            source: source, outputId: outputId, target: target,
            pvTune: usesPVTune, stream: false)
        req.alpha = macaqAlpha
        req.pvPreset = pvPreset
        req.pvTeacher = pvTeacher
        req.calibrationText = calibrationText
        let start = ContinuousClock.now
        do {
            let result: ModelConversionResult
            let destLabel: String
            if let repo = registerCache, out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                result = try await UniversalModelConverter().convert(req) { print("  [\($0.phase)] \($0.message)") }
                destLabel = repo
            } else {
                let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
                result = try await UniversalModelConverter.convertToDirectory(
                    request: req, sourceURL: sourceURL, outputURL: outURL
                ) { print("  [\($0.phase)] \($0.message)") }
                destLabel = outURL.path
                if let repo = registerCache {
                    let snap = try UniversalModelConverter.cacheSnapshotDirectory(for: repo)
                    try Self.copyToCacheSnapshot(from: outURL, to: snap)
                }
            }
            let elapsed = ContinuousClock.now - start
            printSuccess("Converted \(sourceURL.path) → \(destLabel) (\(target.specString))")
            print("  Bytes:     \(formatBytes(result.bytesBefore)) → \(formatBytes(result.bytesAfter)) in \(result.shards) shard(s)")
            if let pv = result.pvTune {
                print(String(
                    format: "  PV-Tune:   valKL %.4f → %.4f, %d code(s) updated",
                    pv.valKLBefore, pv.valKLAfter, pv.updatedCodes))
            }
            print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }
}
