import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension ConvertSubcommand {
    func validateMACaQOptions(isGSQRCO: Bool) throws {
        try validateMACaQKnobs()
        try validateMACaQRefineFlag(isGSQRCO: isGSQRCO)
        guard let spec = experimentalMacaq else { return }
        do {
            _ = try MACaQTargetMode.parse(spec)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let conflict = [
            hasDirectSourceConflict,
            hasBasisTransformFlag,
            experimentalQtip != nil,
            experimentalQtipBudget != nil,
            isGSQRCO,
        ].contains(true)
        if conflict {
            throw ValidationError("--experimental-macaq converts an MLX source directly without --quantize, --dequantize, --adapter or --format hf")
        }
    }

    private func validateMACaQKnobs() throws {
        if macaqBlock != nil, experimentalMacaq == nil {
            throw ValidationError("--macaq-block requires --experimental-macaq")
        }
        if macaqAlpha != nil, experimentalMacaq == nil && experimentalFlatquant == nil {
            throw ValidationError("--macaq-alpha requires --experimental-macaq or --experimental-flatquant")
        }
        if let block = macaqBlock, ![32, 64].contains(block) {
            throw ValidationError("--macaq-block takes 32 or 64, got \(block)")
        }
        if let alpha = macaqAlpha, !alpha.isFinite || !(Float(0) ... Float(1)).contains(alpha) {
            throw ValidationError("--macaq-alpha takes a finite number in [0, 1], got \(alpha)")
        }
    }

    private func validateMACaQRefineFlag(isGSQRCO: Bool) throws {
        guard experimentalMacaqRefine else { return }
        let badBase = format != .mlx || dequantize != nil || isGGUFSource
        if badBase || experimentalMacaq != nil || hasBasisTransformFlag || isGSQRCO {
            throw ValidationError("--experimental-macaq-refine applies to standard quantized MLX conversions (--format mlx without --dequantize)")
        }
    }

    func runOptionalMACaQRefine(
        output: URL,
        teacher: URL,
        calibrationText: String?
    ) async throws -> MACaQSummary? {
        guard experimentalMacaqRefine else { return nil }
        return try await MACaQRefiner.refineConvertedDirectory(
            output: output, teacher: teacher, calibrationText: calibrationText
        ) { print("  [macaq-refine] \($0)") }
    }

    func printMACaQRefineSummary(_ summary: MACaQSummary?) {
        guard let m = summary else { return }
        let reduction = Double(max(Float(0), (Float(1) - m.meanLossRatio) * Float(100)))
        print(String(
            format: "  MACaQ-R:   %d layer(s), %d code flip(s), H-loss -%.1f%%, %d tensor(s) rewritten",
            m.refinedLayers, m.totalFlips, reduction, m.rewrittenTensors))
    }

    func convertMACaQ(spec: String) async throws {
        let mode: MACaQTargetMode
        do {
            mode = try MACaQTargetMode.parse(spec)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let sourceURL = try Self.resolveModelDirectory(source)
        let teacherURL = try pvTeacher.map(Self.resolveModelDirectory) ?? sourceURL
        let calibrationText = try readCalibrationText()
        let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let existedBefore = FileManager.default.fileExists(atPath: outURL.path)
        var createdOutput = false
        var config = MACaQConfig()
        if let block = macaqBlock { config.hadamardBlock = block }
        if let alpha = macaqAlpha { config.alpha = alpha }
        let start = ContinuousClock.now
        do {
            let result = try await MACaQRefiner.convertDirectory(
                source: sourceURL, output: outURL, targetMode: mode,
                calibrationText: calibrationText, config: config
            ) { print("  [macaq] \($0)") }
            createdOutput = true
            let pvSummary = usesPVTune
                ? try await PVTuner.tuneConvertedDirectory(
                    output: outURL, teacher: teacherURL, calibrationText: calibrationText
                ) { print("  [pv-tune] \($0)") }
                : nil
            let elapsed = ContinuousClock.now - start
            let reduction = Double(max(Float(0), (Float(1) - result.summary.meanLossRatio) * Float(100)))
            printSuccess("Converted \(sourceURL.path) → \(outURL.path) (MACaQ \(mode.label), B=\(config.hadamardBlock))")
            print(String(
                format: "  MACaQ:     %d layer(s), %d code flip(s), H-loss -%.1f%% in %d shard(s)",
                result.summary.refinedLayers, result.summary.totalFlips, reduction, result.shards))
            print("  Bytes:     \(formatBytes(Int64(result.bytesAfter)))")
            if let pv = pvSummary {
                print(String(
                    format: "  PV-Tune:   valKL %.4f → %.4f, %d code(s) updated, %d tensor(s) rewritten",
                    pv.valKLBefore, pv.valKLAfter, pv.totalUpdatedCodes, pv.rewrittenTensors))
            }
            print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
        } catch {
            if createdOutput, usesPVTune,
               Self.shouldCleanFailedOutput(source: sourceURL, output: outURL, existedBefore: existedBefore) {
                try? FileManager.default.removeItem(at: outURL)
            }
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
    }
}
