import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension ConvertSubcommand {
    var usesPVTune: Bool { pvTune || experimentalPvTune }

    func resolvedClusterPeers() throws -> [String] {
        guard !clusterPeers.isEmpty else { return [] }
        do {
            return try ClusterPeerEndpoint.parseList(clusterPeers).map(\.urlString)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    func validateClusterPeers() throws {
        _ = try resolvedClusterPeers()
    }

    func validateDestinationOptions() throws {
        if out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, registerCache == nil {
            throw ValidationError("specify --out <directory> or --register-cache <org/repo>")
        }
        if let repo = registerCache {
            let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
            if parts.count != 2 || parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) {
                throw ValidationError("--register-cache must be a repo ID such as local/model-4bit")
            }
        }
    }

    func validateUnifiedToOptions() throws {
        guard let to else { return }
        let legacyConflict = [
            quantize != nil,
            dequantize != nil,
            experimentalMacaq != nil,
            hasBasisTransformFlag,
            experimentalQtip != nil,
            experimentalQtipBudget != nil,
        ].contains(true)
        if legacyConflict {
            throw ValidationError("--to replaces --quantize, --dequantize, --experimental-macaq, --experimental-flatquant, --experimental-lirquant, --experimental-qtip and --experimental-qtip-budget")
        }
        let target: QuantizationTarget
        do {
            target = try QuantizationTarget.parse(to)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        if case .mixed = target {
            guard let budgetGb, budgetGb.isFinite, budgetGb > 0 else {
                throw ValidationError("--to mixed:<lo>:<hi>:<g> requires --budget-gb")
            }
        }
        if usesPVTune, !target.supportsPVTune {
            throw ValidationError("--pv-tune is not supported for target '\(target.specString)'")
        }
        if pvTeacher != nil, !usesPVTune && !experimentalMacaqRefine {
            throw ValidationError("--pv-teacher requires --pv-tune or --experimental-macaq-refine")
        }
    }

    private func buildUnifiedRequest(target: QuantizationTarget, calText: String?) -> ModelConversionRequest {
        let recipe: SINQRecipe = sinq ? .sinq : (aSinq ? .asinq : (quantRecipe ?? .rtn))
        let outputId = registerCache ?? URL(fileURLWithPath: (out as NSString).expandingTildeInPath).lastPathComponent
        var req = ModelConversionRequest(
            source: source, outputId: outputId, target: target,
            recipe: recipe.rawValue, pvTune: usesPVTune, stream: false)
        req.hadamardBlock = macaqBlock
        req.alpha = macaqAlpha
        req.budgetGB = budgetGb
        req.adapter = adapter
        req.macaqRefine = experimentalMacaqRefine
        req.pvPreset = pvPreset
        req.pvTeacher = pvTeacher
        req.calibrationText = calText
        req.clusterPeers = (try? resolvedClusterPeers()) ?? []
        return req
    }

    func runUnifiedConversion() async throws {
        guard let rawTo = to else { return }
        let target = try QuantizationTarget.parse(rawTo)
        let sourceURL = try Self.resolveModelDirectory(source)
        let req = buildUnifiedRequest(target: target, calText: try readCalibrationText())

        let start = ContinuousClock.now
        let result: ModelConversionResult
        let destLabel: String
        do {
            if let repo = registerCache, out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let svc = UniversalModelConverter()
                result = try await svc.convert(req) { prog in
                    print("  [\(prog.phase)] \(prog.message)")
                }
                destLabel = repo
            } else {
                let outURL = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
                result = try await UniversalModelConverter.convertToDirectory(
                    request: req, sourceURL: sourceURL, outputURL: outURL
                ) { prog in
                    print("  [\(prog.phase)] \(prog.message)")
                }
                destLabel = outURL.path
                if let repo = registerCache {
                    let snap = try UniversalModelConverter.cacheSnapshotDirectory(for: repo)
                    try? FileManager.default.removeItem(at: snap)
                    try FileManager.default.createDirectory(at: snap.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try FileManager.default.copyItem(at: outURL, to: snap)
                }
            }
        } catch {
            printError(error.localizedDescription)
            throw ExitCode.failure
        }
        let elapsed = ContinuousClock.now - start
        printSuccess("Converted \(sourceURL.path) → \(destLabel)")
        print("  Bytes:     \(formatBytes(result.bytesBefore)) → \(formatBytes(result.bytesAfter)) in \(result.shards) shard(s)")
        if let pv = result.pvTune {
            print(String(
                format: "  PV-Tune:   valKL %.4f → %.4f, %d code(s) updated",
                pv.valKLBefore, pv.valKLAfter, pv.updatedCodes))
        }
        print("  Time:      \(elapsed.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
    }
}
