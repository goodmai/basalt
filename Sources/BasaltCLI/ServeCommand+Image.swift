import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    func loadImageSynthesizer(gate: GenerationGate) async throws -> (any ImageSynthesizer)? {
        guard let imageModel else { return nil }
        let path = try await resolvedPath(imageModel, quant: nil)
        do {
            let loaded = try await ImageSynthesizerLoader.load(
                identifier: imageModel,
                directory: URL(fileURLWithPath: path)
            )
            log("Serving \(loaded.modelIdentifier) (/v1/images/*, \(loaded.descriptor.defaultCompute.rawValue), \(loaded.descriptor.licenseTier.rawValue))")
            let gpuBusyProbe: @Sendable () async -> Bool = {
                if await gate.inFlight > 0 { return true }
                return await gate.queued > 0
            }
            return RoutedImageSynthesizer(
                entries: [.init(alias: imageModel, synthesizer: loaded)],
                onGPUBusy: .queue,
                isGPUBusy: gpuBusyProbe
            )
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}
