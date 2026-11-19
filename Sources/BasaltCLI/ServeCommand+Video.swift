import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation

extension ServeCommand {
    func loadVideoSynthesizer() async throws -> (any VideoSynthesizer)? {
        guard let videoModel else { return nil }
        let path = try await resolvedPath(videoModel, quant: nil)
        do {
            let runner = try PhasedVideoRunnerActor(
                identifier: videoModel,
                options: .init(directory: URL(fileURLWithPath: path))
            )
            log("Serving \(videoModel) (/v1/videos/*, phased UMA)")
            return runner
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}
