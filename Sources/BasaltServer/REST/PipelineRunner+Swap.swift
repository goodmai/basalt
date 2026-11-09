import Foundation
import BasaltCore

package struct PipelineSwap: Sendable {
    package let ocrPath: String?
    package let audioPath: String?
    package let unloadsChat: Bool
    package let serial: GenerationGate

    package init(ocrPath: String?, audioPath: String?, unloadsChat: Bool, maxQueue: Int = 8) {
        (self.ocrPath, self.audioPath, self.unloadsChat) = (ocrPath, audioPath, unloadsChat)
        serial = GenerationGate(maxConcurrent: 1, maxQueue: maxQueue)
    }

    func path(for perceiver: PipelineCore.Perceiver?) -> String? {
        switch perceiver {
        case .ocrModel?: ocrPath
        case .speechModel?: audioPath
        case .appleVision?, .chatModel?, .speechAnalyzer?, nil: nil
        }
    }
}

extension PipelineRunner {
    func perceiveStage(_ request: PipelineRequest, perceivers: PipelineCore.Perceivers) async throws(BasaltError) -> Perception {
        let order = PipelineCore.perceiveOrder(request.attachments, perceivers: perceivers, grouped: stages.swap != nil)
        guard let swap = stages.swap, order.contains(where: { swap.path(for: $0.perceiver) != nil }) else {
            return try await perceive(order, pages: request.pages, swap: nil)
        }
        var (perception, chatSwap) = try await chat.exclusively(unloadingModel: swap.unloadsChat) { () async throws(BasaltError) in
            try await self.perceive(order, pages: request.pages, swap: swap)
        }
        perception.swap += chatSwap
        return perception
    }

    func bring(_ perceiver: PipelineCore.Perceiver, swap: PipelineSwap, resident: inout ModelOrchestratorActor?) async throws(BasaltError) -> Duration {
        guard let path = swap.path(for: perceiver), let target = perceiver == .ocrModel ? stages.ocr : stages.audio, target !== resident else { return .zero }
        let started = ContinuousClock.now
        if let loaded = resident {
            await loaded.unloadModel()
            resident = nil
        }
        try await target.loadModel(path: path)
        resident = target
        return ContinuousClock.now - started
    }

    static func timed(_ body: () async -> Void) async -> Duration {
        let started = ContinuousClock.now
        await body()
        return ContinuousClock.now - started
    }
}
