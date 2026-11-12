import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

extension ServeCommand {
    static func weightBytes(_ path: String) -> Int64 {
        let root = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let files = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return files.filter { ["safetensors", "aimodel", "mlirb"].contains($0.pathExtension) }.reduce(Int64(0)) { total, file in
            let resolved = file.resolvingSymlinksInPath()
            let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey, .totalFileAllocatedSizeKey]))
            return total + Int64(size?.fileSize ?? size?.totalFileAllocatedSize ?? 0)
        }
    }

    func sidecarWorkloads(embeddings: (any EmbeddingEngine)?, decisions: (any DecisionEngine)?) async throws -> [Workload] {
        var workloads: [Workload] = []
        if let embeddingModel, let embeddings {
            let path = try await resolvedPath(embeddingModel, quant: nil)
            let ane = embeddings is CoreAIEmbeddingService
            workloads.append(Workload(name: embeddingModel, role: "embed", unit: ane ? .ane : .gpu, kind: .encoder, residentBytes: Self.weightBytes(ane ? path + "/encoder.aimodel" : path)))
        }
        if let decisionModel = decision.localModel, let decisions, decisionModel != "self" {
            let path = try await resolvedPath(decisionModel, quant: nil)
            let ane = decisions is CoreAIDecisionEngine
            let bytes = ane ? Self.weightBytes(path + "/decision.aimodel") + Self.weightBytes(path + "/token_table.aimodel")
                : decisions is JevEvaluator ? Self.weightBytes(path) + Self.weightBytes(try await resolvedPath(try JevEvaluator.baseModel(of: URL(fileURLWithPath: path)), quant: nil)) : Self.weightBytes(path)
            workloads.append(Workload(name: decisionModel, role: "decide", unit: ane ? .ane : .gpu, kind: .encoder, residentBytes: bytes))
        }
        return workloads
    }

    func admitComputePlan(chatPath: String, embeddings: (any EmbeddingEngine)?, decisions: (any DecisionEngine)?, perceive paths: PerceivePaths) async throws -> PlanVerdict {
        let weights = ByteCount(Self.weightBytes(chatPath))
        let resident = expertStreaming.setting?.residentBytes(weights: weights, modelId: targetModel) ?? weights
        let chat = Workload(name: model ?? chatPath.split(separator: "/").last.map(String.init) ?? chatPath, role: "chat", unit: .gpu, kind: .decode, residentBytes: resident.rawValue)
        let perceive = [(pipeline.ocrModel, paths.ocr), (pipeline.audioModel, paths.audio)].compactMap { model, path in
            path.map { Self.perceiveWorkload(model ?? $0, path: $0) }
        }
        let workloads = [chat] + (try await sidecarWorkloads(embeddings: embeddings, decisions: decisions)) + perceive
        let budget = await SystemProfiler().detectResources().gpuMemory
        let verdict = ComputePlan.admit(workloads, budgetBytes: budget, power: PowerSource.current(), memory: pipeline.memory)
        for line in ComputePlan.describe(verdict) { log(line) }
        if verdict.degraded { log("\(yellow("[warn]")) running on a weak power source: sidecars wait for decode and speed numbers taken now are not representative") }
        if let refusal = verdict.refusal {
            log("\(red("[error]")) \(refusal); serve fewer or smaller models")
            throw ExitCode.failure
        }
        return verdict
    }
}
