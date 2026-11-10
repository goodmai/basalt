import Foundation
import ArgumentParser
import BasaltCore
import BasaltCloud

struct DecisionOptions: ParsableArguments {
    static let cloudPrefix = "cloud:"

    @Option(name: .customLong("decision-model"), help: "OpenJev decision model served on /v1/systemone (single-pass noul/choice/score): \"self\" reads option letters from the chat model; a Core AI decision bundle (laya_decision.json with decision.aimodel, token_table.aimodel and tokenizer/) runs on the Neural Engine beside decode (on a weak power source it waits between generations); an OpenJev head bundle (openjev_config.json, head.safetensors, PEFT adapter) runs on its Qwen3 base; any other repo, such as openjev/OpenJev-Flash-9B-MLX, is loaded beside the chat model and read out by option letters; cloud:<model>, such as cloud:~typesafe/jev-latest, asks OpenRouter's decisions endpoint first (needs OPENROUTER_API_KEY and --decision-escalate; the state is redacted of secrets, and pipeline states read from documents or audio never leave the machine)")
    var model: String?

    @Option(name: .customLong("decision-escalate"), help: "With --decision-model cloud:<model>, the local decision model (self, repo or directory, as for --decision-model) that answers the questions the cloud model is unsure of, and the whole request when the cloud call fails or times out (3 s)")
    var escalate: String?

    @Option(name: .customLong("decision-escalate-below"), help: "With --decision-model cloud:<model>, questions whose cloud confidence is below this value go to --decision-escalate (0 to 1; default: 0.6)")
    var escalateBelow: Double = 0.6

    var cloudModel: String? {
        model.flatMap { $0.hasPrefix(Self.cloudPrefix) ? String($0.dropFirst(Self.cloudPrefix.count)) : nil }
    }

    var localModel: String? { cloudModel == nil ? model : escalate }

    func validate() throws {
        guard (0 ... 1).contains(escalateBelow) else { throw ValidationError("--decision-escalate-below must lie between 0 and 1, got \(escalateBelow)") }
        guard let cloudModel else {
            if escalate != nil { throw ValidationError("--decision-escalate needs --decision-model cloud:<model>") }
            return
        }
        guard !cloudModel.trimmingCharacters(in: .whitespaces).isEmpty else { throw ValidationError("--decision-model cloud: needs a model, such as cloud:~typesafe/jev-latest") }
        guard let escalate, !escalate.hasPrefix(Self.cloudPrefix) else {
            throw ValidationError("--decision-model cloud:\(cloudModel) needs --decision-escalate <self|repo|dir>: a local decision model for unsure questions and failed cloud calls")
        }
    }
}

extension ServeCommand {
    func servedDecisions(
        _ local: (any DecisionEngine)?, gate: GenerationGate, besideDecode: Bool, apiKey: String = ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? ""
    ) throws -> (any DecisionEngine)? {
        guard let cloud = decision.cloudModel, let local else { return GatedDecisionEngine.serving(local, gate: gate, besideDecode: besideDecode) }
        do {
            let remote = try RemoteDecisionEngine(model: cloud, apiKey: apiKey)
            log("Cloud decisions → \(bold(cloud)) at \(RemoteDecisionEngine.openRouter.host ?? ""); confidence below \(decision.escalateBelow) and failed calls go to \(local.modelID)")
            return CascadeDecisionEngine(remote: remote, local: local, escalateBelow: decision.escalateBelow, gate: gate, besideDecode: besideDecode)
        } catch {
            log("\(red("[error]")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}
