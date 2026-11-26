import ArgumentParser
import BasaltCore

struct OptimizerTrainingOptions: ParsableArguments {
    @Option(name: .long, help: "Optimizer for training: adamw, muon, muonclip or soap")
    var optimizer: TrainingOptimizerKind = .adamw

    @Option(name: .customLong("qk-clip-tau"), help: "MuonClip max pre-softmax attention logit threshold tau (> 0, default 100.0)")
    var qkClipTau: Float?

    @Option(name: .customLong("qk-clip-alpha"), help: "MuonClip Q/K split exponent alpha in [0, 1] (default 0.5)")
    var qkClipAlpha: Float?

    @Option(name: .customLong("soap-precond-freq"), help: "SOAP eigenbasis refresh interval in steps (1...10000, default 10)")
    var soapPrecondFreq: Int?

    @Option(name: .customLong("soap-max-precond-dim"), help: "SOAP max matrix axis dimension to precondition before AdamW fallback (1...4096, default 1024)")
    var soapMaxPrecondDim: Int?

    @Option(name: .customLong("soap-shampoo-beta"), help: "SOAP preconditioner EMA decay in [0, 1) (default 0.95)")
    var soapShampooBeta: Float?

    func buildOptions(weightDecay: Float = TrainingOptimizerOptions.defaultWeightDecay) -> TrainingOptimizerOptions {
        var options = TrainingOptimizerOptions(kind: optimizer, weightDecay: weightDecay)
        if let qkClipTau { options.qkClipTau = qkClipTau }
        if let qkClipAlpha { options.qkClipAlpha = qkClipAlpha }
        if let soapPrecondFreq { options.soapPrecondFreq = soapPrecondFreq }
        if let soapMaxPrecondDim { options.soapMaxPrecondDim = soapMaxPrecondDim }
        if let soapShampooBeta { options.soapShampooBeta = soapShampooBeta }
        return options
    }

    func resolve(weightDecay: Float = TrainingOptimizerOptions.defaultWeightDecay) throws -> TrainingOptimizerOptions {
        if optimizer != .muonclip && (qkClipTau != nil || qkClipAlpha != nil) {
            throw ValidationError("--qk-clip-tau and --qk-clip-alpha require --optimizer muonclip")
        }
        if optimizer != .soap && (soapPrecondFreq != nil || soapMaxPrecondDim != nil || soapShampooBeta != nil) {
            throw ValidationError("--soap-* options require --optimizer soap")
        }
        let options = buildOptions(weightDecay: weightDecay)
        do {
            try options.validateAdapterCompatible(prefix: "optimizer")
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        return options
    }

    func validate() throws {
        _ = try resolve()
    }
}
