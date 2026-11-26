import Foundation
import MLX
import MLXNN
import MLXOptimizers
import BasaltCore

public final class TrainingOptimizer: Optimizer {
    enum Backing {
        case adamw(AdamW)
        case muon(composite: MultiOptimizer, matrix: MatchedRMSMuonOptimizer, fallback: AdamW)
        case soap(composite: MultiOptimizer, matrix: SOAPOptimizer, fallback: AdamW)
    }

    public let kind: TrainingOptimizerKind
    public let options: TrainingOptimizerOptions
    public let muonLRMultiplier: Float
    private let backing: Backing

    public var usesQKClip: Bool { kind == .muonclip }

    public var learningRate: Float {
        didSet {
            switch backing {
            case .adamw(let adamw):
                adamw.learningRate = learningRate
            case .muon(_, let matrix, let fallback):
                matrix.learningRate = learningRate * muonLRMultiplier
                fallback.learningRate = learningRate
            case .soap(_, let matrix, let fallback):
                matrix.learningRate = learningRate
                fallback.learningRate = learningRate
            }
        }
    }

    init(
        kind: TrainingOptimizerKind,
        options: TrainingOptimizerOptions,
        learningRate: Float,
        muonLRMultiplier: Float = 1.0,
        backing: Backing
    ) {
        self.kind = kind
        self.options = options
        self.learningRate = learningRate
        self.muonLRMultiplier = muonLRMultiplier
        self.backing = backing
    }

    public func update(model: Module, gradients: ModuleParameters) {
        guard !gradients.flattened().isEmpty else { return }
        switch backing {
        case .adamw(let adamw):
            adamw.update(model: model, gradients: gradients)
        case .muon(let composite, _, _), .soap(let composite, _, _):
            composite.update(model: model, gradients: gradients)
        }
    }

    public func innerState() -> [MLXArray] {
        switch backing {
        case .adamw(let adamw):
            adamw.innerState()
        case .muon(let composite, _, _), .soap(let composite, _, _):
            composite.innerState()
        }
    }

    public func state() -> ModuleParameters {
        switch backing {
        case .adamw(let adamw):
            adamw.state()
        case .muon(let composite, _, _), .soap(let composite, _, _):
            composite.state()
        }
    }

    public func update(parameters: ModuleParameters) {
        switch backing {
        case .adamw(let adamw):
            adamw.update(parameters: parameters)
        case .muon(let composite, _, _), .soap(let composite, _, _):
            composite.update(parameters: parameters)
        }
    }
}

public enum TrainingOptimizerFactory {
    public static let defaultWeightDecay: Float = 0.0
    public static let defaultMuonLRMultiplier: Float = 1.0

    public static func isMatrixShape(_ shape: [Int]) -> Bool {
        shape.count == 2 && shape[0] > 0 && shape[1] > 0
    }

    public static func isMatrixParameter(_ array: MLXArray) -> Bool {
        isMatrixShape(array.shape)
    }

    public static func isSOAPParameter(_ array: MLXArray, maxPrecondDim: Int) -> Bool {
        MatrixOptimizerMath.isSOAPEligible(shape: array.shape, maxPrecondDim: maxPrecondDim)
    }

    public static func make(
        _ kind: TrainingOptimizerKind,
        learningRate: Float,
        weightDecay: Float = defaultWeightDecay,
        muonLRMultiplier: Float = defaultMuonLRMultiplier
    ) -> sending TrainingOptimizer {
        let options = TrainingOptimizerOptions(kind: kind, weightDecay: weightDecay)
        return make(options: options, learningRate: learningRate, muonLRMultiplier: muonLRMultiplier)
    }

    public static func make(
        options: TrainingOptimizerOptions,
        learningRate: Float,
        muonLRMultiplier: Float = defaultMuonLRMultiplier
    ) -> sending TrainingOptimizer {
        switch options.kind {
        case .adamw:
            let adamw = AdamW(learningRate: learningRate, weightDecay: options.weightDecay)
            return TrainingOptimizer(kind: .adamw, options: options, learningRate: learningRate, backing: .adamw(adamw))
        case .muon, .muonclip:
            let matrix = MatchedRMSMuonOptimizer(
                learningRate: learningRate * muonLRMultiplier,
                weightDecay: options.weightDecay
            )
            let fallback = AdamW(learningRate: learningRate, weightDecay: options.weightDecay)
            let composite = MultiOptimizer(
                optimizers: [matrix, fallback],
                filters: [{ _, gradient in TrainingOptimizerFactory.isMatrixParameter(gradient) }]
            )
            return TrainingOptimizer(
                kind: options.kind,
                options: options,
                learningRate: learningRate,
                muonLRMultiplier: muonLRMultiplier,
                backing: .muon(composite: composite, matrix: matrix, fallback: fallback)
            )
        case .soap:
            let maxDim = options.soapMaxPrecondDim
            let matrix = SOAPOptimizer(learningRate: learningRate, options: options, biasCorrection: true)
            let fallback = AdamW(
                learningRate: learningRate,
                betas: (options.soapBeta1, options.soapBeta2),
                eps: options.soapEps,
                weightDecay: options.weightDecay,
                biasCorrection: true
            )
            let composite = MultiOptimizer(
                optimizers: [matrix, fallback],
                filters: [{ _, gradient in TrainingOptimizerFactory.isSOAPParameter(gradient, maxPrecondDim: maxDim) }]
            )
            return TrainingOptimizer(
                kind: .soap,
                options: options,
                learningRate: learningRate,
                muonLRMultiplier: 1.0,
                backing: .soap(composite: composite, matrix: matrix, fallback: fallback)
            )
        }
    }
}
