import Foundation
import MLX
import MLXNN
import MLXOptimizers
import BasaltCore

public final class TrainingOptimizer: Optimizer {
    enum Backing {
        case adamw(AdamW)
        case muon(composite: MultiOptimizer, matrix: Muon, fallback: AdamW)
    }

    public let kind: TrainingOptimizerKind
    private let backing: Backing

    public var learningRate: Float {
        didSet {
            switch backing {
            case .adamw(let adamw):
                adamw.learningRate = learningRate
            case .muon(_, let matrix, let fallback):
                matrix.learningRate = learningRate
                fallback.learningRate = learningRate
            }
        }
    }

    init(kind: TrainingOptimizerKind, learningRate: Float, backing: Backing) {
        self.kind = kind
        self.learningRate = learningRate
        self.backing = backing
    }

    public func update(model: Module, gradients: ModuleParameters) {
        guard !gradients.flattened().isEmpty else { return }
        switch backing {
        case .adamw(let adamw):
            adamw.update(model: model, gradients: gradients)
        case .muon(let composite, _, _):
            composite.update(model: model, gradients: gradients)
        }
    }

    public func innerState() -> [MLXArray] {
        switch backing {
        case .adamw(let adamw):
            adamw.innerState()
        case .muon(let composite, _, _):
            composite.innerState()
        }
    }

    public func state() -> ModuleParameters {
        switch backing {
        case .adamw(let adamw):
            adamw.state()
        case .muon(let composite, _, _):
            composite.state()
        }
    }

    public func update(parameters: ModuleParameters) {
        switch backing {
        case .adamw(let adamw):
            adamw.update(parameters: parameters)
        case .muon(let composite, _, _):
            composite.update(parameters: parameters)
        }
    }
}

public enum TrainingOptimizerFactory {
    public static let defaultWeightDecay: Float = 0.01

    public static func isMatrixShape(_ shape: [Int]) -> Bool {
        shape.count == 2 && shape[0] > 0 && shape[1] > 0
    }

    public static func isMatrixParameter(_ array: MLXArray) -> Bool {
        isMatrixShape(array.shape)
    }

    public static func make(
        _ kind: TrainingOptimizerKind,
        learningRate: Float,
        weightDecay: Float = defaultWeightDecay
    ) -> sending TrainingOptimizer {
        switch kind {
        case .adamw:
            let adamw = AdamW(learningRate: learningRate, weightDecay: weightDecay)
            return TrainingOptimizer(kind: .adamw, learningRate: learningRate, backing: .adamw(adamw))
        case .muon:
            let matrix = Muon(learningRate: learningRate, weightDecay: weightDecay)
            let fallback = AdamW(learningRate: learningRate, weightDecay: weightDecay)
            let composite = MultiOptimizer(
                optimizers: [matrix, fallback],
                filters: [{ _, gradient in TrainingOptimizerFactory.isMatrixParameter(gradient) }]
            )
            return TrainingOptimizer(
                kind: .muon,
                learningRate: learningRate,
                backing: .muon(composite: composite, matrix: matrix, fallback: fallback)
            )
        }
    }
}
