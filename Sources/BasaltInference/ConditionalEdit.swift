import Foundation
import MLX
import MLXLMCommon
import BasaltCore

final class ConditionalEdit {
    let rule: ConditionalIntervention
    private(set) var gate: TriggerGate
    private var active: MLXArray?
    private let direction: MLXArray?
    private let column: MLXArray?
    private let push: MLXArray?

    init(_ rule: ConditionalIntervention, anatomy: ModelAnatomy) {
        self.rule = rule
        gate = TriggerGate(window: rule.window)
        if case .projection(_, let vector, _) = rule.trigger { direction = MLXArray(vector) } else { direction = nil }
        switch rule.action {
        case .ablateHead(let layer, let head): column = Self.oneHot(head, count: anatomy.layers[layer].heads ?? 0)
        case .ablateExpert(let layer, let expert): column = Self.oneHot(expert, count: anatomy.layers[layer].experts ?? 0)
        case .steer, .conditional: column = nil
        }
        if case .steer(_, let vector, let scale) = rule.action { push = MLXArray(vector.map { $0 * scale }) } else { push = nil }
    }

    static func oneHot(_ index: Int, count: Int) -> MLXArray {
        MLXArray((0 ..< count).map { $0 == index ? Float(1) : 0 })
    }

    private func update(_ fired: MLXArray) {
        active = MLXArray(gate.step(fired.asArray(Bool.self)).map { $0 ? Float(1) : 0 })
    }

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray {
        if case .projection(let trigger, _, let threshold) = rule.trigger, trigger == layer, let direction {
            update(matmul(hidden[0].asType(.float32), direction) .> threshold)
        }
        guard case .steer(let target, _, _) = rule.action, target == layer, let active, let push else { return hidden }
        return hidden + (active[.newAxis, 0..., .newAxis] * push).asType(hidden.dtype)
    }

    func routed(layer: Int, indices: MLXArray, gates: MLXArray) {
        guard case .expertRouted(let trigger, let expert, let threshold) = rule.trigger, trigger == layer else { return }
        let k = indices.dim(-1)
        update(((indices.reshaped(-1, k) .== expert) .&& (gates.reshaped(-1, k).asType(.float32) .> threshold)).any(axis: -1))
    }

    func headMask(layer: Int) -> MLXArray? {
        guard case .ablateHead(let target, _) = rule.action, target == layer else { return nil }
        return mask()
    }

    func expertMask(layer: Int) -> MLXArray? {
        guard case .ablateExpert(let target, _) = rule.action, target == layer else { return nil }
        return mask()
    }

    private func mask() -> MLXArray? {
        guard let active, let column else { return nil }
        return 1 - active[0..., .newAxis] * column[.newAxis]
    }
}

extension MLXInferenceEngine {
    static func validate(_ rule: ConditionalIntervention, against anatomy: ModelAnatomy) throws(BasaltError) {
        if case .conditional = rule.action { throw .invalidRequestStructure(details: "a conditional intervention cannot nest another") }
        try validate(rule.action, against: anatomy)
        let trigger = rule.trigger.layer
        guard anatomy.layers.indices.contains(trigger) else {
            throw .invalidRequestStructure(details: "trigger layer \(trigger) does not exist: the model has layers 0..<\(anatomy.layers.count)")
        }
        guard rule.action.layer > trigger else {
            throw .invalidRequestStructure(details: "the action at layer \(rule.action.layer) must come after the trigger layer \(trigger), so it can apply to the token that fires")
        }
        if let window = rule.window, !ConditionalIntervention.windowTokens.contains(window) {
            throw .invalidRequestStructure(details: "window must be in \(ConditionalIntervention.windowTokens)")
        }
        switch rule.trigger {
        case .projection(_, let vector, let threshold):
            guard vector.count == anatomy.hiddenSize, threshold.isFinite, vector.allSatisfy(\.isFinite) else {
                throw .invalidRequestStructure(details: "the trigger direction needs \(anatomy.hiddenSize) finite values and a finite threshold")
            }
        case .expertRouted(let layer, let expert, let threshold):
            let experts = anatomy.layers[layer].experts ?? 0
            guard (0 ..< experts).contains(expert), threshold.isFinite else {
                throw .invalidRequestStructure(details: "layer \(layer) has \(experts) experts; trigger expert \(expert) does not exist or its gate is not finite")
            }
        }
    }

    static func describe(_ rule: ConditionalIntervention) -> String {
        let trigger = switch rule.trigger {
        case .projection(let layer, let vector, let threshold): "the residual after layer \(layer) projects above \(threshold) on a \(vector.count)-value direction"
        case .expertRouted(let layer, let expert, let threshold): "expert \(expert) of layer \(layer) is routed with gate above \(threshold)"
        }
        let window = rule.window.map { " and for \($0) tokens from the first trigger" } ?? ""
        return "\(describe(rule.action)) when \(trigger)\(window)"
    }
}
