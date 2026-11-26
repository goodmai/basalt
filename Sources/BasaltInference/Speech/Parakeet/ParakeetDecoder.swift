import Foundation
import MLX
import MLXNN

struct ParakeetLSTMState {
    let hidden: [MLXArray]
    let cell: [MLXArray]
}

final class ParakeetLSTMStack: Module {
    @ModuleInfo(key: "lstm") var layers: [LSTM]

    init(size: Int, layers: Int) {
        _layers.wrappedValue = (0 ..< layers).map { _ in LSTM(inputSize: size, hiddenSize: size) }
    }

    func callAsFunction(_ x: MLXArray, state: ParakeetLSTMState?) -> (MLXArray, ParakeetLSTMState) {
        var output = x
        var hidden: [MLXArray] = []
        var cell: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            let (steps, cells) = layer(output, hidden: state?.hidden[index], cell: state?.cell[index])
            output = steps
            hidden.append(steps[0..., -1, 0...])
            cell.append(cells[0..., -1, 0...])
        }
        return (output, ParakeetLSTMState(hidden: hidden, cell: cell))
    }
}

final class ParakeetPrediction: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var rnn: ParakeetLSTMStack

    init(_ predictor: ParakeetConfiguration.Predictor) {
        let hidden = predictor.prediction.hidden
        _embed.wrappedValue = Embedding(embeddingCount: predictor.vocabularySize + 1, dimensions: hidden)
        _rnn.wrappedValue = ParakeetLSTMStack(size: hidden, layers: predictor.prediction.layers)
    }

    func callAsFunction(_ tokens: MLXArray?, state: ParakeetLSTMState?) -> (MLXArray, ParakeetLSTMState) {
        let input = tokens.map { embed($0) } ?? MLXArray.zeros([1, 1, embed.weight.dim(1)], dtype: embed.weight.dtype)
        return rnn(input, state: state)
    }
}

final class ParakeetDecoder: Module {
    @ModuleInfo(key: "prediction") var prediction: ParakeetPrediction

    init(_ predictor: ParakeetConfiguration.Predictor) {
        _prediction.wrappedValue = ParakeetPrediction(predictor)
    }
}

final class ParakeetJoint: Module {
    @ModuleInfo(key: "enc") var encoderProjection: Linear
    @ModuleInfo(key: "pred") var predictionProjection: Linear
    @ModuleInfo(key: "joint_net") var network: [Module]

    init(_ configuration: ParakeetConfiguration) {
        let joint = configuration.joint.network
        let activation: Module = switch joint.activation {
        case "sigmoid": Sigmoid()
        case "tanh": Tanh()
        default: ReLU()
        }
        let classes = configuration.joint.classes + 1 + configuration.joint.extraOutputs
        _encoderProjection.wrappedValue = Linear(joint.encoderHidden, joint.hidden)
        _predictionProjection.wrappedValue = Linear(joint.predictionHidden, joint.hidden)
        _network.wrappedValue = [activation, Identity(), Linear(joint.hidden, classes)]
    }

    func output(_ x: MLXArray) -> MLXArray {
        var y = x
        for case let layer as any UnaryLayer in network {
            y = layer(y)
        }
        return y
    }
}

struct ParakeetPredictionStep {
    let projected: MLXArray
    let state: ParakeetLSTMState
}

extension ParakeetModel {
    static func advance(duration: Int, blank: Bool, symbols: Int, maxSymbols: Int) -> (frames: Int, symbols: Int) {
        if duration > 0 { return (duration, 0) }
        if blank || symbols >= maxSymbols { return (1, 0) }
        return (0, symbols)
    }

    func predictionStep(token: Int?, state: ParakeetLSTMState?) -> ParakeetPredictionStep {
        let tokens = token.map { MLXArray([Int32($0)]).reshaped(1, 1) }
        let (output, next) = decoder.prediction(tokens, state: state)
        return ParakeetPredictionStep(projected: joint.predictionProjection(output[0, 0]), state: next)
    }

    func greedyTokens(_ features: MLXArray) throws -> [Int] {
        let encoded = joint.encoderProjection(features)
        let (blank, durations, maxSymbols) = (configuration.blank, configuration.decoding.durations, configuration.maxSymbols)
        var step = predictionStep(token: nil, state: nil)
        var tokens: [Int] = []
        var (time, symbols) = (0, 0)
        while time < features.dim(0) {
            try Task.checkCancellation()
            let logits = joint.output(encoded[time] + step.projected)
            let token = argMax(logits[..<(blank + 1)])
            let choice = argMax(logits[(blank + 1)...])
            eval(token, choice)
            let (id, duration) = (token.item(Int.self), durations[choice.item(Int.self)])
            if id != blank {
                tokens.append(id)
                step = predictionStep(token: id, state: step.state)
            }
            let advanced = Self.advance(duration: duration, blank: id == blank, symbols: symbols + 1, maxSymbols: maxSymbols)
            (time, symbols) = (time + advanced.frames, advanced.symbols)
        }
        return tokens
    }
}
