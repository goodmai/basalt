import BasaltCore
import Foundation
import MLX
import MLXNN

public struct Qwen35GatedDeltaGate {
    public let a: MLXArray
    public let b: MLXArray
    public let aLog: MLXArray
    public let dtBias: MLXArray

    public init(a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray) {
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
    }

    public func resolvedGamma() -> MLXArray {
        exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    }

    public func resolvedBeta() -> MLXArray {
        sigmoid(b).asType(.float32)
    }
}

public struct GatedDeltaInputs {
    public var queries: MLXArray
    public var keys: MLXArray
    public var values: MLXArray
    public var gamma: MLXArray
    public var beta: MLXArray
    public var initialState: MLXArray?
    public var mask: MLXArray?

    public init(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        gamma: MLXArray,
        beta: MLXArray
    ) {
        self.queries = queries
        self.keys = keys
        self.values = values
        self.gamma = gamma
        self.beta = beta
    }

    public static func fromQwen35(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        gate: Qwen35GatedDeltaGate
    ) -> GatedDeltaInputs {
        GatedDeltaInputs(
            queries: queries,
            keys: keys,
            values: values,
            gamma: gate.resolvedGamma(),
            beta: gate.resolvedBeta()
        )
    }

    public func withInitialState(_ state: MLXArray?) -> GatedDeltaInputs {
        var copy = self
        copy.initialState = state
        return copy
    }

    public func withMask(_ mask: MLXArray?) -> GatedDeltaInputs {
        var copy = self
        copy.mask = mask
        return copy
    }
}

public struct GatedDeltaMaterializedResult: Sendable {
    public let output: MaterializedTensorSnapshot
    public let state: MaterializedTensorSnapshot

    public init(output: MaterializedTensorSnapshot, state: MaterializedTensorSnapshot) {
        self.output = output
        self.state = state
    }
}

public struct GatedDeltaCPUInputs: Sendable {
    public let batch: Int
    public let steps: Int
    public let hk: Int
    public let hv: Int
    public let dk: Int
    public let dv: Int
    public var queries: [Float]
    public var keys: [Float]
    public var values: [Float]
    public var gamma: [Float]
    public var beta: [Float]
    public var initialState: [Float]?
    public var mask: [Bool]?

    public init(
        shape: [Int],
        queries: [Float],
        keys: [Float],
        values: [Float],
        gamma: [Float],
        beta: [Float]
    ) {
        precondition(shape.count == 6, "shape must be [B, T, Hk, Hv, Dk, Dv]")
        (self.batch, self.steps, self.hk, self.hv, self.dk, self.dv) = (
            shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]
        )
        self.queries = queries
        self.keys = keys
        self.values = values
        self.gamma = gamma
        self.beta = beta
    }

    public func withStateAndMask(initialState: [Float]?, mask: [Bool]?) -> GatedDeltaCPUInputs {
        var copy = self
        copy.initialState = initialState
        copy.mask = mask
        return copy
    }
}

public enum GatedDeltaFastPath {
    struct ValidatedDimensions: Equatable, Sendable {
        let batch: Int
        let steps: Int
        let hk: Int
        let hv: Int
        let dk: Int
        let dv: Int
    }

    public static func update(
        _ inputs: GatedDeltaInputs,
        stream: StreamOrDevice = .default
    ) throws(BasaltError) -> (output: MLXArray, state: MLXArray) {
        _ = try validate(inputs)
        return MLXFast.gatedDeltaUpdate(
            queries: inputs.queries,
            keys: inputs.keys,
            values: inputs.values,
            gamma: inputs.gamma,
            beta: inputs.beta,
            initialState: inputs.initialState,
            mask: inputs.mask,
            stream: stream
        )
    }

    public static func materializedUpdate(
        _ inputs: GatedDeltaInputs,
        stream: StreamOrDevice = .default
    ) throws(BasaltError) -> GatedDeltaMaterializedResult {
        let pair = try update(inputs, stream: stream)
        let frozen = MaterializedTensorBridge.freezeBatch([pair.output, pair.state])
        return GatedDeltaMaterializedResult(output: frozen[0], state: frozen[1])
    }

    public static func gatedDeltaStepReference(
        _ inputs: GatedDeltaInputs
    ) throws(BasaltError) -> (output: MLXArray, state: MLXArray) {
        let dims = try validate(inputs)
        let outDType = resolvedOutputDType(inputs.queries.dtype, inputs.keys.dtype)
        var q = inputs.queries.asType(outDType)
        var k = inputs.keys.asType(outDType)
        let v = inputs.values.asType(outDType)
        let gamma = inputs.gamma.asType(outDType)
        let beta = inputs.beta.asType(outDType)
        let repeatFactor = dims.hv / dims.hk
        if repeatFactor > 1 {
            q = repeated(q, count: repeatFactor, axis: 2)
            k = repeated(k, count: repeatFactor, axis: 2)
        }
        var state = inputs.initialState?.asType(.float32)
            ?? MLXArray.zeros([dims.batch, dims.hv, dims.dv, dims.dk], dtype: .float32)
        guard dims.steps > 0 else {
            return (output: MLXArray.zeros([dims.batch, 0, dims.hv, dims.dv], dtype: outDType), state: state)
        }
        let boolMask = inputs.mask?.asType(.bool)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(dims.steps)
        for step in 0 ..< dims.steps {
            let slice = StepSlice(
                q: q[0..., step],
                k: k[0..., step],
                v: v[0..., step],
                gamma: gamma[0..., step],
                beta: beta[0..., step],
                mask: boolMask.map { $0[0..., step] }
            )
            let (yStep, nextState) = referenceStep(slice, state: state, outputDType: outDType)
            outputs.append(yStep)
            state = nextState
        }
        return (output: MLX.stacked(outputs, axis: 1), state: state)
    }

    private struct StepSlice {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let gamma: MLXArray
        let beta: MLXArray
        let mask: MLXArray?
    }

    private static func referenceStep(
        _ slice: StepSlice,
        state: MLXArray,
        outputDType: DType
    ) -> (MLXArray, MLXArray) {
        let decay = slice.gamma.ndim == 2
            ? expandedDimensions(slice.gamma, axes: [2, 3])
            : expandedDimensions(slice.gamma, axis: -2)
        let decayed = state * decay
        let kExpanded = expandedDimensions(slice.k, axis: -2)
        let kvMem = (decayed * kExpanded).sum(axis: -1)
        let delta = (slice.v - kvMem) * expandedDimensions(slice.beta, axis: -1)
        let candidate = decayed + expandedDimensions(delta, axis: -1) * kExpanded
        let nextState: MLXArray
        if let mask = slice.mask {
            nextState = MLX.where(expandedDimensions(mask, axes: [1, 2, 3]), candidate, state)
        } else {
            nextState = candidate
        }
        let rawOut = (nextState * expandedDimensions(slice.q, axis: -2)).sum(axis: -1)
        let stepOut: MLXArray
        if let mask = slice.mask {
            stepOut = MLX.where(expandedDimensions(mask, axes: [1, 2]), rawOut, MLXArray(Float(0)))
        } else {
            stepOut = rawOut
        }
        return (stepOut.asType(outputDType), nextState)
    }
}

extension GatedDeltaFastPath {
    static func validate(_ inputs: GatedDeltaInputs) throws(BasaltError) -> ValidatedDimensions {
        guard inputs.queries.ndim == 4, inputs.keys.ndim == 4, inputs.values.ndim == 4 else {
            throw .invalidRequestStructure(details: "gatedDeltaUpdate requires rank-4 queries, keys, and values")
        }
        let (batch, steps, hk, dk) = (
            inputs.queries.dim(0), inputs.queries.dim(1), inputs.queries.dim(2), inputs.queries.dim(3)
        )
        guard inputs.keys.shape == [batch, steps, hk, dk] else {
            throw .invalidRequestStructure(details: "keys shape \(inputs.keys.shape) must match queries \(inputs.queries.shape)")
        }
        let (hv, dv) = (inputs.values.dim(2), inputs.values.dim(3))
        guard inputs.values.dim(0) == batch, inputs.values.dim(1) == steps,
              hk > 0, hv > 0, dk > 0, dv > 0, hv % hk == 0 else {
            throw .invalidRequestStructure(details: "values shape \(inputs.values.shape) incompatible with Hk=\(hk)")
        }
        let validGamma = inputs.gamma.shape == [batch, steps, hv] || inputs.gamma.shape == [batch, steps, hv, dk]
        guard validGamma, inputs.beta.shape == [batch, steps, hv] else {
            throw .invalidRequestStructure(details: "gamma \(inputs.gamma.shape) or beta \(inputs.beta.shape) mismatch [B,T,Hv]")
        }
        if let state = inputs.initialState, state.shape != [batch, hv, dv, dk] {
            throw .invalidRequestStructure(details: "initialState shape \(state.shape) must be \([batch, hv, dv, dk])")
        }
        if let mask = inputs.mask, mask.shape != [batch, steps] {
            throw .invalidRequestStructure(details: "mask shape \(mask.shape) must be \([batch, steps])")
        }
        return ValidatedDimensions(batch: batch, steps: steps, hk: hk, hv: hv, dk: dk, dv: dv)
    }

    static func resolvedOutputDType(_ lhs: DType, _ rhs: DType) -> DType {
        if lhs == rhs, lhs.isFloatingPoint { return lhs }
        if lhs == .float64 || rhs == .float64 { return .float64 }
        return .float32
    }

    public static func gatedDeltaStepCPUReference(
        _ inputs: GatedDeltaCPUInputs
    ) -> (output: [Float], state: [Float]) {
        let stateCount = inputs.batch * inputs.hv * inputs.dv * inputs.dk
        var state = inputs.initialState ?? [Float](repeating: 0, count: stateCount)
        var output = [Float](repeating: 0, count: inputs.batch * inputs.steps * inputs.hv * inputs.dv)
        let groupSize = inputs.hv / inputs.hk
        for b in 0 ..< inputs.batch {
            for t in 0 ..< inputs.steps {
                let active = inputs.mask.map { $0[b * inputs.steps + t] } ?? true
                for hvIndex in 0 ..< inputs.hv {
                    cpuHeadStep(
                        inputs: inputs,
                        coords: [b, t, hvIndex, hvIndex / groupSize],
                        active: active,
                        state: &state,
                        output: &output
                    )
                }
            }
        }
        return (output: output, state: state)
    }

    private static func cpuHeadStep(
        inputs: GatedDeltaCPUInputs,
        coords: [Int],
        active: Bool,
        state: inout [Float],
        output: inout [Float]
    ) {
        let (b, t, hvIndex, hkIndex) = (coords[0], coords[1], coords[2], coords[3])
        let outBase = ((b * inputs.steps + t) * inputs.hv + hvIndex) * inputs.dv
        guard active else { return }
        let qkBase = ((b * inputs.steps + t) * inputs.hk + hkIndex) * inputs.dk
        let vBase = outBase
        let gateIndex = (b * inputs.steps + t) * inputs.hv + hvIndex
        let decay = inputs.gamma[gateIndex]
        let rate = inputs.beta[gateIndex]
        let stateHeadBase = (b * inputs.hv + hvIndex) * inputs.dv * inputs.dk
        for dvIndex in 0 ..< inputs.dv {
            let rowBase = stateHeadBase + dvIndex * inputs.dk
            var kvMem: Float = 0
            for dkIndex in 0 ..< inputs.dk {
                let decayed = state[rowBase + dkIndex] * decay
                state[rowBase + dkIndex] = decayed
                kvMem += decayed * inputs.keys[qkBase + dkIndex]
            }
            let delta = (inputs.values[vBase + dvIndex] - kvMem) * rate
            var yVal: Float = 0
            for dkIndex in 0 ..< inputs.dk {
                let updated = state[rowBase + dkIndex] + delta * inputs.keys[qkBase + dkIndex]
                state[rowBase + dkIndex] = updated
                yVal += updated * inputs.queries[qkBase + dkIndex]
            }
            output[outBase + dvIndex] = yVal
        }
    }
}
