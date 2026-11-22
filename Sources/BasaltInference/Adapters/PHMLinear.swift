import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

enum PHMContraction {
    static let trainableKeys: Set<String> = ["phm_a", "phm_s", "phm_t"]

    static func validate(
        inputDims: Int,
        outputDims: Int,
        hypercomplexDim: Int,
        factorRank: Int,
        scale: Float
    ) throws(BasaltError) {
        let method = AdapterMethod.phm(
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            sharedBasis: true,
            scale: scale
        )
        try method.validate(inputDim: inputDims, outputDim: outputDims, prefix: "phm")
    }

    static func validateBank(_ mode: PHMBasisMode, hypercomplexDim: Int) throws(BasaltError) {
        switch mode {
        case .independent:
            return
        case .owned(let bank), .borrowed(let bank):
            guard bank.hypercomplexDim == hypercomplexDim,
                  bank.basis.shape == [hypercomplexDim, hypercomplexDim, hypercomplexDim] else {
                throw .invalidRequestStructure(
                    details: "phm: shared basis bank dimension (\(bank.hypercomplexDim)) does not match hypercomplexDim (\(hypercomplexDim))"
                )
            }
        }
    }

    static func resolveBasis(_ mode: PHMBasisMode, hypercomplexDim: Int) -> (bank: PHMBasisBank, owned: MLXArray?) {
        switch mode {
        case .independent:
            let bank = PHMBasisBank(hypercomplexDim: hypercomplexDim)
            return (bank, bank.basis)
        case .owned(let bank):
            return (bank, bank.basis)
        case .borrowed(let bank):
            return (bank, nil)
        }
    }

    static func initialFactors(
        inputDims: Int,
        outputDims: Int,
        hypercomplexDim: Int,
        factorRank: Int
    ) -> (s: MLXArray, t: MLXArray) {
        let p = outputDims / hypercomplexDim
        let q = inputDims / hypercomplexDim
        let tScale = 1.0 / sqrt(Float(max(1, q)))
        let s = MLXArray.zeros([hypercomplexDim, p, factorRank])
        let t = MLXRandom.uniform(low: -tScale, high: tScale, [hypercomplexDim, q, factorRank])
        return (s, t)
    }

    static func forward(
        _ x: MLXArray,
        basis: MLXArray,
        factorS: MLXArray,
        factorT: MLXArray
    ) -> MLXArray {
        let n = basis.dim(0)
        let p = factorS.dim(1)
        let q = factorT.dim(1)
        let r = factorS.dim(2)
        let inputDims = n * q
        let outputDims = n * p
        let prefix = Array(x.shape.dropLast())
        let mCount = x.size / inputDims
        guard mCount > 0 else { return MLXArray.zeros(prefix + [outputDims], dtype: x.dtype) }
        let computeType = factorT.dtype
        let xBlocks = x.asType(computeType).reshaped(mCount * n, q)
        let tFlat = factorT.swappedAxes(0, 1).reshaped(q, n * r)
        let h1 = matmul(xBlocks, tFlat)
        let h1Batched = h1.reshaped(mCount, n, n, r).transposed(2, 0, 3, 1).reshaped(n, mCount * r, n)
        let h2 = matmul(h1Batched, basis.asType(computeType).swappedAxes(-1, -2))
        let h2Flat = h2.reshaped(n, mCount, r, n).transposed(1, 3, 0, 2).reshaped(mCount * n, n * r)
        let sFlat = factorS.asType(computeType).swappedAxes(1, 2).reshaped(n * r, p)
        return matmul(h2Flat, sFlat).reshaped(prefix + [outputDims])
    }

    static func materializeDelta(basis: MLXArray, factorS: MLXArray, factorT: MLXArray) -> MLXArray {
        let n = basis.dim(0)
        let p = factorS.dim(1)
        let q = factorT.dim(1)
        let b = matmul(factorS, factorT.swappedAxes(-1, -2))
        let ab = matmul(basis.reshaped(n, n * n).T, b.reshaped(n, p * q))
        return ab.reshaped(n, n, p, q).transposed(0, 2, 1, 3).reshaped(n * p, n * q)
    }
}

public final class PHMLinear: Linear, ParameterEfficientLayer {
    private let _basisBank: PHMBasisBank
    @ParameterInfo(key: "phm_a") public var ownedBasis: MLXArray?
    @ParameterInfo(key: "phm_s") public var factorS: MLXArray
    @ParameterInfo(key: "phm_t") public var factorT: MLXArray
    public let hypercomplexDim: Int
    public let factorRank: Int
    public let adapterScale: Float
    public var loraEnabled: Bool = true

    public var activeBasis: MLXArray { ownedBasis ?? _basisBank.ownerBasis?() ?? _basisBank.basis }

    private init(
        validated linear: Linear,
        hypercomplexDim: Int,
        factorRank: Int,
        scale: Float,
        basisMode: PHMBasisMode
    ) {
        let (outputDims, inputDims) = linear.shape
        let (bank, owned) = PHMContraction.resolveBasis(basisMode, hypercomplexDim: hypercomplexDim)
        let (s, t) = PHMContraction.initialFactors(
            inputDims: inputDims,
            outputDims: outputDims,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank
        )
        self._basisBank = bank
        self._ownedBasis.wrappedValue = owned
        self._factorS.wrappedValue = s
        self._factorT.wrappedValue = t
        self.hypercomplexDim = hypercomplexDim
        self.factorRank = factorRank
        self.adapterScale = scale
        super.init(weight: linear.weight, bias: linear.bias)
        if case .owned(let sharedBank) = basisMode {
            sharedBank.ownerBasis = { [weak self] in self?.ownedBasis }
        }
        train(linear.training)
        freeze()
    }

    public convenience init(
        linear: Linear,
        hypercomplexDim: Int,
        factorRank: Int = 1,
        scale: Float = 20.0,
        basisMode: PHMBasisMode = .independent
    ) throws(BasaltError) {
        let (outputDims, inputDims) = linear.shape
        try PHMContraction.validate(
            inputDims: inputDims,
            outputDims: outputDims,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale
        )
        try PHMContraction.validateBank(basisMode, hypercomplexDim: hypercomplexDim)
        self.init(
            validated: linear,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale,
            basisMode: basisMode
        )
    }

    public convenience init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        hypercomplexDim: Int,
        factorRank: Int = 1,
        scale: Float = 20.0,
        bias: Bool = false
    ) throws(BasaltError) {
        try PHMContraction.validate(
            inputDims: inputDimensions,
            outputDims: outputDimensions,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale
        )
        let base = Linear(inputDimensions, outputDimensions, bias: bias)
        self.init(
            validated: base,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale,
            basisMode: .independent
        )
    }

    public static func from(
        linear: Linear,
        hypercomplexDim: Int,
        factorRank: Int = 1,
        scale: Float = 20.0,
        basisMode: PHMBasisMode = .independent
    ) throws(BasaltError) -> any ParameterEfficientLayer {
        if let quantized = linear as? QuantizedLinear {
            return try QuantizedPHMLinear(
                linear: quantized,
                hypercomplexDim: hypercomplexDim,
                factorRank: factorRank,
                scale: scale,
                basisMode: basisMode
            )
        }
        return try PHMLinear(
            linear: linear,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale,
            basisMode: basisMode
        )
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let local = keys ?? filterMap(filter: Self.filterLocalParameters).flattened().map(\.0)
        let filtered = local.filter { !PHMContraction.trainableKeys.contains($0) }
        try super.freeze(recursive: recursive, keys: filtered, strict: strict)
    }

    public override func describeExtra(_ indent: Int) -> String {
        let (outputDims, inputDims) = shape
        return "(inputDimensions=\(inputDims), outputDimensions=\(outputDims), hypercomplexDim=\(hypercomplexDim), factorRank=\(factorRank))"
    }

    public func materializeDeltaWeight() -> MLXArray {
        PHMContraction.materializeDelta(basis: activeBasis, factorS: factorS, factorT: factorT)
    }

    public func fused() -> Module {
        let delta = (adapterScale * materializeDeltaWeight()).asType(weight.dtype)
        return Linear(weight: weight + delta, bias: bias)
    }

    public func reverted() -> Module {
        Linear(weight: weight, bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(weight.dtype))
        if !loraEnabled { return y }
        let z = PHMContraction.forward(x, basis: activeBasis, factorS: factorS, factorT: factorT)
        return y + (adapterScale * z).asType(y.dtype)
    }
}

public final class QuantizedPHMLinear: QuantizedLinear, ParameterEfficientLayer {
    private let _basisBank: PHMBasisBank
    @ParameterInfo(key: "phm_a") public var ownedBasis: MLXArray?
    @ParameterInfo(key: "phm_s") public var factorS: MLXArray
    @ParameterInfo(key: "phm_t") public var factorT: MLXArray
    public let hypercomplexDim: Int
    public let factorRank: Int
    public let adapterScale: Float
    public var loraEnabled: Bool = true

    public var activeBasis: MLXArray { ownedBasis ?? _basisBank.ownerBasis?() ?? _basisBank.basis }

    private init(
        validated linear: QuantizedLinear,
        hypercomplexDim: Int,
        factorRank: Int,
        scale: Float,
        basisMode: PHMBasisMode
    ) {
        let (outputDims, inputDims) = linear.shape
        let (bank, owned) = PHMContraction.resolveBasis(basisMode, hypercomplexDim: hypercomplexDim)
        let (s, t) = PHMContraction.initialFactors(
            inputDims: inputDims,
            outputDims: outputDims,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank
        )
        self._basisBank = bank
        self._ownedBasis.wrappedValue = owned
        self._factorS.wrappedValue = s
        self._factorT.wrappedValue = t
        self.hypercomplexDim = hypercomplexDim
        self.factorRank = factorRank
        self.adapterScale = scale
        super.init(
            weight: linear.weight,
            scales: linear.scales,
            biases: linear.biases,
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode
        )
        if let baseBias = linear.bias {
            update(parameters: .unflattened(["bias": baseBias]))
        }
        if case .owned(let sharedBank) = basisMode {
            sharedBank.ownerBasis = { [weak self] in self?.ownedBasis }
        }
        train(linear.training)
        freeze()
    }

    public convenience init(
        linear: QuantizedLinear,
        hypercomplexDim: Int,
        factorRank: Int = 1,
        scale: Float = 20.0,
        basisMode: PHMBasisMode = .independent
    ) throws(BasaltError) {
        guard linear.globalScale == nil else {
            throw .invalidRequestStructure(details: "phm: QuantizedLinear with globalScale is not supported")
        }
        let (outputDims, inputDims) = linear.shape
        try PHMContraction.validate(
            inputDims: inputDims,
            outputDims: outputDims,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale
        )
        try PHMContraction.validateBank(basisMode, hypercomplexDim: hypercomplexDim)
        self.init(
            validated: linear,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale,
            basisMode: basisMode
        )
    }

    public static func from(
        linear: QuantizedLinear,
        hypercomplexDim: Int,
        factorRank: Int = 1,
        scale: Float = 20.0,
        basisMode: PHMBasisMode = .independent
    ) throws(BasaltError) -> QuantizedPHMLinear {
        try QuantizedPHMLinear(
            linear: linear,
            hypercomplexDim: hypercomplexDim,
            factorRank: factorRank,
            scale: scale,
            basisMode: basisMode
        )
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let local = keys ?? filterMap(filter: Self.filterLocalParameters).flattened().map(\.0)
        let filtered = local.filter { !PHMContraction.trainableKeys.contains($0) }
        try super.freeze(recursive: recursive, keys: filtered, strict: strict)
    }

    public override func describeExtra(_ indent: Int) -> String {
        let (outputDims, inputDims) = shape
        return "(inputDimensions=\(inputDims), outputDimensions=\(outputDims), hypercomplexDim=\(hypercomplexDim), factorRank=\(factorRank))"
    }

    public func materializeDeltaWeight() -> MLXArray {
        PHMContraction.materializeDelta(basis: activeBasis, factorS: factorS, factorT: factorT)
    }

    public func fused() -> Module {
        let baseWeight = dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            globalScale: globalScale
        )
        let delta = (adapterScale * materializeDeltaWeight()).asType(baseWeight.dtype)
        return QuantizedLinear(
            weight: baseWeight + delta,
            bias: bias,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            globalScale: globalScale
        )
    }

    public func reverted() -> Module {
        QuantizedLinear(
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            globalScale: globalScale
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x.asType(scales.dtype))
        if !loraEnabled { return y }
        let z = PHMContraction.forward(x, basis: activeBasis, factorS: factorS, factorT: factorT)
        return y + (adapterScale * z).asType(y.dtype)
    }
}
