import Foundation
import MLX
import MLXNN
import MLXLMCommon
import BasaltCore

public protocol OFTAdapterLayer: ParameterEfficientLayer {
    var blockSize: Int { get }
    var totalBlocks: Int { get }
    var parameterization: OrthogonalParameterization { get }
    var blockShared: Bool { get }
    func orthogonalBlocks() -> MLXArray
    func maxOrthogonalityDefect() -> Float
}

public final class OFTLinear: Linear, OFTAdapterLayer {
    private let _skewBasis: OFTSkewBasis
    @ParameterInfo(key: "oft_r") public var oftParams: MLXArray
    public let blockSize: Int
    public let totalBlocks: Int
    public let parameterization: OrthogonalParameterization
    public let blockShared: Bool
    public var adapterScale: Float { 1.0 }
    public var loraEnabled: Bool = true

    private init(
        validated linear: Linear,
        blockSize: Int,
        parameterization: OrthogonalParameterization,
        blockShared: Bool
    ) {
        let (_, inputDims) = linear.shape
        let basis = OFTSkewBasis(blockSize: blockSize)
        let blocks = inputDims / blockSize
        let paramBlocks = blockShared ? 1 : blocks
        self._skewBasis = basis
        self._oftParams.wrappedValue = MLXArray.zeros([paramBlocks, basis.freePerBlock], dtype: .float32)
        self.blockSize = blockSize
        self.totalBlocks = blocks
        self.parameterization = parameterization
        self.blockShared = blockShared
        super.init(weight: linear.weight, bias: linear.bias)
        train(linear.training)
        freeze()
    }

    public convenience init(
        linear: Linear,
        blockSize: Int,
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false
    ) throws(BasaltError) {
        let (outputDims, inputDims) = linear.shape
        try OFTMath.validate(
            inputDims: inputDims,
            outputDims: outputDims,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
        self.init(
            validated: linear,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
    }

    public convenience init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        blockSize: Int,
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false,
        bias: Bool = false
    ) throws(BasaltError) {
        try OFTMath.validate(
            inputDims: inputDimensions,
            outputDims: outputDimensions,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
        let base = Linear(inputDimensions, outputDimensions, bias: bias)
        self.init(
            validated: base,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
    }

    public static func from(
        linear: Linear,
        blockSize: Int,
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false
    ) throws(BasaltError) -> any ParameterEfficientLayer {
        if let quantized = linear as? QuantizedLinear {
            return try QuantizedOFTLinear(
                linear: quantized,
                blockSize: blockSize,
                parameterization: parameterization,
                blockShared: blockShared
            )
        }
        return try OFTLinear(
            linear: linear,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let local = keys ?? filterMap(filter: Self.filterLocalParameters).flattened().map(\.0)
        let filtered = local.filter { !OFTMath.trainableKeys.contains($0) }
        try super.freeze(recursive: recursive, keys: filtered, strict: strict)
    }

    public override func describeExtra(_ indent: Int) -> String {
        let (outputDims, inputDims) = shape
        return "(inputDimensions=\(inputDims), outputDimensions=\(outputDims), blockSize=\(blockSize), parameterization=\(parameterization.rawValue), blockShared=\(blockShared))"
    }

    public func skewBlocks() -> MLXArray {
        OFTMath.skewSymmetric(params: oftParams, basis: _skewBasis)
    }

    public func orthogonalBlocks() -> MLXArray {
        OFTMath.orthogonalBlocks(params: oftParams, basis: _skewBasis, parameterization: parameterization)
    }

    public func maxOrthogonalityDefect() -> Float {
        OFTMath.orthogonalityDefect(orthogonalBlocks())
    }

    public func fusedWeight() -> MLXArray {
        OFTMath.rotateWeight(weight, blocks: orthogonalBlocks()).asType(weight.dtype)
    }

    public func materializeDeltaWeight() -> MLXArray {
        OFTMath.rotateWeight(weight, blocks: orthogonalBlocks()) - weight.asType(.float32)
    }

    public func fused() -> Module {
        Linear(weight: fusedWeight(), bias: bias)
    }

    public func reverted() -> Module {
        Linear(weight: weight, bias: bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let targetDType = weight.dtype
        let baseInput = x.asType(targetDType)
        if !loraEnabled { return super.callAsFunction(baseInput) }
        let rotated = OFTMath.rotateInput(
            baseInput,
            blocks: orthogonalBlocks(),
            totalBlocks: totalBlocks,
            targetDType: targetDType
        )
        return super.callAsFunction(rotated)
    }
}

public final class QuantizedOFTLinear: QuantizedLinear, OFTAdapterLayer {
    private let _skewBasis: OFTSkewBasis
    @ParameterInfo(key: "oft_r") public var oftParams: MLXArray
    public let blockSize: Int
    public let totalBlocks: Int
    public let parameterization: OrthogonalParameterization
    public let blockShared: Bool
    public var adapterScale: Float { 1.0 }
    public var loraEnabled: Bool = true

    private init(
        validated linear: QuantizedLinear,
        blockSize: Int,
        parameterization: OrthogonalParameterization,
        blockShared: Bool
    ) {
        let (_, inputDims) = linear.shape
        let basis = OFTSkewBasis(blockSize: blockSize)
        let blocks = inputDims / blockSize
        let paramBlocks = blockShared ? 1 : blocks
        self._skewBasis = basis
        self._oftParams.wrappedValue = MLXArray.zeros([paramBlocks, basis.freePerBlock], dtype: .float32)
        self.blockSize = blockSize
        self.totalBlocks = blocks
        self.parameterization = parameterization
        self.blockShared = blockShared
        if linear.mode == .affine {
            super.init(weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases, groupSize: linear.groupSize, bits: linear.bits)
        } else {
            super.init(weight: linear.weight, scales: linear.scales, biases: linear.biases, groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode)
        }
        train(linear.training)
        freeze()
    }

    public convenience init(
        linear: QuantizedLinear,
        blockSize: Int,
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false
    ) throws(BasaltError) {
        guard linear.globalScale == nil else {
            throw .invalidRequestStructure(details: "oft: QuantizedLinear with globalScale is not supported")
        }
        let (outputDims, inputDims) = linear.shape
        try OFTMath.validate(
            inputDims: inputDims,
            outputDims: outputDims,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
        self.init(
            validated: linear,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
    }

    public static func from(
        linear: QuantizedLinear,
        blockSize: Int,
        parameterization: OrthogonalParameterization = .cayley,
        blockShared: Bool = false
    ) throws(BasaltError) -> QuantizedOFTLinear {
        try QuantizedOFTLinear(
            linear: linear,
            blockSize: blockSize,
            parameterization: parameterization,
            blockShared: blockShared
        )
    }

    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let local = keys ?? filterMap(filter: Self.filterLocalParameters).flattened().map(\.0)
        let filtered = local.filter { !OFTMath.trainableKeys.contains($0) }
        try super.freeze(recursive: recursive, keys: filtered, strict: strict)
    }

    public override func describeExtra(_ indent: Int) -> String {
        let (outputDims, inputDims) = shape
        return "(inputDimensions=\(inputDims), outputDimensions=\(outputDims), blockSize=\(blockSize), parameterization=\(parameterization.rawValue), blockShared=\(blockShared))"
    }

    public func skewBlocks() -> MLXArray {
        OFTMath.skewSymmetric(params: oftParams, basis: _skewBasis)
    }

    public func orthogonalBlocks() -> MLXArray {
        OFTMath.orthogonalBlocks(params: oftParams, basis: _skewBasis, parameterization: parameterization)
    }

    public func maxOrthogonalityDefect() -> Float {
        OFTMath.orthogonalityDefect(orthogonalBlocks())
    }

    private func dequantizedBaseWeight() -> MLXArray {
        dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            globalScale: globalScale
        )
    }

    public func materializeDeltaWeight() -> MLXArray {
        let base = dequantizedBaseWeight()
        return OFTMath.rotateWeight(base, blocks: orthogonalBlocks()) - base.asType(.float32)
    }

    public func fused() -> Module {
        let base = dequantizedBaseWeight()
        let rotated = OFTMath.rotateWeight(base, blocks: orthogonalBlocks()).asType(base.dtype)
        return QuantizedLinear(
            weight: rotated,
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
        let targetDType = scales.dtype.isFloatingPoint ? scales.dtype : (x.dtype.isFloatingPoint ? x.dtype : .float32)
        let baseInput = x.asType(targetDType)
        if !loraEnabled { return super.callAsFunction(baseInput) }
        let rotated = OFTMath.rotateInput(
            baseInput,
            blocks: orthogonalBlocks(),
            totalBlocks: totalBlocks,
            targetDType: targetDType
        )
        return super.callAsFunction(rotated)
    }
}
