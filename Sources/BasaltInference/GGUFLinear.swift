import BasaltCore
import Foundation
import MLX
import MLXNN

enum GGUFKernels {
    static let matvecRows = 1 ... 8
    static let rowsPerSimdgroup = 4
    static let simdgroups = 2
    static let tileBytes = 64 << 20
    private static let supportedDTypes: Set<DType> = [.float16, .bfloat16, .float32]

    static func template(_ type: GGMLType, dtype: DType) -> [(String, any KernelTemplateArg)] {
        [("Q", Int(type.rawValue)), ("BE", type.block.elements), ("BB", type.block.bytes),
         ("GN", GGUFMetalSource.gridSize(type)), ("T", dtype)]
    }

    private static func cpuDequantize(_ packed: MLXArray, type: GGMLType, rows: Int, columns: Int, dtype: DType) -> MLXArray {
        let bytes = packed.asType(.uint8).asArray(UInt8.self)
        let floats = GGUFDequant.floats(bytes.span.bytes, type: type, count: rows * columns)
        return MLXArray(floats, [rows, columns]).asType(dtype)
    }

    private static func onCPUIfFloat64(_ dtypes: [DType?], _ body: () -> MLXArray) -> MLXArray {
        Device.defaultDevice() == .gpu && dtypes.contains(.float64) ? Device.withDefaultDevice(.cpu, body) : body()
    }

    static func dequantize(_ packed: MLXArray, type: GGMLType, columns: Int, dtype: DType) -> MLXArray {
        onCPUIfFloat64([packed.dtype, dtype]) {
            let rows = packed.ndim > 0 ? packed.dim(0) : 0
            guard rows > 0, columns > 0 else {
                return MLXArray.zeros([rows, max(0, columns)], dtype: dtype)
            }
            let blockElements = type.block.elements
            let expectedRowBytes = blockElements > 0 ? (columns / blockElements) * type.block.bytes : 0
            guard Device.defaultDevice() == .gpu,
                  blockElements >= 32,
                  packed.ndim == 2,
                  packed.dtype == .uint8,
                  supportedDTypes.contains(dtype),
                  columns.isMultiple(of: blockElements),
                  packed.dim(1) == expectedRowBytes
            else {
                return cpuDequantize(packed, type: type, rows: rows, columns: columns, dtype: dtype)
            }
            return dequantKernel(
                [packed], template: template(type, dtype: dtype) + [("K", columns)],
                grid: ((rows * columns / 32 + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, columns]], outputDTypes: [dtype])[0]
        }
    }

    static func matmul(_ x: MLXArray, _ packed: MLXArray, type: GGMLType, columns: Int, tileBytes: Int = tileBytes) -> MLXArray {
        onCPUIfFloat64([x.dtype, packed.dtype]) {
            let rows = packed.ndim > 0 ? packed.dim(0) : 0
            let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [rows]
            guard x.ndim >= 1, rows > 0, columns > 0, x.size > 0 else {
                return MLXArray.zeros(outShape, dtype: x.dtype)
            }
            let inexact = x.dtype.isFloatingPoint || x.dtype.isComplex
            let computeType: DType = inexact ? x.dtype : .float32
            let xIn = inexact ? x : x.asType(.float32)
            let rowBytes = max(1, columns * max(1, computeType.size))
            let tile = max(256, tileBytes / rowBytes / 256 * 256)
            let parts = stride(from: 0, to: rows, by: tile).map { start in
                MLX.matmul(xIn, dequantize(packed[start ..< min(rows, start + tile)], type: type, columns: columns, dtype: computeType).T)
            }
            let combined = parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
            return inexact ? combined : combined.asType(x.dtype)
        }
    }

    static func matvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        onCPUIfFloat64([x.dtype, packed.dtype]) {
            let leading = x.ndim > 0 ? Array(x.shape.dropLast()) : []
            let columns = x.ndim > 0 ? x.dim(-1) : 0
            let rows = columns > 0 ? x.size / columns : 0
            let outputs = packed.ndim > 0 ? packed.dim(0) : 0
            let outShape = leading + [outputs]
            guard rows > 0, columns > 0, outputs > 0 else {
                return MLXArray.zeros(outShape, dtype: x.dtype)
            }
            let x2d = x.ndim == 2 ? x : x.reshaped(rows, columns)
            let iqReady = IQMetalKernels.supports(type) && supportedDTypes.contains(x2d.dtype)
                && packed.ndim == 2 && packed.dtype == .uint8
            let result: MLXArray
            if Device.defaultDevice() != .gpu {
                result = matmul(x2d, packed, type: type, columns: columns)
            } else if iqReady, rows == 1 {
                result = IQMetalKernels.matvec(x2d, packed, type: type)
            } else if iqReady {
                result = IQMetalKernels.smallM(x2d, packed, type: type)
            } else {
                result = legacyMatvec(x2d, packed, type: type)
            }
            return result.reshaped(outShape)
        }
    }

    static func legacyMatvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        onCPUIfFloat64([x.dtype, packed.dtype]) {
            let leading = x.ndim > 0 ? Array(x.shape.dropLast()) : []
            let columns = x.ndim > 0 ? x.dim(-1) : 0
            let rows = columns > 0 ? x.size / columns : 0
            let outputs = packed.ndim > 0 ? packed.dim(0) : 0
            let outShape = leading + [outputs]
            guard rows > 0, columns > 0, outputs > 0 else {
                return MLXArray.zeros(outShape, dtype: x.dtype)
            }
            let x2d = x.ndim == 2 ? x : x.reshaped(rows, columns)
            let blockElements = type.block.elements
            let expectedRowBytes = blockElements > 0 ? (columns / blockElements) * type.block.bytes : 0
            guard Device.defaultDevice() == .gpu,
                  blockElements >= 32,
                  matvecRows.contains(rows),
                  columns.isMultiple(of: blockElements),
                  packed.ndim == 2,
                  packed.dtype == .uint8,
                  supportedDTypes.contains(x2d.dtype),
                  packed.dim(1) == expectedRowBytes
            else {
                return matmul(x2d, packed, type: type, columns: columns).reshaped(outShape)
            }
            let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
            let out = matvecKernel(
                [x2d, packed], template: template(type, dtype: x2d.dtype) + [("K", columns), ("M", rows), ("NR", rowsPerSimdgroup), ("SG", simdgroups)],
                grid: (32, simdgroups * groups, 1), threadGroup: (32, simdgroups, 1),
                outputShapes: [[rows, outputs]], outputDTypes: [x2d.dtype])[0]
            return out.reshaped(outShape)
        }
    }

    static func halfGrid(_ type: GGMLType, dtype: DType) -> Bool {
        dtype != .float32 && (type == .iq3S || type == .iq3XXS)
    }

    static let dequantSource = """
        const uint N = uint(w_shape[0]);
        const size_t ROW = size_t(w_shape[1]);
        constexpr uint GPR = uint(K) / 32u;
        constexpr uint SUB = uint(BE) / 32u;
        threadgroup ulong grid[GN];
        gguf_load_grid<Q, GN>(grid, thread_index_in_threadgroup, 256u);
        const uint gid = thread_position_in_grid.x;
        if (gid >= N * GPR) return;
        const size_t row = size_t(gid / GPR);
        const uint g = gid % GPR;
        float y[32];
        gguf_decode32<Q>(w + row * ROW + size_t(g / SUB) * size_t(BB), g % SUB, y, grid);
        for (uint j = 0; j < 32u; ++j) out[row * size_t(K) + size_t(g) * size_t(32) + size_t(j)] = static_cast<T>(y[j]);
        """

    static let matvecSource = """
        const size_t N = size_t(w_shape[0]);
        const size_t ROW = size_t(w_shape[1]);
        constexpr uint GPR = uint(K) / 32u;
        constexpr uint SUB = uint(BE) / 32u;
        if (N == 0 || GPR == 0u) return;
        threadgroup ulong grid[GN];
        gguf_load_grid<Q, GN>(grid, thread_index_in_threadgroup, 32u * uint(SG));
        const uint lane = thread_index_in_simdgroup;
        const size_t row0 = (size_t(threadgroup_position_in_grid.y) * size_t(SG) + size_t(simdgroup_index_in_threadgroup)) * size_t(NR);
        float acc[NR][M];
        for (uint r = 0; r < uint(NR); ++r) for (uint m = 0; m < uint(M); ++m) acc[r][m] = 0.0f;
        for (uint g = lane; g < GPR; g += 32u) {
            const size_t blkOff = size_t(g / SUB) * size_t(BB);
            const uint subIdx = g % SUB;
            if (M == 1) {
                float4 xr[8];
                auto xs = x + size_t(g) * size_t(32);
                for (uint j = 0; j < 8u; ++j) xr[j] = float4(xs[4 * j], xs[4 * j + 1], xs[4 * j + 2], xs[4 * j + 3]);
                for (uint r = 0; r < uint(NR); ++r) {
                    const size_t rn = min(row0 + size_t(r), N - size_t(1));
                    acc[r][0] += gguf_dot32<Q>(w + rn * ROW + blkOff, subIdx, xr, grid);
                }
                continue;
            }
            for (uint r = 0; r < uint(NR) && row0 + size_t(r) < N; ++r) {
                float y[32];
                gguf_decode32<Q>(w + (row0 + size_t(r)) * ROW + blkOff, subIdx, y, grid);
                for (uint m = 0; m < uint(M); ++m) {
                    auto xs = x + size_t(m) * size_t(K) + size_t(g) * size_t(32);
                    float s = 0.0f;
                    for (uint j = 0; j < 32u; ++j) s += y[j] * float(xs[j]);
                    acc[r][m] += s;
                }
            }
        }
        for (uint r = 0; r < uint(NR) && row0 + size_t(r) < N; ++r)
            for (uint m = 0; m < uint(M); ++m) {
                float v = simd_sum(acc[r][m]);
                if (lane == 0u) out[size_t(m) * N + row0 + size_t(r)] = static_cast<T>(v);
            }
        """
}

private let dequantKernel = MLXFast.metalKernel(
    name: "gguf_dequant", inputNames: ["w"], outputNames: ["out"], source: GGUFKernels.dequantSource, header: GGUFMetalSource.header)

private let matvecKernel = MLXFast.metalKernel(
    name: "gguf_matvec", inputNames: ["x", "w"], outputNames: ["out"], source: GGUFKernels.matvecSource, header: GGUFMetalSource.header)

final class GGUFLinear: Linear {
    let type: GGMLType
    let inputs: Int
    let hadamardBlock: Int?
    @ParameterInfo(key: "input_order") var inputOrder: MLXArray?
    @ParameterInfo(key: "pre_scale") var preScale: MLXArray?

    init(
        type: GGMLType,
        outputs: Int,
        inputs: Int,
        permutedInput: Bool = false,
        hadamardBlock: Int? = nil,
        hasPreScale: Bool = false
    ) {
        (self.type, self.inputs, self.hadamardBlock) = (type, inputs, hadamardBlock)
        self._inputOrder.wrappedValue = permutedInput ? MLXArray.zeros([inputs], dtype: .int32) : nil
        self._preScale.wrappedValue = hasPreScale ? MLXArray.ones([inputs], dtype: .float32) : nil
        super.init(weight: MLXArray.zeros([outputs, inputs / type.block.elements * type.block.bytes], dtype: .uint8))
        freeze()
        train(false)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Device.defaultDevice() == .gpu, [x, weight, preScale, inputOrder].contains(where: { $0?.dtype == .float64 }) {
            return Device.withDefaultDevice(.cpu) { self(x) }
        }
        let outputs = weight.ndim > 0 ? weight.dim(0) : 0
        let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
        guard x.ndim >= 1, inputs > 0, outputs > 0, x.size > 0 else {
            return MLXArray.zeros(outShape, dtype: x.dtype)
        }
        let origType = x.dtype
        let inexact = origType.isFloatingPoint || origType.isComplex
        var x = inputOrder.map { x.take($0, axis: -1) } ?? x
        if !inexact { x = x.asType(.float32) }
        if let block = hadamardBlock, block > 1, x.dim(-1) == inputs {
            let scale = preScale ?? MLXArray.ones([inputs], dtype: x.dtype)
            if training {
                let xs = x.asType(.float32) * scale.asType(.float32)
                x = hadamardTransform(xs.reshaped(-1, block), scale: 1.0 / Float(block).squareRoot()).reshaped(x.shape).asType(x.dtype)
            } else {
                x = MACaQFusedKernels.blockHadamard(x, scaleVector: scale, blockSize: block, isInverse: false)
            }
        } else if let preScale {
            x = (x.asType(.float32) * preScale.asType(.float32)).asType(x.dtype)
        }
        let out: MLXArray
        if !training, Device.defaultDevice() == .gpu, x.dim(-1) == inputs, inputs.isMultiple(of: type.block.elements),
           GGUFKernels.matvecRows.contains(x.size / inputs) {
            out = GGUFKernels.matvec(x.reshaped(x.size / inputs, inputs), weight, type: type).reshaped(outShape)
        } else {
            out = GGUFKernels.matmul(x, weight, type: type, columns: inputs)
        }
        return inexact ? out : out.asType(origType)
    }
}

final class GGUFEmbedding: Embedding {
    let type: GGMLType
    let dimensions: Int
    let dtype: DType

    init(type: GGMLType, count: Int, dimensions: Int, dtype: DType) {
        (self.type, self.dimensions, self.dtype) = (type, dimensions, dtype)
        super.init(weight: MLXArray.zeros([count, dimensions / type.block.elements * type.block.bytes], dtype: .uint8))
        freeze()
        train(false)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || weight.dtype == .float64 || dtype == .float64) {
            return Device.withDefaultDevice(.cpu) { self(x) }
        }
        guard x.size > 0, dimensions > 0, weight.dim(0) > 0 else {
            return MLXArray.zeros(x.shape + [max(0, dimensions)], dtype: dtype)
        }
        let rows = weight.take(x.flattened(), axis: 0)
        return GGUFKernels.dequantize(rows, type: type, columns: dimensions, dtype: dtype).reshaped(x.shape + [dimensions])
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || weight.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) { asLinear(x) }
        }
        let outputs = weight.ndim > 0 ? weight.dim(0) : 0
        let outShape = (x.ndim > 0 ? Array(x.shape.dropLast()) : []) + [outputs]
        guard x.ndim >= 1, dimensions > 0, outputs > 0, x.size > 0 else {
            return MLXArray.zeros(outShape, dtype: x.dtype)
        }
        guard !training,
              Device.defaultDevice() == .gpu,
              x.dim(-1) == dimensions,
              dimensions.isMultiple(of: type.block.elements)
        else {
            return GGUFKernels.matmul(x, weight, type: type, columns: dimensions)
        }
        let rows = x.size / dimensions
        guard GGUFKernels.matvecRows.contains(rows) else {
            return GGUFKernels.matmul(x, weight, type: type, columns: dimensions)
        }
        return GGUFKernels.matvec(x.reshaped(rows, dimensions), weight, type: type).reshaped(outShape)
    }
}
