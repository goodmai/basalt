import BasaltCore
import Foundation
import MLX
import MLXNN

enum GGUFKernels {
    static let matvecRows = 1 ... 8
    static let rowsPerSimdgroup = 4
    static let simdgroups = 2
    static let tileBytes = 64 << 20

    static func template(_ type: GGMLType, columns: Int, dtype: DType) -> [(String, any KernelTemplateArg)] {
        [("Q", Int(type.rawValue)), ("K", columns), ("BE", type.block.elements), ("BB", type.block.bytes),
         ("GN", GGUFMetalSource.gridSize(type)), ("T", dtype)]
    }

    static func dequantize(_ packed: MLXArray, type: GGMLType, columns: Int, dtype: DType) -> MLXArray {
        let rows = packed.dim(0)
        return dequantKernel(
            [packed], template: template(type, columns: columns, dtype: dtype),
            grid: ((rows * columns / 32 + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, columns]], outputDTypes: [dtype])[0]
    }

    static func matmul(_ x: MLXArray, _ packed: MLXArray, type: GGMLType, columns: Int, tileBytes: Int = tileBytes) -> MLXArray {
        let rows = packed.dim(0)
        let tile = max(256, tileBytes / (columns * x.dtype.size) / 256 * 256)
        let parts = stride(from: 0, to: rows, by: tile).map { start in
            MLX.matmul(x, dequantize(packed[start ..< min(rows, start + tile)], type: type, columns: columns, dtype: x.dtype).T)
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    static func matvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        let (rows, columns, outputs) = (x.dim(0), x.dim(1), packed.dim(0))
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        return matvecKernel(
            [x, packed], template: template(type, columns: columns, dtype: x.dtype) + [("M", rows), ("NR", rowsPerSimdgroup), ("SG", simdgroups), ("H", halfGrid(type, dtype: x.dtype))],
            grid: (32, simdgroups * groups, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
    }

    static func halfGrid(_ type: GGMLType, dtype: DType) -> Bool {
        (type == .iq3S || type == .iq3XXS) && dtype != .float32
    }

    static let dequantSource = """
        const uint GPR = K / 32u, SUB = BE / 32u, ROW = (K / BE) * BB;
        threadgroup ulong grid[GN];
        gguf_load_grid<Q, GN>(grid, thread_index_in_threadgroup, 256u);
        uint gid = thread_position_in_grid.x;
        if (gid >= uint(w_shape[0]) * GPR) return;
        uint row = gid / GPR, g = gid % GPR;
        float y[32];
        gguf_decode32<Q>(w + row * ROW + (g / SUB) * BB, g % SUB, y, grid);
        for (uint j = 0; j < 32u; ++j) out[row * K + g * 32u + j] = static_cast<T>(y[j]);
        """

    static let matvecSource = """
        const uint GPR = K / 32u, SUB = BE / 32u, ROW = (K / BE) * BB;
        const uint N = uint(w_shape[0]);
        threadgroup ulong grid[GN];
        gguf_load_grid<Q, GN>(grid, thread_index_in_threadgroup, 32u * uint(SG));
        threadgroup ulong hgrid[H ? GN : 1];
        if (H) gguf_load_half_grid<GN>(hgrid, grid, thread_index_in_threadgroup, 32u * uint(SG));
        uint lane = thread_index_in_simdgroup;
        uint row0 = (threadgroup_position_in_grid.y * uint(SG) + simdgroup_index_in_threadgroup) * uint(NR);
        float acc[NR][M];
        for (uint r = 0; r < uint(NR); ++r) for (uint m = 0; m < uint(M); ++m) acc[r][m] = 0.0f;
        for (uint g = lane; g < GPR; g += 32u) {
            if (M == 1 && H) {
                half4 xh[8];
                device const T* xs = x + g * 32u;
                for (uint j = 0; j < 8u; ++j) xh[j] = half4(clamp(float4(xs[4 * j], xs[4 * j + 1], xs[4 * j + 2], xs[4 * j + 3]), -65504.0f, 65504.0f));
                for (uint r = 0; r < uint(NR); ++r)
                    acc[r][0] += gguf_dot32_h<Q>(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xh, hgrid);
                continue;
            }
            if (M == 1) {
                float4 xr[8];
                device const T* xs = x + g * 32u;
                for (uint j = 0; j < 8u; ++j) xr[j] = float4(xs[4 * j], xs[4 * j + 1], xs[4 * j + 2], xs[4 * j + 3]);
                for (uint r = 0; r < uint(NR); ++r)
                    acc[r][0] += gguf_dot32<Q>(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xr, grid);
                continue;
            }
            for (uint r = 0; r < uint(NR) && row0 + r < N; ++r) {
                float y[32];
                gguf_decode32<Q>(w + (row0 + r) * ROW + (g / SUB) * BB, g % SUB, y, grid);
                for (uint m = 0; m < uint(M); ++m) {
                    device const T* xs = x + m * K + g * 32u;
                    float s = 0.0f;
                    for (uint j = 0; j < 32u; ++j) s += y[j] * float(xs[j]);
                    acc[r][m] += s;
                }
            }
        }
        for (uint r = 0; r < uint(NR) && row0 + r < N; ++r)
            for (uint m = 0; m < uint(M); ++m) {
                float v = simd_sum(acc[r][m]);
                if (lane == 0) out[m * N + row0 + r] = static_cast<T>(v);
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
    @ParameterInfo(key: "input_order") var inputOrder: MLXArray?

    init(type: GGMLType, outputs: Int, inputs: Int, permutedInput: Bool = false) {
        (self.type, self.inputs) = (type, inputs)
        self._inputOrder.wrappedValue = permutedInput ? MLXArray.zeros([inputs], dtype: .int32) : nil
        super.init(weight: MLXArray.zeros([outputs, inputs / type.block.elements * type.block.bytes], dtype: .uint8))
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = inputOrder.map { x.take($0, axis: -1) } ?? x
        let rows = x.size / inputs
        guard GGUFKernels.matvecRows.contains(rows) else { return GGUFKernels.matmul(x, weight, type: type, columns: inputs) }
        return GGUFKernels.matvec(x.reshaped(rows, inputs), weight, type: type).reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
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
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rows = weight.take(x.flattened(), axis: 0)
        return GGUFKernels.dequantize(rows, type: type, columns: dimensions, dtype: dtype).reshaped(x.shape + [dimensions])
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        GGUFKernels.matmul(x, weight, type: type, columns: dimensions)
    }
}
