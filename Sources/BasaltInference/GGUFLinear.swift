import BasaltCore
import Foundation
import MLX
import MLXNN

enum GGUFKernels {
    static let matvecRows = 1 ... 8
    static let rowsPerSimdgroup = Int(ProcessInfo.processInfo.environment["GGUF_NR"] ?? "") ?? 4
    static let simdgroups = Int(ProcessInfo.processInfo.environment["GGUF_SG"] ?? "") ?? 2
    static let variant = Int(ProcessInfo.processInfo.environment["GGUF_V"] ?? "") ?? 0

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

    static func matvec(_ x: MLXArray, _ packed: MLXArray, type: GGMLType) -> MLXArray {
        let (rows, columns, outputs) = (x.dim(0), x.dim(1), packed.dim(0))
        let groups = (outputs + simdgroups * rowsPerSimdgroup - 1) / (simdgroups * rowsPerSimdgroup)
        return matvecKernel(
            [x, packed, deviceGrid], template: template(type, columns: columns, dtype: x.dtype) + [("M", rows), ("NR", rowsPerSimdgroup), ("SG", simdgroups), ("V", variant)],
            grid: (32, simdgroups * groups, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
    }

    static var deviceGrid: MLXArray { MLXArray(GGUFMetalSource.iq3sGrid) }

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
        threadgroup ulong grid[V == 11 ? 1 : GN];
        gguf_load_grid<Q, V == 11 ? 1 : GN>(grid, thread_index_in_threadgroup, 32u * uint(SG));
        threadgroup uint sgrid[V == 11 ? 8192 : 1];
        if (V == 11) gguf_load_signed_grid(sgrid, thread_index_in_threadgroup, 32u * uint(SG));
        threadgroup float4 fgrid[V == 10 ? 512 : 1];
        threadgroup uint4 masks[V == 10 ? 16 : 1];
        if (V == 10) gguf_load_float_grid(fgrid, masks, thread_index_in_threadgroup, 32u * uint(SG));
        uint lane = thread_index_in_simdgroup;
        uint row0 = (threadgroup_position_in_grid.y * uint(SG) + simdgroup_index_in_threadgroup) * uint(NR);
        float acc[NR][M];
        for (uint r = 0; r < uint(NR); ++r) for (uint m = 0; m < uint(M); ++m) acc[r][m] = 0.0f;
        for (uint g = lane; g < GPR; g += 32u) {
            if (M == 1) {
                float4 xr[8];
                device const T* xs = x + g * 32u;
                for (uint j = 0; j < 8u; ++j) xr[j] = float4(xs[4 * j], xs[4 * j + 1], xs[4 * j + 2], xs[4 * j + 3]);
                if (V == 12) {
                    for (uint r = 0; r < uint(NR); ++r)
                        acc[r][0] += gguf_dot_iq3s_device(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xr, dg);
                    continue;
                }
                if (V == 11) {
                    for (uint r = 0; r < uint(NR); ++r)
                        acc[r][0] += gguf_dot_iq3s_signed(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xr, sgrid);
                    continue;
                }
                if (V == 10) {
                    for (uint r = 0; r < uint(NR); ++r)
                        acc[r][0] += gguf_dot_iq3s_float(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xr, fgrid, masks);
                    continue;
                }
                if (V == 0 || V >= 6) {
                    for (uint r = 0; r < uint(NR); ++r)
                        acc[r][0] += gguf_dot32<Q, V>(w + min(row0 + r, N - 1) * ROW + (g / SUB) * BB, g % SUB, xr, grid);
                    continue;
                }
                for (uint r = 0; r < uint(NR) && row0 + r < N; ++r) {
                    if (V == 1) { acc[r][0] += xr[0].x * float(w[(row0 + r) * ROW + (g / SUB) * BB + (g % SUB)]); continue; }
                    if (V == 3) {
                        device const uchar* b = w + (row0 + r) * ROW + (g / SUB) * BB;
                        device const ushort* qs = (device const ushort*)(b + 2 + (g % SUB) * 8);
                        device const ushort* sg = (device const ushort*)(b + 74 + (g % SUB) * 4);
                        float t = float(b[106 + (g % SUB) / 2]) + float(b[66 + (g % SUB)]) + gguf_h16(b, 0);
                        for (uint l = 0; l < 4; ++l) t += float(qs[l]);
                        t += float(sg[0]) + float(sg[1]);
                        acc[r][0] += t * xr[0].x;
                        continue;
                    }
                    if (V == 4) { acc[r][0] += gguf_dot32<Q>(w + (g % SUB) * 0, g % SUB, xr, grid); continue; }
                    if (V == 2) { acc[r][0] += gguf_dot32<12>(w + (row0 + r) * ROW + (g / SUB) * BB, g % SUB, xr, grid); continue; }
                    acc[r][0] += gguf_dot32<Q>(w + (row0 + r) * ROW + (g / SUB) * BB, g % SUB, xr, grid);
                }
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
    name: "gguf_matvec", inputNames: ["x", "w", "dg"], outputNames: ["out"], source: GGUFKernels.matvecSource, header: GGUFMetalSource.header)

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
        let shape = Array(x.shape.dropLast()) + [weight.dim(0)]
        guard GGUFKernels.matvecRows.contains(rows) else {
            return matmul(x, GGUFKernels.dequantize(weight, type: type, columns: inputs, dtype: x.dtype).T)
        }
        return GGUFKernels.matvec(x.reshaped(rows, inputs), weight, type: type).reshaped(shape)
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
        matmul(x, GGUFKernels.dequantize(weight, type: type, columns: dimensions, dtype: x.dtype).T)
    }
}
