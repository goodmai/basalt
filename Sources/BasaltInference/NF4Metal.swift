import Foundation
import MLX

public enum NF4Error: Error, Equatable, Sendable {
    case groupSize(Int)
    case shape(String)
}

public enum NF4Metal {
    public static let method = "nf4"
    public static let groupSizes: Set<Int> = [32, 64, 128]
    public static let dtypes: Set<DType> = [.float16, .bfloat16, .float32]
    public static let maxKernelRows = 16

    public static let codebook: [Float] = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    public static let midpoints: [Float] = zip(codebook, codebook.dropFirst()).map { ($0 + $1) * 0.5 }

    public static func quantized(_ weight: MLXArray, groupSize: Int = 32) -> (wq: MLXArray, scales: MLXArray) {
        precondition(groupSizes.contains(groupSize), "NF4 groupSize must be 32, 64, or 128")
        let columns = weight.dim(-1)
        precondition(weight.ndim >= 2 && columns % groupSize == 0, "NF4 input dimension \(columns) must be divisible by \(groupSize)")
        let leading = Array(weight.shape.dropLast())
        let groups = weight.asType(.float32).reshaped(-1, groupSize)
        let scales = abs(groups).max(axis: -1, keepDims: true).asType(.float16)
        let scale = scales.asType(.float32)
        let normalized = groups / which(scale .> 0, scale, 1)
        var codes = MLXArray.zeros(normalized.shape, dtype: .uint32)
        for mid in midpoints {
            codes = codes + (normalized .> mid).asType(.uint32)
        }
        let shifts = MLXArray((0 ..< 8).map { UInt32(1) << UInt32(4 * $0) })
        let packed = (codes.reshaped(-1, 8) * shifts).sum(axis: -1).asType(.uint32)
        return (
            wq: packed.reshaped(leading + [columns / 8]),
            scales: scales.reshaped(leading + [columns / groupSize])
        )
    }

    public static func dequantized(
        _ codes: MLXArray, scales: MLXArray, groupSize: Int = 32, dtype: DType? = nil
    ) -> MLXArray {
        let leading = Array(codes.shape.dropLast())
        let words = codes.dim(-1)
        let columns = words * 8
        let groups = columns / groupSize
        let shifts = MLXArray((0 ..< 8).map { UInt32(4 * $0) }).reshaped(Array(repeating: 1, count: codes.ndim) + [8])
        let nibbles = ((codes.expandedDimensions(axis: -1) >> shifts) & UInt32(15))
            .asType(.int32)
            .reshaped(leading + [groups, groupSize])
        let values = MLXArray(codebook).take(nibbles)
        let restored = (values * scales.asType(.float32).expandedDimensions(axis: -1))
            .reshaped(leading + [columns])
        return restored.asType(dtype ?? scales.dtype)
    }

    public static func matmul(
        _ x: MLXArray, codes: MLXArray, scales: MLXArray, groupSize: Int = 32
    ) -> MLXArray {
        let inputs = codes.dim(-1) * 8
        let outputs = codes.dim(0)
        let leading = Array(x.shape.dropLast())
        guard inputs > 0, outputs > 0, x.dim(-1) == inputs, scales.dim(-1) * groupSize == inputs else {
            return outputs == 0 ? MLXArray.zeros(leading + [0], dtype: x.dtype) : MLX.matmul(x, dequantized(codes, scales: scales, groupSize: groupSize, dtype: x.dtype).T)
        }
        let rows = x.size / inputs
        guard rows > 0 else {
            return MLXArray.zeros(leading + [outputs], dtype: x.dtype)
        }
        if rows <= maxKernelRows, groupSizes.contains(groupSize), dtypes.contains(x.dtype), codes.ndim == 2 {
            let rowGroups = (outputs + 3) / 4
            let tgRows = min(max(1, rowGroups), 2)
            let paddedGroups = ((rowGroups + tgRows - 1) / tgRows) * tgRows
            let out = nf4QmvKernel(
                [x.reshaped(rows, inputs), codes, scales],
                template: [("T", x.dtype), ("G", groupSize), ("M", rows)],
                grid: (32, paddedGroups, 1),
                threadGroup: (32, tgRows, 1),
                outputShapes: [[rows, outputs]],
                outputDTypes: [x.dtype]
            )[0]
            return out.reshaped(leading + [outputs])
        }
        let dense = dequantized(codes, scales: scales, groupSize: groupSize, dtype: x.dtype)
        return MLX.matmul(x, dense.T)
    }

    public static func gatherMatmul(
        _ x: MLXArray, codes: MLXArray, scales: MLXArray, rhsIndices: MLXArray, groupSize: Int = 32
    ) -> MLXArray {
        let batchShape = rhsIndices.shape
        let batch = rhsIndices.size
        let inputs = codes.dim(-1) * 8
        let outputs = codes.dim(1)
        let rows = x.ndim >= 2 ? x.dim(-2) : 0
        guard batch > 0, rows > 0, outputs > 0, inputs > 0, x.dim(-1) == inputs, scales.dim(-1) * groupSize == inputs else {
            return MLXArray.zeros(batchShape + [max(0, rows), max(0, outputs)], dtype: x.dtype)
        }
        let xExpanded = x.shape == (batchShape + [rows, inputs]) ? x : broadcast(x, to: batchShape + [rows, inputs])
        if rows <= maxKernelRows, groupSizes.contains(groupSize), dtypes.contains(x.dtype), codes.ndim == 3 {
            let rowGroups = (outputs + 3) / 4
            let tgRows = min(max(1, rowGroups), 2)
            let paddedGroups = ((rowGroups + tgRows - 1) / tgRows) * tgRows
            let out = nf4GatherQmvKernel(
                [xExpanded.reshaped(batch, rows, inputs), codes, scales, rhsIndices.reshaped(batch)],
                template: [("T", x.dtype), ("G", groupSize), ("M", rows)],
                grid: (32, paddedGroups, batch),
                threadGroup: (32, tgRows, 1),
                outputShapes: [[batch, rows, outputs]],
                outputDTypes: [x.dtype],
                initValue: 0
            )[0]
            return out.reshaped(batchShape + [rows, outputs])
        }
        let numExperts = codes.dim(0)
        let picks = rhsIndices.reshaped(batch).asType(.int32).asArray(Int32.self)
        let x3D = xExpanded.reshaped(batch, rows, inputs)
        let pieces = picks.enumerated().map { idx, expert -> MLXArray in
            guard expert >= 0, Int(expert) < numExperts else {
                return MLXArray.zeros([1, rows, outputs], dtype: x.dtype)
            }
            let dense = dequantized(codes[Int(expert)], scales: scales[Int(expert)], groupSize: groupSize, dtype: x.dtype)
            return MLX.matmul(x3D[idx], dense.T).expandedDimensions(axis: 0)
        }
        return concatenated(pieces, axis: 0).reshaped(batchShape + [rows, outputs])
    }
}

private enum NF4MetalSource {
    static let header = """
        #include <metal_simdgroup>
        constant float NF4_LUT[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        """

    static let qmv = """
        constexpr int R = 4;
        const uint lane = uint(thread_index_in_simdgroup);
        const uint n0 = uint(thread_position_in_grid.y) * uint(R);
        const uint K = uint(x_shape[1]);
        const uint N = uint(w_shape[0]);
        const uint KW = uint(w_shape[1]);
        const uint KG = uint(scales_shape[1]);
        if (n0 >= N) {
            return;
        }
        float acc[M][R];
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                acc[m][r] = 0.0f;
            }
        }
        for (uint k8 = lane; k8 < KW; k8 += 32u) {
            float xv[M][8];
            for (int m = 0; m < M; m++) {
                const size_t xOff = size_t(m) * size_t(K) + size_t(k8) * 8u;
                for (int j = 0; j < 8; j++) {
                    xv[m][j] = float(x[xOff + size_t(j)]);
                }
            }
            const uint g = (k8 * 8u) / uint(G);
            for (int r = 0; r < R; r++) {
                const uint n = n0 + uint(r);
                if (n >= N) {
                    break;
                }
                const uint32_t word = w[size_t(n) * size_t(KW) + size_t(k8)];
                const float s = float(scales[size_t(n) * size_t(KG) + size_t(g)]);
                for (int m = 0; m < M; m++) {
                    float p = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        p += NF4_LUT[(word >> (4u * uint(j))) & 15u] * xv[m][j];
                    }
                    acc[m][r] += p * s;
                }
            }
        }
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                const float v = simd_sum(acc[m][r]);
                const uint n = n0 + uint(r);
                if (lane == 0u && n < N) {
                    out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(v);
                }
            }
        }
        """

    static let gatherQmv = """
        constexpr int R = 4;
        const uint lane = uint(thread_index_in_simdgroup);
        const uint n0 = uint(thread_position_in_grid.y) * uint(R);
        const uint b = uint(thread_position_in_grid.z);
        const uint B = uint(x_shape[0]);
        const uint K = uint(x_shape[2]);
        const uint E = uint(w_shape[0]);
        const uint N = uint(w_shape[1]);
        const uint KW = uint(w_shape[2]);
        const uint KG = uint(scales_shape[2]);
        if (b >= B || n0 >= N) {
            return;
        }
        const uint e = uint(indices[b]);
        if (e >= E) {
            return;
        }
        const device T* xb = x + size_t(b) * size_t(M) * size_t(K);
        const device uint32_t* we = w + size_t(e) * size_t(N) * size_t(KW);
        const size_t sBase = size_t(e) * size_t(N) * size_t(KG);
        float acc[M][R];
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                acc[m][r] = 0.0f;
            }
        }
        for (uint k8 = lane; k8 < KW; k8 += 32u) {
            float xv[M][8];
            for (int m = 0; m < M; m++) {
                const size_t xOff = size_t(m) * size_t(K) + size_t(k8) * 8u;
                for (int j = 0; j < 8; j++) {
                    xv[m][j] = float(xb[xOff + size_t(j)]);
                }
            }
            const uint g = (k8 * 8u) / uint(G);
            for (int r = 0; r < R; r++) {
                const uint n = n0 + uint(r);
                if (n >= N) {
                    break;
                }
                const uint32_t word = we[size_t(n) * size_t(KW) + size_t(k8)];
                const float s = float(scales[sBase + size_t(n) * size_t(KG) + size_t(g)]);
                for (int m = 0; m < M; m++) {
                    float p = 0.0f;
                    for (int j = 0; j < 8; j++) {
                        p += NF4_LUT[(word >> (4u * uint(j))) & 15u] * xv[m][j];
                    }
                    acc[m][r] += p * s;
                }
            }
        }
        device T* outb = out + size_t(b) * size_t(M) * size_t(N);
        for (int m = 0; m < M; m++) {
            for (int r = 0; r < R; r++) {
                const float v = simd_sum(acc[m][r]);
                const uint n = n0 + uint(r);
                if (lane == 0u && n < N) {
                    outb[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(v);
                }
            }
        }
        """
}

private let nf4QmvKernel = MLXFast.metalKernel(
    name: "basalt_nf4_qmv",
    inputNames: ["x", "w", "scales"],
    outputNames: ["out"],
    source: NF4MetalSource.qmv,
    header: NF4MetalSource.header
)

private let nf4GatherQmvKernel = MLXFast.metalKernel(
    name: "basalt_nf4_gather_qmv",
    inputNames: ["x", "w", "scales", "indices"],
    outputNames: ["out"],
    source: NF4MetalSource.gatherQmv,
    header: NF4MetalSource.header
)
