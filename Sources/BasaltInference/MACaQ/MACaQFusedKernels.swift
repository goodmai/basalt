import Foundation
import MLX

package struct MACaQPackedWeights {
    package var weight: MLXArray
    package var scales: MLXArray
    package var biases: MLXArray?

    package init(weight: MLXArray, scales: MLXArray, biases: MLXArray? = nil) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }
}

package enum MACaQFusedKernels {
    package static let dtypes: Set<DType> = [.bfloat16, .float16, .float32]
    package static let maxSingleSimdHadamard = 128

    package static func supportsSingleSimdHadamard(blockSize: Int, dtype: DType) -> Bool {
        blockSize >= 2
            && blockSize <= maxSingleSimdHadamard
            && (blockSize & (blockSize - 1)) == 0
            && dtypes.contains(dtype)
    }

    package static func differentiableBlockHadamard(
        _ x: MLXArray,
        scaleVector: MLXArray,
        blockSize: Int
    ) -> MLXArray {
        let origType = x.dtype
        let clamped = clip(scaleVector.asType(.float32), min: Float(1e-4), max: Float(1e4))
        let scaled = x.asType(.float32) * clamped
        guard blockSize > 1, scaled.ndim > 0, scaled.dim(-1).isMultiple(of: blockSize) else {
            return scaled.asType(origType)
        }
        let scale = 1.0 / Float(blockSize).squareRoot()
        return hadamardTransform(scaled.reshaped(-1, blockSize), scale: scale)
            .reshaped(scaled.shape)
            .asType(origType)
    }

    package static func differentiableSwitchHadamard(
        _ x: MLXArray,
        indices: MLXArray,
        preScale: MLXArray,
        blockSize: Int
    ) -> MLXArray {
        let origType = x.dtype
        let clamped = clip(preScale.asType(.float32), min: Float(1e-4), max: Float(1e4))
        let scaled: MLXArray
        if clamped.ndim == 1 {
            scaled = x.asType(.float32) * clamped
        } else {
            let gathered = expandedDimensions(clamped.take(indices, axis: 0), axis: -2)
            scaled = x.asType(.float32) * gathered
        }
        guard blockSize > 1, scaled.ndim > 0, scaled.dim(-1).isMultiple(of: blockSize) else {
            return scaled.asType(origType)
        }
        let scale = 1.0 / Float(blockSize).squareRoot()
        return hadamardTransform(scaled.reshaped(-1, blockSize), scale: scale)
            .reshaped(scaled.shape)
            .asType(origType)
    }

    package static func blockHadamard(
        _ x: MLXArray,
        scaleVector: MLXArray,
        blockSize: Int,
        isInverse: Bool = false
    ) -> MLXArray {
        let origType = x.dtype
        guard x.ndim > 0, blockSize > 0 else { return x }
        let width = x.dim(-1)
        let rows = width > 0 ? x.size / width : 0
        guard rows > 0, width > 0, width.isMultiple(of: blockSize) else { return x }
        if blockSize == 1 {
            return (x.asType(.float32) * scaleVector.asType(.float32)).asType(origType)
        }
        let numBlocks = width / blockSize
        let totalBlocks = rows * numBlocks
        guard supportsSingleSimdHadamard(blockSize: blockSize, dtype: origType),
              scaleVector.size == width,
              totalBlocks <= Int(Int32.max / 2)
        else {
            return FusedKroneckerRHT.blockHadamardFallback(
                x, signs: scaleVector, blockSize: blockSize, isInverse: isInverse)
        }
        let cleanScale = dtypes.contains(scaleVector.dtype) ? scaleVector : scaleVector.asType(.float32)
        let simdPerTG = totalBlocks.isMultiple(of: 4) ? 4 : (totalBlocks.isMultiple(of: 2) ? 2 : 1)
        let out = singleSimdBlockWHTKernel(
            [x.reshaped(totalBlocks, blockSize), cleanScale.reshaped(width)],
            template: [
                ("T", origType),
                ("TS", cleanScale.dtype),
                ("N", blockSize),
                ("NUM_BLOCKS", numBlocks),
                ("IS_INV", isInverse),
            ],
            grid: (32, totalBlocks, 1),
            threadGroup: (32, simdPerTG, 1),
            outputShapes: [[totalBlocks, blockSize]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(x.shape)
    }

    package static func switchBlockHadamard(
        _ x: MLXArray,
        indices: MLXArray,
        preScale: MLXArray,
        blockSize: Int,
        numExperts: Int
    ) -> MLXArray {
        let origType = x.dtype
        let batchShape = indices.shape
        let rows = x.dim(-2)
        let width = x.dim(-1)
        let targetShape = batchShape + [rows, width]
        let xExp = x.shape == targetShape ? x : broadcast(x, to: targetShape)
        let idxExp = rows == 1
            ? indices.asType(.int32)
            : broadcast(indices.asType(.int32).expandedDimensions(axis: -1), to: batchShape + [rows])
        let flatTasks = indices.size * rows
        guard flatTasks > 0, width > 0, width.isMultiple(of: blockSize),
              supportsSingleSimdHadamard(blockSize: blockSize, dtype: origType) else {
            return differentiableSwitchHadamard(x, indices: indices, preScale: preScale, blockSize: blockSize)
        }
        let numBlocks = width / blockSize
        let cleanScale = dtypes.contains(preScale.dtype) ? preScale : preScale.asType(.float32)
        let simdPerTG = numBlocks.isMultiple(of: 4) ? 4 : (numBlocks.isMultiple(of: 2) ? 2 : 1)
        let out = switchSimdBlockWHTKernel(
            [xExp.reshaped(flatTasks, width), idxExp.reshaped(flatTasks), cleanScale],
            template: [
                ("T", origType),
                ("TS", cleanScale.dtype),
                ("N", blockSize),
                ("K", width),
                ("E", numExperts),
                ("PER_EXPERT_PRE", preScale.ndim == 2),
            ],
            grid: (32, numBlocks, flatTasks),
            threadGroup: (32, simdPerTG, 1),
            outputShapes: [[flatTasks, width]],
            outputDTypes: [origType]
        )[0]
        return out.reshaped(targetShape)
    }
}

private enum MACaQFusedMetal {
    static let header = """
        #include <metal_simdgroup>

        template <int INNER>
        inline float macaq_wht_simd(float v, int lane) {
            for (int h = 1; h < INNER; h <<= 1) {
                const float other = simd_shuffle_xor(v, ushort(h));
                v = other + (((lane & h) != 0) ? -v : v);
            }
            return v;
        }
        """

    static let blockWHTSource = """
        constexpr int R = (N >= 32) ? (N / 32) : 1;
        constexpr int INNER = (N < 32) ? N : 32;
        const int blk_idx = int(thread_position_in_grid.y);
        const int lane = int(thread_index_in_simdgroup);
        const size_t base = size_t(blk_idx) * size_t(N);
        const size_t sign_base = size_t(blk_idx % NUM_BLOCKS) * size_t(N);
        const float scale = rsqrt(float(N));
        float a[R];
        for (int r = 0; r < R; ++r) {
            const int col = r * 32 + lane;
            if (col < N) {
                float xv = float(inp[base + size_t(col)]);
                if (!IS_INV) { xv *= float(signs[sign_base + size_t(col)]); }
                a[r] = xv;
            } else {
                a[r] = 0.0f;
            }
            a[r] = macaq_wht_simd<INNER>(a[r], lane);
        }
        if (R == 2) {
            const float u0 = a[0], u1 = a[1 % R];
            a[0] = u0 + u1;
            a[1 % R] = u0 - u1;
        } else if (R == 4) {
            const float t0 = a[0] + a[1 % R], t1 = a[0] - a[1 % R];
            const float t2 = a[2 % R] + a[3 % R], t3 = a[2 % R] - a[3 % R];
            a[0] = t0 + t2;
            a[1 % R] = t1 + t3;
            a[2 % R] = t0 - t2;
            a[3 % R] = t1 - t3;
        }
        for (int r = 0; r < R; ++r) {
            const int col = r * 32 + lane;
            if (col < N) {
                float res = a[r] * scale;
                if (IS_INV) { res *= float(signs[sign_base + size_t(col)]); }
                out[base + size_t(col)] = static_cast<T>(res);
            }
        }
        """

    static let switchBlockWHTSource = """
        constexpr int R = (N >= 32) ? (N / 32) : 1;
        constexpr int INNER = (N < 32) ? N : 32;
        const int blk = int(thread_position_in_grid.y);
        const int task = int(thread_position_in_grid.z);
        const int lane = int(thread_index_in_simdgroup);
        const int e = int(rhs_indices[task]);
        const size_t row_off = size_t(task) * size_t(K) + size_t(blk) * size_t(N);
        if (e < 0 || e >= E) {
            for (int r = 0; r < R; ++r) {
                const int col = r * 32 + lane;
                if (col < N) { out[row_off + size_t(col)] = static_cast<T>(0.0f); }
            }
            return;
        }
        const size_t p_off = (PER_EXPERT_PRE ? (size_t(e) * size_t(K)) : size_t(0)) + size_t(blk) * size_t(N);
        const float scale = rsqrt(float(N));
        float a[R];
        for (int r = 0; r < R; ++r) {
            const int col = r * 32 + lane;
            if (col < N) {
                a[r] = float(inp[row_off + size_t(col)]) * float(signs[p_off + size_t(col)]);
            } else {
                a[r] = 0.0f;
            }
            a[r] = macaq_wht_simd<INNER>(a[r], lane);
        }
        if (R == 2) {
            const float u0 = a[0], u1 = a[1 % R];
            a[0] = u0 + u1;
            a[1 % R] = u0 - u1;
        } else if (R == 4) {
            const float t0 = a[0] + a[1 % R], t1 = a[0] - a[1 % R];
            const float t2 = a[2 % R] + a[3 % R], t3 = a[2 % R] - a[3 % R];
            a[0] = t0 + t2;
            a[1 % R] = t1 + t3;
            a[2 % R] = t0 - t2;
            a[3 % R] = t1 - t3;
        }
        for (int r = 0; r < R; ++r) {
            const int col = r * 32 + lane;
            if (col < N) {
                out[row_off + size_t(col)] = static_cast<T>(a[r] * scale);
            }
        }
        """
}

private let singleSimdBlockWHTKernel = MLXFast.metalKernel(
    name: "macaq_single_simd_wht",
    inputNames: ["inp", "signs"],
    outputNames: ["out"],
    source: MACaQFusedMetal.blockWHTSource,
    header: MACaQFusedMetal.header
)

private let switchSimdBlockWHTKernel = MLXFast.metalKernel(
    name: "macaq_switch_simd_wht",
    inputNames: ["inp", "rhs_indices", "signs"],
    outputNames: ["out"],
    source: MACaQFusedMetal.switchBlockWHTSource,
    header: MACaQFusedMetal.header
)
