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
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || scaleVector.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) {
                differentiableBlockHadamard(x, scaleVector: scaleVector, blockSize: blockSize)
            }
        }
        let origType = x.dtype
        let clamped = clip(scaleVector.asType(.float32), min: Float(1e-4), max: Float(1e4))
        let scaled = x.asType(.float32) * clamped
        guard blockSize > 1, scaled.ndim > 0, scaled.size > 0, scaled.dim(-1).isMultiple(of: blockSize) else {
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
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || indices.dtype == .float64 || preScale.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) {
                differentiableSwitchHadamard(x, indices: indices, preScale: preScale, blockSize: blockSize)
            }
        }
        let origType = x.dtype
        let clamped = clip(preScale.asType(.float32), min: Float(1e-4), max: Float(1e4))
        let scaled: MLXArray
        if clamped.ndim < 2 {
            scaled = x.asType(.float32) * clamped
        } else {
            let numExperts = clamped.dim(0)
            guard numExperts > 0, indices.size > 0 else {
                let xBatch = x.ndim >= 2 ? Array(x.shape.dropLast(2)) : []
                let batch = BasaltMetalSubstrate.broadcastShape(xBatch, indices.shape) ?? indices.shape
                let tail = x.ndim >= 2 ? [x.dim(-2), x.dim(-1)] : x.shape
                return MLXArray.zeros(batch + tail, dtype: origType)
            }
            let idx64 = indices.asType(.int64)
            let valid = (idx64 .>= Int64(0)) .&& (idx64 .< Int64(numExperts))
            let safeIdx = which(valid, idx64, Int64(0)).asType(.int32)
            let gathered = expandedDimensions(clamped.take(safeIdx, axis: 0), axis: -2)
            let rawScaled = x.asType(.float32) * gathered
            scaled = which(valid.reshaped(indices.shape + [1, 1]), rawScaled, MLXArray.zeros(rawScaled.shape, dtype: .float32))
        }
        guard blockSize > 1, scaled.ndim > 0, scaled.size > 0, scaled.dim(-1).isMultiple(of: blockSize) else {
            return scaled.asType(origType)
        }
        let scale = 1.0 / Float(blockSize).squareRoot()
        return hadamardTransform(scaled.reshaped(-1, blockSize), scale: scale)
            .reshaped(scaled.shape)
            .asType(origType)
    }

    private static func cleanScale(_ array: MLXArray) -> MLXArray {
        dtypes.contains(array.dtype) ? array : array.asType(.float32, stream: array.dtype == .float64 ? .cpu : .default)
    }

    package static func blockHadamard(
        _ x: MLXArray,
        scaleVector: MLXArray,
        blockSize: Int,
        isInverse: Bool = false
    ) -> MLXArray {
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || scaleVector.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) {
                blockHadamard(x, scaleVector: scaleVector, blockSize: blockSize, isInverse: isInverse)
            }
        }
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
        guard Device.defaultDevice() == .gpu else {
            return pureBlockHadamard(x, scaleVector: scaleVector, blockSize: blockSize, isInverse: isInverse)
        }
        let isRowScale = scaleVector.ndim > 0 && scaleVector.ndim <= x.ndim
            && scaleVector.size == width && scaleVector.dim(-1) == width
        guard supportsSingleSimdHadamard(blockSize: blockSize, dtype: origType),
              isRowScale,
              totalBlocks <= Int(Int32.max / 2)
        else {
            if !isRowScale {
                return pureBlockHadamard(x, scaleVector: scaleVector, blockSize: blockSize, isInverse: isInverse)
            }
            return FusedKroneckerRHT.blockHadamardFallback(
                x, signs: scaleVector, blockSize: blockSize, isInverse: isInverse)
        }
        let clean = cleanScale(scaleVector)
        let simdPerTG = totalBlocks.isMultiple(of: 4) ? 4 : (totalBlocks.isMultiple(of: 2) ? 2 : 1)
        let out = singleSimdBlockWHTKernel(
            [x.reshaped(totalBlocks, blockSize), clean.reshaped(width)],
            template: [
                ("T", origType),
                ("TS", clean.dtype),
                ("N", blockSize),
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
        if Device.defaultDevice() == .gpu, (x.dtype == .float64 || indices.dtype == .float64 || preScale.dtype == .float64) {
            return Device.withDefaultDevice(.cpu) {
                switchBlockHadamard(x, indices: indices, preScale: preScale, blockSize: blockSize, numExperts: numExperts)
            }
        }
        let origType = x.dtype
        guard x.ndim >= 2 else { return x }
        let xBatch = Array(x.shape.dropLast(2))
        let batchShape = BasaltMetalSubstrate.broadcastShape(xBatch, indices.shape) ?? indices.shape
        let rows = x.dim(-2)
        let width = x.dim(-1)
        let targetShape = batchShape + [rows, width]
        let flatTasks = batchShape.reduce(1, *) * max(0, rows)
        guard flatTasks > 0, x.size > 0, indices.size > 0, width > 0, rows > 0, numExperts > 0,
              BasaltMetalSubstrate.broadcastShape(xBatch, indices.shape) != nil else {
            return MLXArray.zeros(targetShape, dtype: origType)
        }
        let xExp = x.shape == targetShape ? x : broadcast(x, to: targetShape)
        let idxB = indices.shape == batchShape ? indices : broadcast(indices, to: batchShape)
        guard canDispatchSwitchKernel(
            preScale: preScale, width: width, blockSize: blockSize, numExperts: numExperts, dtype: origType)
        else {
            return switchBlockHadamardFallback(
                xExp, indices: idxB, preScale: preScale, blockSize: blockSize, numExperts: numExperts)
        }
        let idxExp = rows == 1
            ? idxB.asType(.int64)
            : broadcast(idxB.asType(.int64).expandedDimensions(axis: -1), to: batchShape + [rows])
        let numBlocks = width / blockSize
        let clean = cleanScale(preScale)
        let simdPerTG = numBlocks.isMultiple(of: 4) ? 4 : (numBlocks.isMultiple(of: 2) ? 2 : 1)
        let out = switchSimdBlockWHTKernel(
            [xExp.reshaped(flatTasks, width), idxExp.reshaped(flatTasks), clean, MLXArray([Int32(numExperts)])],
            template: [
                ("T", origType),
                ("TS", clean.dtype),
                ("N", blockSize),
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

private extension MACaQFusedKernels {
    static func canDispatchSwitchKernel(
        preScale: MLXArray, width: Int, blockSize: Int, numExperts: Int, dtype: DType
    ) -> Bool {
        let validPreScale = (preScale.ndim == 1 && preScale.dim(0) == width)
            || (preScale.ndim == 2 && preScale.dim(0) >= numExperts && preScale.dim(1) == width)
        return Device.defaultDevice() == .gpu && blockSize > 0 && width.isMultiple(of: blockSize)
            && validPreScale && supportsSingleSimdHadamard(blockSize: blockSize, dtype: dtype)
    }

    static func pureBlockHadamard(
        _ x: MLXArray,
        scaleVector: MLXArray,
        blockSize: Int,
        isInverse: Bool
    ) -> MLXArray {
        let origType = x.dtype
        let scale = 1.0 / Float(blockSize).squareRoot()
        let sv = scaleVector.asType(.float32)
        if isInverse {
            let h = hadamardTransform(x.asType(.float32).reshaped(-1, blockSize), scale: scale).reshaped(x.shape)
            return (h * sv).asType(origType)
        }
        let xs = x.asType(.float32) * sv
        guard xs.size > 0 else { return xs.asType(origType) }
        return hadamardTransform(xs.reshaped(-1, blockSize), scale: scale).reshaped(xs.shape).asType(origType)
    }

    static func switchBlockHadamardFallback(
        _ xExp: MLXArray,
        indices: MLXArray,
        preScale: MLXArray,
        blockSize: Int,
        numExperts: Int
    ) -> MLXArray {
        let origType = xExp.dtype
        let targetShape = xExp.shape
        let effectiveExperts = preScale.ndim == 2 ? min(numExperts, preScale.dim(0)) : numExperts
        guard effectiveExperts > 0, xExp.size > 0 else {
            return MLXArray.zeros(targetShape, dtype: origType)
        }
        let idx64 = indices.asType(.int64)
        let valid = (idx64 .>= Int64(0)) .&& (idx64 .< Int64(effectiveExperts))
        let safeIdx = which(valid, idx64, Int64(0)).asType(.int32)
        let ps = preScale.asType(.float32)
        let scaled: MLXArray
        if ps.ndim == 2 {
            let gathered = expandedDimensions(ps.take(safeIdx, axis: 0), axis: -2)
            scaled = xExp.asType(.float32) * gathered
        } else {
            scaled = xExp.asType(.float32) * ps
        }
        let validMask = valid.reshaped(indices.shape + [1, 1])
        let masked = which(validMask, scaled, MLXArray.zeros(scaled.shape, dtype: .float32))
        guard blockSize > 1, masked.dim(-1).isMultiple(of: blockSize) else {
            return masked.asType(origType)
        }
        let scale = 1.0 / Float(blockSize).squareRoot()
        return hadamardTransform(masked.reshaped(-1, blockSize), scale: scale)
            .reshaped(targetShape)
            .asType(origType)
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
        const int num_blocks = signs_shape[0] / N;
        const size_t base = size_t(blk_idx) * size_t(N);
        const size_t sign_base = size_t(blk_idx % num_blocks) * size_t(N);
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
        const int K = inp_shape[1];
        const int E = int(meta[0]);
        const int64_t e = int64_t(rhs_indices[task]);
        const size_t row_off = size_t(task) * size_t(K) + size_t(blk) * size_t(N);
        if (e < 0 || e >= int64_t(E)) {
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
    inputNames: ["inp", "rhs_indices", "signs", "meta"],
    outputNames: ["out"],
    source: MACaQFusedMetal.switchBlockWHTSource,
    header: MACaQFusedMetal.header
)
