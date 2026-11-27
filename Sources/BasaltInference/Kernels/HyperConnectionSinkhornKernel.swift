import Foundation
import MLX

public enum HyperConnectionSinkhornKernel {
    public static let maxMetalExpansionRate = 8
    public static let supportedDTypes: Set<DType> = [.float32, .bfloat16, .float16]

    public static func isEligible(
        logits: MLXArray,
        iterations: Int,
        eps: Float = 1e-6
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              logits.ndim >= 2,
              logits.size > 0,
              iterations >= 1,
              iterations <= Int(Int32.max),
              eps.isFinite,
              eps > 0
        else {
            return false
        }
        let n = logits.dim(-1)
        guard logits.dim(-2) == n, n >= 1, n <= maxMetalExpansionRate else {
            return false
        }
        let count = logits.size / (n * n)
        return count <= Int(Int32.max) - 63 && supportedDTypes.contains(logits.dtype)
    }

    public static func project(
        logits: MLXArray,
        iterations: Int = 20,
        eps: Float = 1e-6,
        useMetal: Bool = true
    ) -> MLXArray {
        if Device.defaultDevice() == .gpu, logits.dtype == .float64 {
            return Device.withDefaultDevice(.cpu) {
                project(logits: logits, iterations: iterations, eps: eps, useMetal: useMetal)
            }
        }
        guard logits.ndim >= 2 else { return logits.asType(.float32) }
        let n = logits.dim(-1)
        guard logits.dim(-2) == n, n >= 1 else { return logits.asType(.float32) }
        guard logits.size > 0 else { return MLXArray.zeros(logits.shape, dtype: .float32) }
        guard useMetal, isEligible(logits: logits, iterations: iterations, eps: eps) else {
            return pureMLX(logits: logits, iterations: iterations, eps: eps)
        }
        let fn = CustomFunction {
            Forward { arrays in
                [metalDispatch(logits: arrays[0], n: n, iterations: iterations, eps: eps)]
            }
            VJP { primals, cotangents in
                vjp({ [pureMLX(logits: $0[0], iterations: iterations, eps: eps)] }, primals: primals, cotangents: cotangents).1
            }
        }
        return fn([logits])[0]
    }

    private static func metalDispatch(
        logits: MLXArray, n: Int, iterations: Int, eps: Float
    ) -> MLXArray {
        let count = logits.size / (n * n)
        let threadsPerGroup = min(count, 64)
        let gridThreads = ((count + threadsPerGroup - 1) / threadsPerGroup) * threadsPerGroup
        let configInts = MLXArray([Int32(count), Int32(iterations)])
        let epsArr = MLXArray([eps])
        let outputs = hyperConnectionSinkhornMetalKernel(
            [logits.reshaped(count, n * n), configInts, epsArr],
            template: [("N", n)],
            grid: (gridThreads, 1, 1),
            threadGroup: (threadsPerGroup, 1, 1),
            outputShapes: [[count, n * n]],
            outputDTypes: [.float32]
        )
        return outputs[0].reshaped(logits.shape)
    }

    public static func pureMLX(
        logits: MLXArray,
        iterations: Int = 20,
        eps: Float = 1e-6
    ) -> MLXArray {
        if Device.defaultDevice() == .gpu, logits.dtype == .float64 {
            return Device.withDefaultDevice(.cpu) {
                pureMLX(logits: logits, iterations: iterations, eps: eps)
            }
        }
        let x = logits.asType(.float32)
        guard x.ndim >= 2, x.size > 0 else { return x }
        let rowMax = x.max(axis: -1, keepDims: true)
        let rowHasNaN = isNaN(x).any(axis: -1, keepDims: true)
        let isPosInfMax = rowMax .== MLXArray(Float.infinity)
        let isNegInfMax = rowMax .== MLXArray(-Float.infinity)
        let finiteShift = MLX.where(isPosInfMax .|| isNegInfMax, MLXArray(Float(0)), rowMax)
        var p = exp(x - finiteShift)
        p = MLX.where(isPosInfMax, MLX.where(x .== MLXArray(Float.infinity), MLXArray(Float(1)), MLXArray(Float(0))), p)
        p = MLX.where(isNegInfMax, MLXArray(Float(0)), p)
        p = MLX.where(rowHasNaN, MLXArray(Float.nan), p)
        let epsArr = MLXArray(eps)
        for _ in 0 ..< max(iterations, 1) {
            p = p / (p.sum(axis: -1, keepDims: true) + epsArr)
            p = p / (p.sum(axis: -2, keepDims: true) + epsArr)
        }
        return p
    }
}

private enum HyperConnectionSinkhornMSL {
    static let header = "#include <metal_math>\n"

    static let source = """
        const int idx = int(thread_position_in_grid.x);
        const int total = int(config_ints[0]);
        const int iters = int(config_ints[1]);
        if (idx >= total) {
            return;
        }
        const float eps = float(eps_arr[0]);
        constexpr int NN = N * N;
        const size_t base = size_t(idx) * size_t(NN);
        float p[NN];
        #pragma unroll
        for (int r = 0; r < N; r++) {
            float row_max = -INFINITY;
            bool row_nan = false;
            #pragma unroll
            for (int c = 0; c < N; c++) {
                const float v = float(logits[base + size_t(r * N + c)]);
                p[r * N + c] = v;
                if (isnan(v)) {
                    row_nan = true;
                } else if (v > row_max) {
                    row_max = v;
                }
            }
            #pragma unroll
            for (int c = 0; c < N; c++) {
                const float v = p[r * N + c];
                if (row_nan || isnan(v)) {
                    p[r * N + c] = NAN;
                } else if (row_max == INFINITY) {
                    p[r * N + c] = (v == INFINITY) ? 1.0f : 0.0f;
                } else if (row_max == -INFINITY) {
                    p[r * N + c] = 0.0f;
                } else {
                    p[r * N + c] = metal::precise::exp(v - row_max);
                }
            }
        }
        for (int step = 0; step < iters; step++) {
            #pragma unroll
            for (int r = 0; r < N; r++) {
                float row_sum = 0.0f;
                #pragma unroll
                for (int c = 0; c < N; c++) {
                    row_sum += p[r * N + c];
                }
                const float row_denom = row_sum + eps;
                #pragma unroll
                for (int c = 0; c < N; c++) {
                    p[r * N + c] /= row_denom;
                }
            }
            #pragma unroll
            for (int c = 0; c < N; c++) {
                float col_sum = 0.0f;
                #pragma unroll
                for (int r = 0; r < N; r++) {
                    col_sum += p[r * N + c];
                }
                const float col_denom = col_sum + eps;
                #pragma unroll
                for (int r = 0; r < N; r++) {
                    p[r * N + c] /= col_denom;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < NN; k++) {
            out[base + size_t(k)] = p[k];
        }
        """
}

private let hyperConnectionSinkhornMetalKernel = MLXFast.metalKernel(
    name: "basalt_hyper_connection_sinkhorn",
    inputNames: ["logits", "config_ints", "eps_arr"],
    outputNames: ["out"],
    source: HyperConnectionSinkhornMSL.source,
    header: HyperConnectionSinkhornMSL.header
)
