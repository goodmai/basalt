import Foundation
import MLX

package enum BailingHybridKDAKernel {
    package static let supportedDTypes: Set<DType> = [.float32, .float16, .bfloat16]
    private static let unmaskedKernel = makeKernel(hasMask: false)
    private static let maskedKernel = makeKernel(hasMask: true)

    package static func canUseKernel(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              hasSupportedDTypes(inputs, state: state),
              inputs.q.ndim == 4, inputs.k.ndim == 4, inputs.v.ndim == 4,
              inputs.decay.ndim == 4, inputs.beta.ndim == 3
        else {
            return false
        }
        return hasValidShapes(inputs, state: state, mask: mask)
    }

    private static func hasSupportedDTypes(_ inputs: BailingHybridKDAInputs, state: MLXArray?) -> Bool {
        let allSupported = [inputs.q.dtype, inputs.k.dtype, inputs.v.dtype, inputs.decay.dtype, inputs.beta.dtype]
            .allSatisfy(supportedDTypes.contains)
        let stateOK = state.map { supportedDTypes.contains($0.dtype) } ?? true
        let qkvMatch = inputs.q.dtype == .float32 || (inputs.k.dtype == inputs.q.dtype && inputs.v.dtype == inputs.q.dtype)
        return allSupported && stateOK && qkvMatch
    }

    private static func hasValidShapes(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray?,
        mask: MLXArray?
    ) -> Bool {
        let (batch, steps, hk, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        guard batch > 0, steps >= 0, steps <= Int(Int32.max), hk > 0, hv > 0, dk > 0, dk <= 512, dv > 0,
              batch * hv <= Int(Int32.max), dk.isMultiple(of: 32), hv.isMultiple(of: hk),
              inputs.k.shape == [batch, steps, hk, dk],
              inputs.v.dim(0) == batch, inputs.v.dim(1) == steps,
              inputs.decay.shape == [batch, steps, hv, dk],
              inputs.beta.shape == [batch, steps, hv]
        else {
            return false
        }
        if let state, state.shape != [batch, hv, dv, dk] {
            return false
        }
        if let mask, mask.shape != [batch, steps] || mask.dtype == .float64 {
            return false
        }
        return true
    }

    package static func run(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray?,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        guard canUseKernel(inputs, state: state, mask: mask) else {
            let arrays: [MLXArray?] = [inputs.q, inputs.k, inputs.v, inputs.decay, inputs.beta, state, mask]
            if Device.defaultDevice() == .gpu && arrays.contains(where: { $0?.dtype == .float64 }) {
                return Device.withDefaultDevice(.cpu) {
                    BailingHybridKDAOps.recurrentUpdateOps(inputs, state: state, mask: mask)
                }
            }
            return BailingHybridKDAOps.recurrentUpdateOps(inputs, state: state, mask: mask)
        }
        let batch = inputs.q.dim(0)
        let steps = inputs.q.dim(1)
        let dk = inputs.q.dim(3)
        let hv = inputs.v.dim(2)
        let dv = inputs.v.dim(3)
        let inDType = inputs.q.dtype
        let stateIn = contiguous(
            (state ?? MLXArray.zeros([batch, hv, dv, dk], dtype: .float32)).asType(.float32)
        )
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, hv, dv], dtype: inDType), stateIn)
        }
        var args = [inputs.q, inputs.k, inputs.v, inputs.decay, inputs.beta, stateIn]
        if let mask { args.append(mask) }
        let outputs = metalDispatch(arrays: args, inDType: inDType, dk: dk)
        return (outputs[0], outputs[1])
    }

    private static func metalDispatch(
        arrays: [MLXArray], inDType: DType, dk: Int
    ) -> [MLXArray] {
        let (batch, steps) = (arrays[0].dim(0), arrays[0].dim(1))
        let (hv, dv) = (arrays[2].dim(2), arrays[2].dim(3))
        let q = contiguous(arrays[0])
        let k = contiguous(arrays[1].asType(inDType))
        let v = contiguous(arrays[2].asType(inDType))
        let decay = contiguous(arrays[3].asType(.float32))
        let beta = contiguous(arrays[4].asType(.float32))
        let stateIn = contiguous(arrays[5].asType(.float32))

        let selected: MLXFast.MLXFastKernel
        var kernelInputs: [MLXArray] = [q, k, v, decay, beta, stateIn]
        if arrays.count > 6 {
            selected = maskedKernel
            kernelInputs.append(contiguous(arrays[6].asType(.bool)))
        } else {
            selected = unmaskedKernel
        }

        return selected(
            kernelInputs,
            template: [
                ("InT", inDType),
                ("Dk", dk),
            ],
            grid: (32, dv, batch * hv),
            threadGroup: (32, min(4, dv), 1),
            outputShapes: [[batch, steps, hv, dv], [batch, hv, dv, dk]],
            outputDTypes: [inDType, .float32]
        )
    }

    private static func makeKernel(hasMask: Bool) -> MLXFast.MLXFastKernel {
        let maskExpr = hasMask ? "mask[b_idx * steps + t_step]" : "true"
        let source = BailingHybridKDAMetal.setupSource + BailingHybridKDAMetal.loopSource(maskExpr: maskExpr)
        var inputNames = ["q", "k", "v", "g", "beta", "state_in"]
        if hasMask {
            inputNames.append("mask")
        }
        return MLXFast.metalKernel(
            name: hasMask ? "bailing_kda_step_mask" : "bailing_kda_step",
            inputNames: inputNames,
            outputNames: ["y", "state_out"],
            source: source
        )
    }
}

private enum BailingHybridKDAMetal {
    static let setupSource = """
        const size_t steps = size_t(q_shape[1]);
        const size_t Hk = size_t(q_shape[2]);
        const size_t Hv = size_t(v_shape[2]);
        const size_t Dv = size_t(v_shape[3]);
        const size_t n = size_t(thread_position_in_grid.z);
        const size_t b_idx = n / Hv;
        const size_t hv_idx = n % Hv;
        const size_t hk_idx = hv_idx / (Hv / Hk);
        const size_t dk_idx = size_t(thread_position_in_threadgroup.x);
        const size_t dv_idx = size_t(thread_position_in_grid.y);
        if (dv_idx >= Dv) { return; }

        constexpr int n_per_t = Dk / 32;

        auto q_ptr = q + (b_idx * steps * Hk + hk_idx) * size_t(Dk);
        auto k_ptr = k + (b_idx * steps * Hk + hk_idx) * size_t(Dk);
        auto v_ptr = v + (b_idx * steps * Hv + hv_idx) * Dv;
        auto y_ptr = y + (b_idx * steps * Hv + hv_idx) * Dv;
        auto g_ptr = g + (b_idx * steps * Hv + hv_idx) * size_t(Dk);
        auto b_ptr = beta + b_idx * steps * Hv + hv_idx;
        auto i_state = state_in + (n * Dv + dv_idx) * size_t(Dk);
        auto o_state = state_out + (n * Dv + dv_idx) * size_t(Dk);

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        """

    static func loopSource(maskExpr: String) -> String {
        """
        for (size_t t_step = 0; t_step < steps; ++t_step) {
          if (\(maskExpr)) {
            float kv_mem = 0.0f;
            {
              #pragma clang fp reassociate(off)
              #pragma clang fp contract(off)
              float kv_comp = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                int s_idx = n_per_t * int(dk_idx) + i;
                state[i] = state[i] * static_cast<float>(g_ptr[s_idx]);
                float product = state[i] * static_cast<float>(k_ptr[s_idx]);
                float corrected = product - kv_comp;
                float next_sum = kv_mem + corrected;
                kv_comp = isfinite(next_sum) ? ((next_sum - kv_mem) - corrected) : 0.0f;
                kv_mem = next_sum;
              }
            }
            kv_mem = simd_sum(kv_mem);

            float delta = (static_cast<float>(v_ptr[dv_idx]) - kv_mem) * static_cast<float>(b_ptr[0]);

            float out_val = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              int s_idx = n_per_t * int(dk_idx) + i;
              state[i] = state[i] + static_cast<float>(k_ptr[s_idx]) * delta;
              out_val += state[i] * static_cast<float>(q_ptr[s_idx]);
            }
            out_val = simd_sum(out_val);
            if (thread_index_in_simdgroup == 0) {
              y_ptr[dv_idx] = static_cast<InT>(out_val);
            }
          } else if (thread_index_in_simdgroup == 0) {
            y_ptr[dv_idx] = static_cast<InT>(0);
          }
          q_ptr += Hk * size_t(Dk);
          k_ptr += Hk * size_t(Dk);
          v_ptr += Hv * Dv;
          y_ptr += Hv * Dv;
          g_ptr += Hv * size_t(Dk);
          b_ptr += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          o_state[s_idx] = state[i];
        }
        """
    }
}
