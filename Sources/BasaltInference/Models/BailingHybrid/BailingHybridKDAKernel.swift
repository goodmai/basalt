import Foundation
import MLX

package enum BailingHybridKDAKernel {
    private static let unmaskedKernel = makeKernel(hasMask: false)
    private static let maskedKernel = makeKernel(hasMask: true)

    package static func canUseKernel(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              inputs.q.dtype.isFloatingPoint,
              inputs.k.dtype.isFloatingPoint,
              inputs.v.dtype.isFloatingPoint,
              inputs.q.dtype == .float32 || (inputs.k.dtype == inputs.q.dtype && inputs.v.dtype == inputs.q.dtype),
              inputs.q.ndim == 4, inputs.k.ndim == 4, inputs.v.ndim == 4,
              inputs.decay.ndim == 4, inputs.beta.ndim == 3
        else {
            return false
        }
        let (batch, steps, hk, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        guard batch > 0, steps >= 0, steps <= Int(Int32.max), hk > 0, hv > 0, dk > 0, dk <= 512, dv > 0,
              dk.isMultiple(of: 32), hv.isMultiple(of: hk),
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
        if let mask, mask.shape != [batch, steps] {
            return false
        }
        return true
    }

    package static func run(
        _ inputs: BailingHybridKDAInputs,
        state: MLXArray?,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let batch = inputs.q.dim(0)
        let steps = inputs.q.dim(1)
        let hk = inputs.q.dim(2)
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

        let q = contiguous(inputs.q)
        let k = contiguous(inputs.k.asType(inDType))
        let v = contiguous(inputs.v.asType(inDType))
        let decay = contiguous(inputs.decay.asType(.float32))
        let beta = contiguous(inputs.beta.asType(.float32))

        let selected: MLXFast.MLXFastKernel
        var kernelInputs: [MLXArray] = [q, k, v, decay, beta, stateIn, MLXArray(Int32(steps))]
        if let mask {
            selected = maskedKernel
            kernelInputs.append(contiguous(mask.asType(.bool)))
        } else {
            selected = unmaskedKernel
        }

        let outputs = selected(
            kernelInputs,
            template: [
                ("InT", inDType),
                ("StT", DType.float32),
                ("Dk", dk),
                ("Dv", dv),
                ("Hk", hk),
                ("Hv", hv),
            ],
            grid: (32, dv, batch * hv),
            threadGroup: (32, min(4, dv), 1),
            outputShapes: [[batch, steps, hv, dv], [batch, hv, dv, dk]],
            outputDTypes: [inDType, .float32]
        )
        return (outputs[0], outputs[1])
    }

    private static func makeKernel(hasMask: Bool) -> MLXFast.MLXFastKernel {
        let maskExpr = hasMask ? "mask[b_idx * steps + t_step]" : "true"
        let source = kernelSetupSource + kernelLoopSource(maskExpr: maskExpr)
        var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
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

    private static let kernelSetupSource = """
        size_t n = size_t(thread_position_in_grid.z);
        size_t b_idx = n / size_t(Hv);
        size_t hv_idx = n % size_t(Hv);
        size_t hk_idx = hv_idx / size_t(Hv / Hk);
        size_t dk_idx = size_t(thread_position_in_threadgroup.x);
        size_t dv_idx = size_t(thread_position_in_grid.y);
        if (dv_idx >= size_t(Dv)) { return; }

        constexpr int n_per_t = Dk / 32;
        size_t steps = size_t(T);

        auto q_ptr = q + (b_idx * steps * size_t(Hk) + hk_idx) * size_t(Dk);
        auto k_ptr = k + (b_idx * steps * size_t(Hk) + hk_idx) * size_t(Dk);
        auto v_ptr = v + (b_idx * steps * size_t(Hv) + hv_idx) * size_t(Dv);
        auto y_ptr = y + (b_idx * steps * size_t(Hv) + hv_idx) * size_t(Dv);
        auto g_ptr = g + (b_idx * steps * size_t(Hv) + hv_idx) * size_t(Dk);
        auto b_ptr = beta + b_idx * steps * size_t(Hv) + hv_idx;
        auto i_state = state_in + (n * size_t(Dv) + dv_idx) * size_t(Dk);
        auto o_state = state_out + (n * size_t(Dv) + dv_idx) * size_t(Dk);

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        """

    private static func kernelLoopSource(maskExpr: String) -> String {
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
                kv_comp = (next_sum - kv_mem) - corrected;
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
          q_ptr += size_t(Hk) * size_t(Dk);
          k_ptr += size_t(Hk) * size_t(Dk);
          v_ptr += size_t(Hv) * size_t(Dv);
          y_ptr += size_t(Hv) * size_t(Dv);
          g_ptr += size_t(Hv) * size_t(Dk);
          b_ptr += size_t(Hv);
        }

        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
        """
    }
}
