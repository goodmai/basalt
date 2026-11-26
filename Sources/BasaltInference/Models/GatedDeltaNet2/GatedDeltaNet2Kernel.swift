import Foundation
import MLX

package struct GatedDeltaNet2Inputs {
    package let q: MLXArray
    package let k: MLXArray
    package let v: MLXArray
    package let gLog: MLXArray
    package let erase: MLXArray
    package let write: MLXArray
    package var betaGuard: Bool

    package init(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        gLog: MLXArray,
        erase: MLXArray,
        write: MLXArray
    ) {
        self.q = q
        self.k = k
        self.v = v
        self.gLog = gLog
        self.erase = erase
        self.write = write
        self.betaGuard = false
    }

    package func withBetaGuard(_ enabled: Bool) -> GatedDeltaNet2Inputs {
        var copy = self
        copy.betaGuard = enabled
        return copy
    }

    package func slice(at step: Int) -> GatedDeltaNet2Inputs {
        GatedDeltaNet2Inputs(
            q: q[0..., step],
            k: k[0..., step],
            v: v[0..., step],
            gLog: gLog[0..., step],
            erase: erase[0..., step],
            write: write[0..., step]
        ).withBetaGuard(betaGuard)
    }
}

package enum GatedDeltaNet2Kernel {
    private static let supportedDTypes: Set<DType> = [.float16, .bfloat16, .float32]
    private static let unmaskedKernel = makeKernel(hasMask: false)
    private static let maskedKernel = makeKernel(hasMask: true)

    package static func canUseKernel(
        _ inputs: GatedDeltaNet2Inputs,
        state: MLXArray? = nil,
        mask: MLXArray? = nil
    ) -> Bool {
        guard Device.defaultDevice() == .gpu,
              supportedDTypes.contains(inputs.q.dtype),
              supportedDTypes.contains(inputs.k.dtype),
              supportedDTypes.contains(inputs.v.dtype),
              supportedDTypes.contains(inputs.gLog.dtype),
              supportedDTypes.contains(inputs.erase.dtype),
              supportedDTypes.contains(inputs.write.dtype),
              state.map({ supportedDTypes.contains($0.dtype) }) ?? true,
              inputs.q.dtype == .float32 || (inputs.k.dtype == inputs.q.dtype && inputs.v.dtype == inputs.q.dtype),
              inputs.q.ndim == 4, inputs.k.ndim == 4, inputs.v.ndim == 4,
              inputs.gLog.ndim == 4, inputs.erase.ndim == 4, inputs.write.ndim == 4
        else { return false }

        let (batch, steps, hk, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(2), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        guard batch > 0, steps >= 0, steps <= Int(Int32.max), hk > 0, hv > 0, dk > 0, dk <= 512, dv > 0,
              batch * hv <= Int(Int32.max), dk.isMultiple(of: 32), hv.isMultiple(of: hk),
              inputs.k.shape == [batch, steps, hk, dk],
              inputs.v.dim(0) == batch, inputs.v.dim(1) == steps,
              inputs.gLog.shape == [batch, steps, hv, dk],
              inputs.erase.shape == [batch, steps, hv, dk],
              inputs.write.shape == [batch, steps, hv, dv]
        else { return false }

        if let state, state.shape != [batch, hv, dv, dk] { return false }
        if let mask, mask.shape != [batch, steps] || mask.dtype == .float64 { return false }
        return true
    }

    package static func run(
        _ inputs: GatedDeltaNet2Inputs,
        state: MLXArray?,
        mask: MLXArray? = nil
    ) -> (output: MLXArray, state: MLXArray) {
        guard canUseKernel(inputs, state: state, mask: mask) else {
            let arrays: [MLXArray?] = [inputs.q, inputs.k, inputs.v, inputs.gLog, inputs.erase, inputs.write, state, mask]
            if Device.defaultDevice() == .gpu && arrays.contains(where: { $0?.dtype == .float64 }) {
                return Device.withDefaultDevice(.cpu) { GatedDeltaNet2Ops.recurrentOps(inputs, state: state, mask: mask) }
            }
            return GatedDeltaNet2Ops.recurrentOps(inputs, state: state, mask: mask)
        }
        let (batch, steps, dk) = (inputs.q.dim(0), inputs.q.dim(1), inputs.q.dim(3))
        let (hv, dv) = (inputs.v.dim(2), inputs.v.dim(3))
        let inDType = inputs.q.dtype
        let stateIn = contiguous(
            (state ?? MLXArray.zeros([batch, hv, dv, dk], dtype: .float32)).asType(.float32)
        )
        guard steps > 0 else {
            return (MLXArray.zeros([batch, 0, hv, dv], dtype: inDType), stateIn)
        }
        let betaGuard = inputs.betaGuard
        var args = [inputs.q, inputs.k, inputs.v, inputs.gLog, inputs.erase, inputs.write, stateIn]
        if let mask { args.append(mask) }
        let fn = CustomFunction {
            Forward { arrays in
                metalDispatch(
                    arrays: arrays,
                    inDType: inDType,
                    dk: dk,
                    betaGuard: betaGuard
                )
            }
            VJP { primals, cotangents in
                vjp({ p in
                    let inp = GatedDeltaNet2Inputs(
                        q: p[0], k: p[1], v: p[2], gLog: p[3], erase: p[4], write: p[5]
                    ).withBetaGuard(betaGuard)
                    let stepMask = p.count > 7 ? stopGradient(p[7]) : nil
                    let (out, next) = GatedDeltaNet2Ops.recurrentOps(inp, state: p[6], mask: stepMask)
                    return [out, next]
                }, primals: primals, cotangents: cotangents).1
            }
        }
        let outputs = fn(args)
        return (outputs[0], outputs[1])
    }

    private static func metalDispatch(
        arrays: [MLXArray],
        inDType: DType,
        dk: Int,
        betaGuard: Bool
    ) -> [MLXArray] {
        let (batch, steps) = (arrays[0].dim(0), arrays[0].dim(1))
        let (hv, dv) = (arrays[2].dim(2), arrays[2].dim(3))
        let q = contiguous(arrays[0])
        let k = contiguous(arrays[1].asType(inDType))
        let v = contiguous(arrays[2].asType(inDType))
        let decay = contiguous(MLX.exp(arrays[3].asType(.float32)))
        let erase = contiguous(arrays[4].asType(.float32))
        let write = contiguous(arrays[5].asType(.float32))
        let stateIn = contiguous(arrays[6].asType(.float32))

        let selected: MLXFast.MLXFastKernel
        var kernelInputs: [MLXArray] = [q, k, v, decay, erase, write, stateIn, MLXArray(Int32(steps))]
        if arrays.count > 7 {
            selected = maskedKernel
            kernelInputs.append(contiguous(arrays[7].asType(.bool)))
        } else {
            selected = unmaskedKernel
        }

        return selected(
            kernelInputs,
            template: [
                ("InT", inDType),
                ("Dk", dk),
                ("BetaGuard", betaGuard ? 1 : 0),
            ],
            grid: (32, dv, batch * hv),
            threadGroup: (32, min(4, dv), 1),
            outputShapes: [[batch, steps, hv, dv], [batch, hv, dv, dk]],
            outputDTypes: [inDType, .float32]
        )
    }

    private static func makeKernel(hasMask: Bool) -> MLXFast.MLXFastKernel {
        let maskExpr = hasMask ? "mask[b_idx * steps + t_step]" : "true"
        let source = GatedDeltaNet2MSL.kernelSetupSource + GatedDeltaNet2MSL.kernelLoopSource(maskExpr: maskExpr)
        var inputNames = ["q", "k", "v", "g", "erase", "write", "state_in", "T"]
        if hasMask { inputNames.append("mask") }
        return MLXFast.metalKernel(
            name: hasMask ? "gdn2_recurrent_step_mask" : "gdn2_recurrent_step",
            inputNames: inputNames,
            outputNames: ["y", "state_out"],
            source: source
        )
    }
}

private enum GatedDeltaNet2MSL {
    static let kernelSetupSource = """
        size_t Hk = size_t(q_shape[2]);
        size_t Hv = size_t(v_shape[2]);
        size_t Dv = size_t(v_shape[3]);
        size_t n = size_t(thread_position_in_grid.z);
        size_t b_idx = n / Hv;
        size_t hv_idx = n % Hv;
        size_t hk_idx = hv_idx / (Hv / Hk);
        size_t dk_idx = size_t(thread_position_in_threadgroup.x);
        size_t dv_idx = size_t(thread_position_in_grid.y);
        if (dv_idx >= Dv) { return; }

        constexpr int n_per_t = Dk / 32;
        size_t steps = size_t(T);

        auto q_ptr = q + (b_idx * steps * Hk + hk_idx) * size_t(Dk);
        auto k_ptr = k + (b_idx * steps * Hk + hk_idx) * size_t(Dk);
        auto v_ptr = v + (b_idx * steps * Hv + hv_idx) * Dv;
        auto y_ptr = y + (b_idx * steps * Hv + hv_idx) * Dv;
        auto g_ptr = g + (b_idx * steps * Hv + hv_idx) * size_t(Dk);
        auto e_ptr = erase + (b_idx * steps * Hv + hv_idx) * size_t(Dk);
        auto w_ptr = write + (b_idx * steps * Hv + hv_idx) * Dv;
        auto i_state = state_in + (n * Dv + dv_idx) * size_t(Dk);
        auto o_state = state_out + (n * Dv + dv_idx) * size_t(Dk);

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        """

    private static let betaGuardSource = """
            float k_reg[n_per_t];
            float b_reg[n_per_t];
            float d_reg[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              int s_idx = n_per_t * int(dk_idx) + i;
              k_reg[i] = static_cast<float>(k_ptr[s_idx]);
              b_reg[i] = static_cast<float>(e_ptr[s_idx]);
              d_reg[i] = static_cast<float>(g_ptr[s_idx]);
            }
            if (BetaGuard != 0) {
              float local_a = 0.0f;
              float local_s = 0.0f;
              float local_dmax = 0.0f;
              bool local_d_nan = false;
              for (int i = 0; i < n_per_t; ++i) {
                float k2 = k_reg[i] * k_reg[i];
                local_a += k2;
                local_s += b_reg[i] * k2;
                local_d_nan = local_d_nan || isnan(d_reg[i]);
                local_dmax = max(local_dmax, d_reg[i]);
              }
              float a_sum = simd_sum(local_a);
              float s_sum = simd_sum(local_s);
              float d_max = simd_any(local_d_nan) ? NAN : min(simd_max(local_dmax), 1.0f);
              if (a_sum > 1e-12f) {
                float b_mean = s_sum / a_sum;
                float local_var = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  float diff = b_reg[i] - b_mean;
                  local_var += diff * diff * k_reg[i] * k_reg[i];
                }
                float delta2 = a_sum * simd_sum(local_var);
                float d2 = d_max * d_max;
                float lhs = d2 * delta2;
                float one_minus_s = 1.0f - s_sum;
                float factor = max(0.0f, 1.0f - d2 * one_minus_s * one_minus_s);
                float rhs = max(0.0f, (1.0f - d2) * factor);
                if (lhs > rhs && lhs > 0.0f) {
                  float rho = sqrt(rhs / lhs);
                  for (int i = 0; i < n_per_t; ++i) {
                    b_reg[i] = b_mean + rho * (b_reg[i] - b_mean);
                  }
                }
              }
            }
        """

    private static let recurrenceUpdateSource = """
            float kv_mem = 0.0f;
            {
              #pragma clang fp reassociate(off)
              #pragma clang fp contract(off)
              float kv_comp = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                state[i] = state[i] * d_reg[i];
                float product = state[i] * (b_reg[i] * k_reg[i]);
                float corrected = product - kv_comp;
                float next_sum = kv_mem + corrected;
                kv_comp = isfinite(next_sum) ? ((next_sum - kv_mem) - corrected) : 0.0f;
                kv_mem = next_sum;
              }
            }
            kv_mem = simd_sum(kv_mem);

            float delta = static_cast<float>(w_ptr[dv_idx]) * static_cast<float>(v_ptr[dv_idx]) - kv_mem;
            float out_val = 0.0f;
            {
              #pragma clang fp reassociate(off)
              #pragma clang fp contract(off)
              float out_comp = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                int s_idx = n_per_t * int(dk_idx) + i;
                state[i] = state[i] + k_reg[i] * delta;
                float product = state[i] * static_cast<float>(q_ptr[s_idx]);
                float corrected = product - out_comp;
                float next_sum = out_val + corrected;
                out_comp = isfinite(next_sum) ? ((next_sum - out_val) - corrected) : 0.0f;
                out_val = next_sum;
              }
            }
            out_val = simd_sum(out_val);
            if (thread_index_in_simdgroup == 0) {
              y_ptr[dv_idx] = static_cast<InT>(out_val);
            }
        """

    private static let kernelTailSource = """
          } else if (thread_index_in_simdgroup == 0) {
            y_ptr[dv_idx] = static_cast<InT>(0);
          }
          q_ptr += Hk * size_t(Dk);
          k_ptr += Hk * size_t(Dk);
          v_ptr += Hv * Dv;
          y_ptr += Hv * Dv;
          g_ptr += Hv * size_t(Dk);
          e_ptr += Hv * size_t(Dk);
          w_ptr += Hv * Dv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          int s_idx = n_per_t * int(dk_idx) + i;
          o_state[s_idx] = state[i];
        }
        """

    static func kernelLoopSource(maskExpr: String) -> String {
        "for (size_t t_step = 0; t_step < steps; ++t_step) {\n  if (\(maskExpr)) {\n"
            + betaGuardSource + "\n"
            + recurrenceUpdateSource + "\n"
            + kernelTailSource
    }
}
