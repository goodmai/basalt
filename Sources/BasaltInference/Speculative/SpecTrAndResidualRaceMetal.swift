import BasaltCore
import Foundation
import MLX

package enum SpeculativeVerifyMode: Int32, Sendable {
    case residualRace = 0
    case specTrOT = 1
}

package struct SpeculativeVerifyOutput: Sendable, Equatable {
    package var acceptedCount: Int
    package var emittedTokens: [Int32]

    package init(acceptedCount: Int, emittedTokens: [Int32]) {
        self.acceptedCount = acceptedCount
        self.emittedTokens = emittedTokens
    }
}

package enum SpecTrAndResidualRaceMetal {
    package static let maxVocabKernel = 4096

    package static func keyedUniform(seed: UInt32, step: UInt32, index: UInt32) -> Float {
        var h = seed &+ (step &* 0x9E37_79B9) &+ (index &* 0x85EB_CA6B)
        h ^= h >> 16
        h &*= 0x7FEB_352D
        h ^= h >> 15
        h &*= 0x846C_A68B
        h ^= h >> 16
        let u = (Float(h >> 8) + 0.5) * (1.0 / 16_777_216.0)
        return min(max(u, 1e-7), 1.0 - 1e-7)
    }

    package static func keyedExp(seed: UInt32, step: UInt32, index: UInt32) -> Float {
        -logf(keyedUniform(seed: seed, step: step, index: index))
    }

    package static func draftWithRaceKey(
        logits: [Float],
        temperature: Float,
        seed: UInt32,
        step: UInt32
    ) -> Int32 {
        if temperature <= 0 {
            return Int32(argMaxHost(logits))
        }
        let probs = softmaxHost(logits, temperature: temperature)
        var bestIdx = 0
        var bestTime = Float.infinity
        for v in 0 ..< probs.count where probs[v] > 0 {
            let e = keyedExp(seed: seed, step: step, index: UInt32(v) ^ 0x1357_9BDF)
            let t = e / probs[v]
            if t < bestTime {
                bestTime = t
                bestIdx = v
            }
        }
        return Int32(bestIdx)
    }

    package static func verifyGPU(
        targetLogits: MLXArray,
        draftLogits: MLXArray,
        draftTokens: MLXArray,
        temperature: Float,
        mode: SpeculativeVerifyMode,
        seed: UInt32
    ) -> SpeculativeVerifyOutput {
        let steps = draftTokens.dim(0)
        let numCandidates = draftTokens.dim(1)
        let vocab = targetLogits.dim(-1)
        precondition(vocab <= maxVocabKernel && steps >= 1 && numCandidates >= 1)
        let tF = targetLogits.asType(.float32).reshaped(steps + 1, vocab)
        let dF = draftLogits.asType(.float32).reshaped(steps, numCandidates, vocab)
        let dTok = draftTokens.asType(.int32).reshaped(steps, numCandidates)
        let tempArr = MLXArray([max(temperature, 0)], [1])
        let seedArr = MLXArray([seed], [1])
        let outs = verifyKernel(
            [tF, dF, dTok, tempArr, seedArr],
            template: [
                ("STEPS", steps),
                ("K_CAND", numCandidates),
                ("MODE", Int(mode.rawValue)),
            ],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[steps + 1], [1]],
            outputDTypes: [.int32, .int32]
        )
        eval(outs[0], outs[1])
        let accCount = Int(outs[1].item(Int32.self))
        let allTokens = outs[0].asArray(Int32.self)
        return SpeculativeVerifyOutput(
            acceptedCount: accCount,
            emittedTokens: Array(allTokens.prefix(accCount + 1))
        )
    }
}

extension SpecTrAndResidualRaceMetal {
    package static func verifyCPU(
        targetLogits: [[Float]],
        draftLogits: [[[Float]]],
        draftTokens: [[Int32]],
        temperature: Float,
        mode: SpeculativeVerifyMode,
        seed: UInt32
    ) -> SpeculativeVerifyOutput {
        if temperature <= 0 {
            return verifyGreedyCPU(targetLogits: targetLogits, draftTokens: draftTokens, mode: mode)
        }
        let steps = draftTokens.count
        var emitted: [Int32] = []
        for s in 0 ..< steps {
            var pRes = softmaxHost(targetLogits[s], temperature: temperature)
            let cands = mode == .residualRace ? 1 : draftTokens[s].count
            var stepAccepted: Int32? = nil
            for m in 0 ..< cands {
                let q = softmaxHost(draftLogits[s][m], temperature: temperature)
                let tok = Int(draftTokens[s][m])
                let uAcc = keyedUniform(seed: seed, step: UInt32(s * 31 + m), index: 0xACCE_55ED)
                if q[tok] <= 0 ? (pRes[tok] > 0) : (uAcc * q[tok] <= pRes[tok]) {
                    stepAccepted = Int32(tok)
                    break
                }
                var sumPos: Float = 0
                for v in 0 ..< pRes.count {
                    let diff = max(0, pRes[v] - q[v])
                    pRes[v] = diff
                    sumPos += diff
                }
                if sumPos <= 1e-12 {
                    stepAccepted = Int32(tok)
                    break
                }
                let inv = 1.0 / sumPos
                for v in 0 ..< pRes.count { pRes[v] *= inv }
            }
            if let accepted = stepAccepted {
                emitted.append(accepted)
            } else {
                emitted.append(sampleRaceFromProbs(pRes, seed: seed, step: UInt32(s), salt: 0x5EED_CAFE))
                return SpeculativeVerifyOutput(acceptedCount: s, emittedTokens: emitted)
            }
        }
        let pBonus = softmaxHost(targetLogits[steps], temperature: temperature)
        emitted.append(sampleRaceFromProbs(pBonus, seed: seed, step: UInt32(steps), salt: 0xB090_55ED))
        return SpeculativeVerifyOutput(acceptedCount: steps, emittedTokens: emitted)
    }

    private static func verifyGreedyCPU(
        targetLogits: [[Float]],
        draftTokens: [[Int32]],
        mode: SpeculativeVerifyMode
    ) -> SpeculativeVerifyOutput {
        let steps = draftTokens.count
        var emitted: [Int32] = []
        for s in 0 ..< steps {
            let win = Int32(argMaxHost(targetLogits[s]))
            let pool = mode == .residualRace ? Array(draftTokens[s].prefix(1)) : draftTokens[s]
            emitted.append(win)
            if !pool.contains(win) {
                return SpeculativeVerifyOutput(acceptedCount: s, emittedTokens: emitted)
            }
        }
        emitted.append(Int32(argMaxHost(targetLogits[steps])))
        return SpeculativeVerifyOutput(acceptedCount: steps, emittedTokens: emitted)
    }

    package static func softmaxHost(_ logits: [Float], temperature: Float) -> [Float] {
        let invT: Float = 1.0 / max(temperature, 1e-6)
        let m = logits.max() ?? 0
        let exps = logits.map { expf(($0 - m) * invT) }
        let s = max(exps.reduce(0, +), 1e-20)
        return exps.map { $0 / s }
    }

    private static func argMaxHost(_ vals: [Float]) -> Int {
        var best = 0
        for i in 1 ..< vals.count where vals[i] > vals[best] { best = i }
        return best
    }

    private static func sampleRaceFromProbs(_ probs: [Float], seed: UInt32, step: UInt32, salt: UInt32) -> Int32 {
        var best = 0
        var bestTime = Float.infinity
        for v in 0 ..< probs.count where probs[v] > 0 {
            let t = keyedExp(seed: seed, step: step, index: UInt32(v) ^ salt) / probs[v]
            if t < bestTime {
                bestTime = t
                best = v
            }
        }
        return Int32(best)
    }
}

private let verifyHeader = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float basalt_spec_u01(uint seed, uint step, uint idx) {
    uint h = seed + step * 0x9E3779B9u + idx * 0x85EBCA6Bu;
    h ^= h >> 16u;
    h *= 0x7FEB352Du;
    h ^= h >> 15u;
    h *= 0x846CA68Bu;
    h ^= h >> 16u;
    float u = (float(h >> 8u) + 0.5f) * (1.0f / 16777216.0f);
    return clamp(u, 1e-7f, 1.0f - 1e-7f);
}

inline float basalt_spec_exp(uint seed, uint step, uint idx) {
    return -metal::log(basalt_spec_u01(seed, step, idx));
}
"""

private let verifySource = """
    const int V = int(t_logits_shape[1]);
    const uint lane = uint(thread_index_in_simdgroup);
    const float temp = temp_in[0];
    const uint seed = seed_in[0];

    threadgroup float p_res[4096];
    threadgroup float q_buf[4096];

    if (temp <= 0.0f) {
        int acc = STEPS;
        for (int s = 0; s < STEPS; s++) {
            float loc_max = -INFINITY;
            int loc_idx = 0;
            for (int v = int(lane); v < V; v += 32) {
                float val = t_logits[s * V + v];
                if (val > loc_max) { loc_max = val; loc_idx = v; }
            }
            float g_max = simd_max(loc_max);
            int win_tok = simd_min(loc_max == g_max ? loc_idx : V);

            bool matched = false;
            int cands = (MODE == 0) ? 1 : K_CAND;
            for (int m = 0; m < cands; m++) {
                if (d_tokens[s * K_CAND + m] == win_tok) { matched = true; }
            }
            if (lane == 0u) { out_tokens[s] = win_tok; }
            if (!matched) {
                acc = s;
                break;
            }
        }
        if (acc == STEPS) {
            float loc_max = -INFINITY;
            int loc_idx = 0;
            for (int v = int(lane); v < V; v += 32) {
                float val = t_logits[STEPS * V + v];
                if (val > loc_max) { loc_max = val; loc_idx = v; }
            }
            float g_max = simd_max(loc_max);
            int win_tok = simd_min(loc_max == g_max ? loc_idx : V);
            if (lane == 0u) { out_tokens[STEPS] = win_tok; }
        }
        if (lane == 0u) { out_count[0] = acc; }
        return;
    }

    const float inv_t = 1.0f / max(temp, 1e-6f);
    int acc = STEPS;
    for (int s = 0; s < STEPS; s++) {
        float t_max = -INFINITY;
        for (int v = int(lane); v < V; v += 32) { t_max = max(t_max, t_logits[s * V + v]); }
        t_max = simd_max(t_max);
        float t_sum = 0.0f;
        for (int v = int(lane); v < V; v += 32) {
            float e = metal::exp((t_logits[s * V + v] - t_max) * inv_t);
            p_res[v] = e;
            t_sum += e;
        }
        t_sum = max(simd_sum(t_sum), 1e-20f);
        for (int v = int(lane); v < V; v += 32) { p_res[v] /= t_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int cands = (MODE == 0) ? 1 : K_CAND;
        int accepted_tok = -1;
        for (int m = 0; m < cands; m++) {
            const int d_base = (s * K_CAND + m) * V;
            float q_max = -INFINITY;
            for (int v = int(lane); v < V; v += 32) { q_max = max(q_max, d_logits[d_base + v]); }
            q_max = simd_max(q_max);
            float q_sum = 0.0f;
            for (int v = int(lane); v < V; v += 32) {
                float e = metal::exp((d_logits[d_base + v] - q_max) * inv_t);
                q_buf[v] = e;
                q_sum += e;
            }
            q_sum = max(simd_sum(q_sum), 1e-20f);
            for (int v = int(lane); v < V; v += 32) { q_buf[v] /= q_sum; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int tok = d_tokens[s * K_CAND + m];
            float u_acc = basalt_spec_u01(seed, uint(s * 31 + m), 0xACCE55EDu);
            float q_tok = q_buf[tok];
            float p_tok = p_res[tok];
            bool ok = (q_tok <= 0.0f) ? (p_tok > 0.0f) : (u_acc * q_tok <= p_tok);
            if (ok) {
                accepted_tok = tok;
                break;
            }
            float pos_sum = 0.0f;
            for (int v = int(lane); v < V; v += 32) {
                float diff = max(0.0f, p_res[v] - q_buf[v]);
                p_res[v] = diff;
                pos_sum += diff;
            }
            pos_sum = simd_sum(pos_sum);
            if (pos_sum <= 1e-12f) {
                accepted_tok = tok;
                break;
            }
            float inv_pos = 1.0f / pos_sum;
            for (int v = int(lane); v < V; v += 32) { p_res[v] *= inv_pos; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (accepted_tok >= 0) {
            if (lane == 0u) { out_tokens[s] = accepted_tok; }
        } else {
            float loc_time = INFINITY;
            int loc_v = 0;
            for (int v = int(lane); v < V; v += 32) {
                float pv = p_res[v];
                if (pv > 0.0f) {
                    float t_race = basalt_spec_exp(seed, uint(s), uint(v) ^ 0x5EEDCAFEu) / pv;
                    if (t_race < loc_time) { loc_time = t_race; loc_v = v; }
                }
            }
            float min_time = simd_min(loc_time);
            int fallback_tok = simd_min(loc_time == min_time ? loc_v : V);
            if (lane == 0u) { out_tokens[s] = fallback_tok; }
            acc = s;
            break;
        }
    }

    if (acc == STEPS) {
        float t_max = -INFINITY;
        for (int v = int(lane); v < V; v += 32) { t_max = max(t_max, t_logits[STEPS * V + v]); }
        t_max = simd_max(t_max);
        float t_sum = 0.0f;
        for (int v = int(lane); v < V; v += 32) {
            float e = metal::exp((t_logits[STEPS * V + v] - t_max) * inv_t);
            p_res[v] = e;
            t_sum += e;
        }
        t_sum = max(simd_sum(t_sum), 1e-20f);
        float loc_time = INFINITY;
        int loc_v = 0;
        for (int v = int(lane); v < V; v += 32) {
            float pv = p_res[v] / t_sum;
            if (pv > 0.0f) {
                float t_race = basalt_spec_exp(seed, uint(STEPS), uint(v) ^ 0xB09055EDu) / pv;
                if (t_race < loc_time) { loc_time = t_race; loc_v = v; }
            }
        }
        float min_time = simd_min(loc_time);
        int bonus_tok = simd_min(loc_time == min_time ? loc_v : V);
        if (lane == 0u) { out_tokens[STEPS] = bonus_tok; }
    }
    if (lane == 0u) { out_count[0] = acc; }
"""

private let verifyKernel = MLXFast.metalKernel(
    name: "speculative_verify_ot_residual_race",
    inputNames: ["t_logits", "d_logits", "d_tokens", "temp_in", "seed_in"],
    outputNames: ["out_tokens", "out_count"],
    source: verifySource,
    header: verifyHeader
)
