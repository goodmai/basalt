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
        if !(temperature > 0) {
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
        guard draftTokens.ndim >= 2, targetLogits.ndim >= 2, draftLogits.ndim >= 3 else {
            return SpeculativeVerifyOutput(acceptedCount: 0, emittedTokens: [])
        }
        let steps = draftTokens.dim(0)
        let numCandidates = draftTokens.dim(1)
        let vocab = targetLogits.dim(-1)
        guard vocab > 0, steps > 0, numCandidates > 0,
              draftTokens.size == steps * numCandidates,
              targetLogits.size == (steps + 1) * vocab,
              draftLogits.size == steps * numCandidates * vocab
        else {
            return SpeculativeVerifyOutput(acceptedCount: 0, emittedTokens: [])
        }
        guard Device.defaultDevice() == .gpu, vocab <= maxVocabKernel else {
            return verifyViaCPU(
                targetLogits: targetLogits, draftLogits: draftLogits, draftTokens: draftTokens,
                temperature: temperature, mode: mode, seed: seed)
        }
        let tF = targetLogits.asType(.float32, stream: targetLogits.dtype == .float64 ? .cpu : .default)
            .reshaped(steps + 1, vocab)
        let dF = draftLogits.asType(.float32, stream: draftLogits.dtype == .float64 ? .cpu : .default)
            .reshaped(steps, numCandidates, vocab)
        let dTok = sanitizedDraftTokens(draftTokens, steps: steps, numCandidates: numCandidates, vocab: vocab)
        let cleanTemp: Float = temperature > 0 ? temperature : 0
        let tempArr = MLXArray([cleanTemp], [1])
        let seedArr = MLXArray([seed], [1])
        let outs = verifyKernel(
            [tF, dF, dTok, tempArr, seedArr],
            template: [("MODE", Int(mode.rawValue))],
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

    private static func verifyViaCPU(
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
        let tFlat = targetLogits.asType(.float32, stream: targetLogits.dtype == .float64 ? .cpu : .default)
            .reshaped(steps + 1, vocab).asArray(Float.self)
        let dFlat = draftLogits.asType(.float32, stream: draftLogits.dtype == .float64 ? .cpu : .default)
            .reshaped(steps, numCandidates, vocab).asArray(Float.self)
        let tokFlat = sanitizedDraftTokens(draftTokens, steps: steps, numCandidates: numCandidates, vocab: vocab)
            .asArray(Int32.self)
        let tHost = (0 ... steps).map { s in Array(tFlat[(s * vocab) ..< ((s + 1) * vocab)]) }
        let dHost = (0 ..< steps).map { s in
            (0 ..< numCandidates).map { m in
                let base = (s * numCandidates + m) * vocab
                return Array(dFlat[base ..< (base + vocab)])
            }
        }
        let tokHost = (0 ..< steps).map { s in
            Array(tokFlat[(s * numCandidates) ..< ((s + 1) * numCandidates)])
        }
        return verifyCPU(
            targetLogits: tHost, draftLogits: dHost, draftTokens: tokHost,
            temperature: temperature, mode: mode, seed: seed)
    }

    private static func sanitizedDraftTokens(
        _ draftTokens: MLXArray, steps: Int, numCandidates: Int, vocab: Int
    ) -> MLXArray {
        guard draftTokens.dtype.isInteger else {
            return MLXArray.full([steps, numCandidates], values: MLXArray(Int32(-1)))
        }
        if draftTokens.dtype == .int32 {
            return draftTokens.reshaped(steps, numCandidates)
        }
        let tok64 = draftTokens.asType(.int64).reshaped(steps, numCandidates)
        let valid = (tok64 .>= Int64(0)) .&& (tok64 .< Int64(vocab))
        return MLX.where(valid, tok64, MLXArray(Int64(-1))).asType(.int32)
    }
}

private struct SpecVerifyStepContext {
    let temperature: Float
    let mode: SpeculativeVerifyMode
    let seed: UInt32
    let step: Int
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
        let steps = draftTokens.count
        guard steps > 0, let firstRow = targetLogits.first, !firstRow.isEmpty,
              targetLogits.count >= steps + 1, draftLogits.count >= steps
        else {
            return SpeculativeVerifyOutput(acceptedCount: 0, emittedTokens: [])
        }
        if !(temperature > 0) {
            return verifyGreedyCPU(targetLogits: targetLogits, draftTokens: draftTokens, mode: mode)
        }
        var emitted: [Int32] = []
        emitted.reserveCapacity(steps + 1)
        for s in 0 ..< steps {
            let outcome = verifyStepCPU(
                targetRow: targetLogits[s],
                draftRows: draftLogits[s],
                stepTokens: draftTokens[s],
                context: SpecVerifyStepContext(temperature: temperature, mode: mode, seed: seed, step: s))
            emitted.append(outcome.token)
            if !outcome.accepted {
                return SpeculativeVerifyOutput(acceptedCount: s, emittedTokens: emitted)
            }
        }
        let pBonus = softmaxHost(targetLogits[steps], temperature: temperature)
        emitted.append(sampleRaceFromProbs(pBonus, seed: seed, step: UInt32(steps), salt: 0xB090_55ED))
        return SpeculativeVerifyOutput(acceptedCount: steps, emittedTokens: emitted)
    }

    private static func verifyStepCPU(
        targetRow: [Float],
        draftRows: [[Float]],
        stepTokens: [Int32],
        context: SpecVerifyStepContext
    ) -> (accepted: Bool, token: Int32) {
        var pRes = softmaxHost(targetRow, temperature: context.temperature)
        let available = min(stepTokens.count, draftRows.count)
        let cands = context.mode == .residualRace ? min(1, available) : available
        for m in 0 ..< cands {
            let q = softmaxHost(draftRows[m], temperature: context.temperature)
            let tok = Int(stepTokens[m])
            guard tok >= 0, tok < pRes.count, tok < q.count else { continue }
            let uAcc = keyedUniform(seed: context.seed, step: UInt32(context.step * 31 + m), index: 0xACCE_55ED)
            if q[tok] <= 0 ? (pRes[tok] > 0) : (uAcc * q[tok] <= pRes[tok]) {
                return (true, Int32(tok))
            }
            let sumPos = subtractResidualHost(&pRes, draft: q)
            if sumPos <= 1e-12 {
                if pRes[tok] > 0 { return (true, Int32(tok)) }
                break
            }
            let inv = 1.0 / sumPos
            for v in 0 ..< pRes.count { pRes[v] *= inv }
        }
        let sampled = sampleRaceFromProbs(pRes, seed: context.seed, step: UInt32(context.step), salt: 0x5EED_CAFE)
        return (false, sampled)
    }

    private static func subtractResidualHost(_ pRes: inout [Float], draft q: [Float]) -> Float {
        var sumPos: Float = 0
        for v in 0 ..< pRes.count {
            let qv = v < q.count ? q[v] : 0
            let diff = max(0, pRes[v] - qv)
            pRes[v] = diff
            sumPos += diff
        }
        return sumPos
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
        if logits.contains(where: { $0 == .infinity }) {
            let weights = logits.map { $0 == .infinity ? Float(1) : Float(0) }
            let total = weights.reduce(0, +)
            return total > 0 ? weights.map { $0 / total } : weights
        }
        var maxFinite = -Float.infinity
        var hasFinite = false
        for x in logits where x.isFinite {
            if !hasFinite || x > maxFinite {
                maxFinite = x
                hasFinite = true
            }
        }
        guard hasFinite else {
            return [Float](repeating: 0, count: logits.count)
        }
        let exps = logits.map { $0.isFinite ? expf(($0 - maxFinite) * invT) : Float(0) }
        let total = exps.reduce(0, +)
        return total > 0 ? exps.map { $0 / total } : [Float](repeating: 0, count: logits.count)
    }

    private static func argMaxHost(_ vals: [Float]) -> Int {
        var best = -1
        var bestVal = -Float.infinity
        for i in 0 ..< vals.count {
            let v = vals[i]
            if !v.isNaN && (best < 0 || v > bestVal) {
                best = i
                bestVal = v
            }
        }
        return best >= 0 ? best : 0
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

template <typename Ptr>
inline int basalt_spec_argmax(Ptr logits, size_t base, int V, uint lane) {
    float loc_max = -INFINITY;
    int loc_idx = V;
    for (int v = int(lane); v < V; v += 32) {
        float val = logits[base + size_t(v)];
        if (!metal::isnan(val) && (loc_idx == V || val > loc_max)) {
            loc_max = val;
            loc_idx = v;
        }
    }
    float g_max = simd_max(loc_idx < V ? loc_max : -INFINITY);
    int raw_win = simd_min((loc_idx < V && loc_max == g_max) ? loc_idx : V);
    return (raw_win < V) ? raw_win : 0;
}

template <typename Ptr>
inline void basalt_spec_softmax(Ptr logits, size_t base, int V, float inv_t, uint lane, threadgroup float* dst) {
    int loc_pos_inf = 0;
    int loc_finite = 0;
    float loc_max = -INFINITY;
    for (int v = int(lane); v < V; v += 32) {
        float x = logits[base + size_t(v)];
        if (x == INFINITY) {
            loc_pos_inf = 1;
        } else if (metal::isfinite(x)) {
            loc_finite = 1;
            loc_max = max(loc_max, x);
        }
    }
    const bool has_pos_inf = (simd_max(loc_pos_inf) != 0);
    const bool has_finite = (simd_max(loc_finite) != 0);
    const float g_max = simd_max(loc_max);
    float loc_sum = 0.0f;
    for (int v = int(lane); v < V; v += 32) {
        float x = logits[base + size_t(v)];
        float e = 0.0f;
        if (has_pos_inf) {
            e = (x == INFINITY) ? 1.0f : 0.0f;
        } else if (has_finite && metal::isfinite(x)) {
            e = metal::exp((x - g_max) * inv_t);
        }
        dst[v] = e;
        loc_sum += e;
    }
    const float g_sum = simd_sum(loc_sum);
    if (g_sum > 0.0f) {
        for (int v = int(lane); v < V; v += 32) {
            dst[v] /= g_sum;
        }
    } else {
        for (int v = int(lane); v < V; v += 32) {
            dst[v] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline int basalt_spec_sample_race(const threadgroup float* probs, int V, uint seed, uint step, uint salt, uint lane) {
    float loc_time = INFINITY;
    int loc_v = V;
    for (int v = int(lane); v < V; v += 32) {
        float pv = probs[v];
        if (pv > 0.0f) {
            float t_race = basalt_spec_exp(seed, step, uint(v) ^ salt) / pv;
            if (t_race < loc_time) {
                loc_time = t_race;
                loc_v = v;
            }
        }
    }
    float min_time = simd_min(loc_time);
    int raw = simd_min((loc_v < V && loc_time == min_time) ? loc_v : V);
    return (raw < V) ? raw : 0;
}
"""

private let verifySource = """
    const int STEPS = int(d_tokens_shape[0]);
    const int K_CAND = int(d_tokens_shape[1]);
    const int V = int(t_logits_shape[1]);
    const uint lane = uint(thread_index_in_simdgroup);
    const float temp = temp_in[0];
    const uint seed = seed_in[0];

    threadgroup float p_res[4096];
    threadgroup float q_buf[4096];

    if (!(temp > 0.0f)) {
        int acc = STEPS;
        for (int s = 0; s < STEPS; s++) {
            int win_tok = basalt_spec_argmax(t_logits, size_t(s) * size_t(V), V, lane);
            bool matched = false;
            int cands = (MODE == 0) ? 1 : K_CAND;
            for (int m = 0; m < cands; m++) {
                if (d_tokens[size_t(s) * size_t(K_CAND) + size_t(m)] == win_tok) { matched = true; }
            }
            if (lane == 0u) { out_tokens[s] = win_tok; }
            if (!matched) {
                acc = s;
                break;
            }
        }
        if (acc == STEPS) {
            int win_tok = basalt_spec_argmax(t_logits, size_t(STEPS) * size_t(V), V, lane);
            if (lane == 0u) { out_tokens[STEPS] = win_tok; }
        }
        if (lane == 0u) { out_count[0] = acc; }
        return;
    }

    const float inv_t = 1.0f / max(temp, 1e-6f);
    int acc = STEPS;
    for (int s = 0; s < STEPS; s++) {
        basalt_spec_softmax(t_logits, size_t(s) * size_t(V), V, inv_t, lane, p_res);

        int cands = (MODE == 0) ? 1 : K_CAND;
        int accepted_tok = -1;
        for (int m = 0; m < cands; m++) {
            const size_t d_base = (size_t(s) * size_t(K_CAND) + size_t(m)) * size_t(V);
            basalt_spec_softmax(d_logits, d_base, V, inv_t, lane, q_buf);

            int tok = d_tokens[size_t(s) * size_t(K_CAND) + size_t(m)];
            bool valid_tok = (tok >= 0 && tok < V);
            if (!valid_tok) {
                continue;
            }
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
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pos_sum = simd_sum(pos_sum);
            if (pos_sum <= 1e-12f) {
                if (p_res[tok] > 0.0f) {
                    accepted_tok = tok;
                }
                break;
            }
            float inv_pos = 1.0f / pos_sum;
            for (int v = int(lane); v < V; v += 32) { p_res[v] *= inv_pos; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (accepted_tok >= 0) {
            if (lane == 0u) { out_tokens[s] = accepted_tok; }
        } else {
            int fallback_tok = basalt_spec_sample_race(p_res, V, seed, uint(s), 0x5EEDCAFEu, lane);
            if (lane == 0u) { out_tokens[s] = fallback_tok; }
            acc = s;
            break;
        }
    }

    if (acc == STEPS) {
        basalt_spec_softmax(t_logits, size_t(STEPS) * size_t(V), V, inv_t, lane, p_res);
        int bonus_tok = basalt_spec_sample_race(p_res, V, seed, uint(STEPS), 0xB09055EDu, lane);
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
