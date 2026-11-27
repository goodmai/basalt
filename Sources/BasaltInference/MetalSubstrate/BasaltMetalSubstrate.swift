import Foundation
import MLX

package enum BasaltMetalSubstrate {
    package static let simdTransformHeader = """
        inline float basalt_wht_inner(float v, int lane, int width) {
            for (int h = 1; h < width; h <<= 1) {
                const float other = simd_shuffle_xor(v, ushort(h));
                v = other + (((lane & h) != 0) ? -v : v);
            }
            return v;
        }

        inline float basalt_wht32_simd(float v, int lane) {
            return basalt_wht_inner(v, lane, 32) * 0.1767766953f;
        }

        inline void basalt_wht64_simd(float v0, float v1, int lane, thread float& out0, thread float& out1) {
            const float a0 = basalt_wht_inner(v0, lane, 32);
            const float a1 = basalt_wht_inner(v1, lane, 32);
            out0 = (a0 + a1) * 0.125f;
            out1 = (a0 - a1) * 0.125f;
        }

        inline float basalt_e2m1_val(uint nibble) {
            constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            const float mag = lut[nibble & 7u];
            return ((nibble & 8u) != 0u) ? -mag : mag;
        }

        inline float basalt_e8m0_scale(uint8_t exp_byte) {
            if (exp_byte == 0u) { return as_type<float>(0x00400000u); }
            return as_type<float>(uint(exp_byte) << 23u);
        }

        inline uint8_t basalt_compute_e8m0(float max_abs, float grid_max) {
            if (!(max_abs > 0.0f)) { return 0u; }
            const float raw = max_abs / grid_max;
            const uint bits = as_type<uint>(raw);
            int biased_exp = int((bits >> 23u) & 255u);
            if ((bits & 0x7fffffu) != 0u && biased_exp < 254) {
                biased_exp += 1;
            }
            return uint8_t(clamp(biased_exp, 0, 254));
        }

        inline uint basalt_quant_e2m1(float v, float inv_scale) {
            const float sv = v * inv_scale;
            const uint sign = (sv < 0.0f) ? 8u : 0u;
            const float a = fabs(sv);
            uint code = 0u;
            if (a >= 5.0f) { code = 7u; }
            else if (a >= 3.5f) { code = 6u; }
            else if (a >= 2.5f) { code = 5u; }
            else if (a >= 1.75f) { code = 4u; }
            else if (a >= 1.25f) { code = 3u; }
            else if (a >= 0.75f) { code = 2u; }
            else if (a >= 0.25f) { code = 1u; }
            return sign | code;
        }

        inline float basalt_quant_dequant_fp8_e4m3(float v, float scale) {
            if (isnan(v) || isnan(scale)) { return NAN; }
            const float sv = clamp(v / scale, -448.0f, 448.0f);
            const float sgn = (sv < 0.0f) ? -1.0f : 1.0f;
            const float a = fabs(sv);
            if (a <= 0.0009765625f) { return 0.0f; }
            const int e = clamp(int((as_type<uint>(a) >> 23u) & 255u) - 127, -6, 8);
            const float step = ldexp(1.0f, e - 3);
            const float q = rint(a / step) * step;
            return sgn * min(q, 448.0f) * scale;
        }

        inline void basalt_givens_pair(float xi, float xj, float c, float s, thread float& yi, thread float& yj) {
            yi = c * xi + s * xj;
            yj = -s * xi + c * xj;
        }

        template <typename PtrW>
        inline uint basalt_extract_code(PtrW w_row, int col, int bits) {
            if (bits == 4) {
                return (w_row[col >> 3] >> uint((col & 7) * 4)) & 15u;
            }
            if (bits == 8) {
                return (w_row[col >> 2] >> uint((col & 3) * 8)) & 255u;
            }
            if (bits == 2) {
                return (w_row[col >> 4] >> uint((col & 15) * 2)) & 3u;
            }
            const int blk = col >> 5;
            const int bit_pos = (col & 31) * 3;
            const int wi = bit_pos >> 5;
            const int sh = bit_pos & 31;
            const uint32_t lo = w_row[blk * 3 + wi] >> uint(sh);
            if (sh <= 29) { return lo & 7u; }
            const uint32_t hi = w_row[blk * 3 + wi + 1] << uint(32 - sh);
            return (lo | hi) & 7u;
        }
        """

    package static let dualScaleEpilogueHeader = """
        template <typename T, bool HAS_ROW_SCALE, bool HAS_BIAS, typename PtrRS, typename PtrB>
        inline T basalt_dual_scale_epilogue(
            float acc,
            PtrRS row_scale,
            PtrB bias,
            int n
        ) {
            float val = acc;
            if (HAS_ROW_SCALE) {
                val *= float(row_scale[n]);
            }
            if (HAS_BIAS) {
                val += float(bias[n]);
            }
            return static_cast<T>(val);
        }
        """

    package static var combinedHeader: String {
        "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"
            + simdTransformHeader + "\n" + dualScaleEpilogueHeader + "\n"
    }
}

extension BasaltMetalSubstrate {
    package static func cpuHadamardBlock(_ x: [Float], scale: [Float], blockSize: Int) -> [Float] {
        precondition(x.count.isMultiple(of: blockSize) && !scale.isEmpty)
        var out = [Float](repeating: 0, count: x.count)
        let norm = 1.0 / Float(blockSize).squareRoot()
        for base in stride(from: 0, to: x.count, by: blockSize) {
            var buf = (0 ..< blockSize).map { j in
                let col = base + j
                return x[col] * scale[col % scale.count]
            }
            var h = 1
            while h < blockSize {
                for i in stride(from: 0, to: blockSize, by: h * 2) {
                    for j in i ..< (i + h) {
                        let u = buf[j]
                        let v = buf[j + h]
                        buf[j] = u + v
                        buf[j + h] = u - v
                    }
                }
                h <<= 1
            }
            for k in 0 ..< blockSize {
                out[base + k] = buf[k] * norm
            }
        }
        return out
    }

    private static func cpuE2M1Magnitude(_ a: Float) -> Float {
        if a >= 5.0 { return 6.0 }
        if a >= 3.5 { return 4.0 }
        if a >= 2.5 { return 3.0 }
        if a >= 1.75 { return 2.0 }
        if a >= 1.25 { return 1.5 }
        if a >= 0.75 { return 1.0 }
        return a >= 0.25 ? 0.5 : 0.0
    }

    package static func cpuMXFP4QuantDequantBlock32(_ block: [Float]) -> [Float] {
        precondition(block.count == 32)
        if block.contains(where: \.isNaN) { return [Float](repeating: .nan, count: 32) }
        let maxAbs = block.map { abs($0) }.max() ?? 0
        if !(maxAbs > 0) { return [Float](repeating: 0, count: 32) }
        if maxAbs.isInfinite { return block.map { $0.isInfinite ? $0 : 0 } }
        let raw = maxAbs / 6.0
        let bits = raw.bitPattern
        var exp = Int((bits >> 23) & 255)
        if (bits & 0x7f_ffff) != 0, exp < 254 { exp += 1 }
        exp = min(max(exp, 0), 254)
        let scale = exp == 0 ? Float(bitPattern: 0x0040_0000) : Float(bitPattern: UInt32(exp) << 23)
        if !(scale > 0) { return [Float](repeating: 0, count: 32) }
        return block.map { v in
            let sv = v / scale
            return (sv < 0 ? -1.0 : 1.0) * cpuE2M1Magnitude(abs(sv)) * scale
        }
    }

    package static func cpuFP8E4M3QuantDequantBlock32(_ block: [Float]) -> [Float] {
        precondition(block.count == 32)
        if block.contains(where: \.isNaN) { return [Float](repeating: .nan, count: 32) }
        let maxAbs = block.map { abs($0) }.max() ?? 0
        if !(maxAbs > 0) { return [Float](repeating: 0, count: 32) }
        if maxAbs.isInfinite { return block.map { $0.isInfinite ? $0 : 0 } }
        let scale = maxAbs / 448.0
        if !(scale > 0) { return [Float](repeating: 0, count: 32) }
        return block.map { v in
            let sv = min(max(v / scale, -448.0), 448.0)
            let sgn: Float = sv < 0 ? -1.0 : 1.0
            let a = abs(sv)
            if a <= 0.0009765625 { return 0 }
            let e = min(max(Int((a.bitPattern >> 23) & 255) - 127, -6), 8)
            let step = scalbnf(1.0, Int32(e - 3))
            let q = rintf(a / step) * step
            return sgn * min(q, 448.0) * scale
        }
    }

    package static func broadcastShape(_ a: [Int], _ b: [Int]) -> [Int]? {
        let count = max(a.count, b.count)
        var result = [Int](repeating: 1, count: count)
        for i in 0 ..< count {
            let da = i < count - a.count ? 1 : a[i - (count - a.count)]
            let db = i < count - b.count ? 1 : b[i - (count - b.count)]
            if da == db {
                result[i] = da
            } else if da == 1 {
                result[i] = db
            } else if db == 1 {
                result[i] = da
            } else {
                return nil
            }
        }
        return result
    }
}
