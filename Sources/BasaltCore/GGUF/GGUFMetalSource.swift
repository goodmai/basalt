import Foundation

public enum GGUFMetalSource {
    public static let header = helpers + grids + decoders + dots

    static func table<T: FixedWidthInteger>(_ name: String, _ type: String, _ values: [T], suffix: String = "") -> String {
        let body = values.map { "0x" + String(UInt64(truncatingIfNeeded: $0) & (T.bitWidth == 64 ? .max : (1 << T.bitWidth) - 1), radix: 16) + suffix }
        return "constant \(type) \(name)[\(values.count)] = {\(body.joined(separator: ","))};\n"
    }

    static let grids = table("gguf_iq1s", "ulong", ggmlIQ1SGrid, suffix: "ul") + table("gguf_iq2xxs", "ulong", ggmlIQ2XXSGrid, suffix: "ul")
        + table("gguf_iq2xs", "ulong", ggmlIQ2XSGrid, suffix: "ul") + table("gguf_iq2s", "ulong", ggmlIQ2SGrid, suffix: "ul")
        + table("gguf_iq3xxs", "uint", ggmlIQ3XXSGrid, suffix: "u") + table("gguf_iq3s", "uint", ggmlIQ3SGrid, suffix: "u")
        + table("gguf_ksigns", "uchar", ggmlSignsIQ2XS)
        + "constant float gguf_kvalues[16] = {\(ggmlValuesIQ4NL.map { "\($0).0f" }.joined(separator: ","))};\n"

    static let helpers = """
        #include <metal_stdlib>
        using namespace metal;
        inline uint gguf_u16(device const uchar* b, uint o) { return uint(b[o]) | (uint(b[o + 1]) << 8); }
        inline uint gguf_u32(device const uchar* b, uint o) { return gguf_u16(b, o) | (gguf_u16(b, o + 2) << 16); }
        inline float gguf_h16(device const uchar* b, uint o) { return float(as_type<half>(ushort(gguf_u16(b, o)))); }
        inline float gguf_byte(ulong v, uint j) { return float((v >> (8 * j)) & 0xff); }
        inline float gguf_signed(float v, uint signs, uint bit) { return ((signs >> bit) & 1) ? -v : v; }

        """

    public static func gridSize(_ type: GGMLType) -> Int {
        switch type {
        case .iq1M: ggmlIQ1SGrid.count
        case .iq2XXS: ggmlIQ2XXSGrid.count
        case .iq2XS: ggmlIQ2XSGrid.count
        case .iq2S: ggmlIQ2SGrid.count
        case .iq3XXS: ggmlIQ3XXSGrid.count
        case .iq3S: ggmlIQ3SGrid.count
        default: 1
        }
    }

    static let decoders = """
        template <int Q> inline ulong gguf_grid_value(uint i) {
            if (Q == 29) return gguf_iq1s[i];
            if (Q == 16) return gguf_iq2xxs[i];
            if (Q == 17) return gguf_iq2xs[i];
            if (Q == 22) return gguf_iq2s[i];
            if (Q == 18) return gguf_iq3xxs[i];
            if (Q == 21) return gguf_iq3s[i];
            return 0;
        }
        template <int Q, int GN> inline void gguf_load_grid(threadgroup ulong* grid, uint first, uint stride) {
            for (uint i = first; i < uint(GN); i += stride) grid[i] = gguf_grid_value<Q>(i);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline void gguf_lattice8(thread float* y, float db, ulong grid, uint signs) {
            for (uint j = 0; j < 8; ++j) y[j] = db * gguf_signed(gguf_byte(grid, j), signs, j);
        }
        inline void gguf_lattice4x2(thread float* y, float db, ulong first, ulong second, uint signs) {
            for (uint j = 0; j < 4; ++j) {
                y[j] = db * gguf_signed(gguf_byte(first, j), signs, j);
                y[j + 4] = db * gguf_signed(gguf_byte(second, j), signs, j + 4);
            }
        }
        template <int Q> inline void gguf_decode32(device const uchar* b, uint g, thread float* y, threadgroup const ulong* grid) {
            if (Q == 8) {
                float d = gguf_h16(b, 0);
                for (uint j = 0; j < 32; ++j) y[j] = d * float(as_type<char>(b[2 + j]));
            } else if (Q == 20) {
                float d = gguf_h16(b, 0);
                for (uint j = 0; j < 16; ++j) { y[j] = d * gguf_kvalues[b[2 + j] & 15]; y[j + 16] = d * gguf_kvalues[b[2 + j] >> 4]; }
            } else if (Q == 23) {
                uint high = gguf_u16(b, 2);
                uint ls = ((uint(b[4 + g / 2]) >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4);
                float dl = gguf_h16(b, 0) * float(int(ls) - 32);
                for (uint j = 0; j < 16; ++j) { uint q = b[8 + g * 16 + j]; y[j] = dl * gguf_kvalues[q & 15]; y[j + 16] = dl * gguf_kvalues[q >> 4]; }
            } else if (Q == 12) {
                uint s, m;
                if (g < 4) { s = b[4 + g] & 63; m = b[8 + g] & 63; }
                else { s = (b[8 + g] & 15) | ((b[g] >> 6) << 4); m = (b[8 + g] >> 4) | ((b[4 + g] >> 6) << 4); }
                float scale = gguf_h16(b, 0) * float(s), offset = gguf_h16(b, 2) * float(m);
                for (uint l = 0; l < 32; ++l) { uint q = b[16 + (g / 2) * 32 + l]; y[l] = scale * float((g & 1) ? q >> 4 : q & 15) - offset; }
            } else if (Q == 10) {
                float d = gguf_h16(b, 80), dmin = gguf_h16(b, 82);
                for (uint h = 0; h < 2; ++h) {
                    uint sc = b[g * 2 + h];
                    float dl = d * float(sc & 15), ml = dmin * float(sc >> 4);
                    for (uint l = 0; l < 16; ++l) y[h * 16 + l] = dl * float((b[16 + (g / 4) * 32 + h * 16 + l] >> (2 * (g % 4))) & 3) - ml;
                }
            } else if (Q == 14) {
                float d = gguf_h16(b, 208);
                uint chunk = g / 4, part = g % 4;
                for (uint l = 0; l < 32; ++l) {
                    uint low = b[chunk * 64 + l + (part & 1) * 32];
                    uint q = (part < 2 ? low & 15 : low >> 4) | (((b[128 + chunk * 32 + l] >> (2 * part)) & 3) << 4);
                    y[l] = d * float(as_type<char>(b[192 + chunk * 8 + l / 16 + 2 * part])) * float(int(q) - 32);
                }
            } else if (Q == 21) {
                uint sc = b[106 + g / 2];
                float db = gguf_h16(b, 0) * float(1 + 2 * ((g & 1) ? sc >> 4 : sc & 15));
                uint qh = b[66 + g];
                for (uint l = 0; l < 4; ++l) {
                    uint first = b[2 + g * 8 + 2 * l] | ((qh << (8 - 2 * l)) & 256);
                    uint second = b[2 + g * 8 + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
                    gguf_lattice4x2(y + l * 8, db, grid[first], grid[second], b[74 + g * 4 + l]);
                }
            } else if (Q == 18) {
                uint aux = gguf_u32(b, 66 + 4 * g);
                float db = gguf_h16(b, 0) * (0.5f + float(aux >> 28)) * 0.5f;
                for (uint l = 0; l < 4; ++l)
                    gguf_lattice4x2(y + l * 8, db, grid[b[2 + g * 8 + 2 * l]], grid[b[2 + g * 8 + 2 * l + 1]], gguf_ksigns[(aux >> (7 * l)) & 127]);
            } else if (Q == 22) {
                float d = gguf_h16(b, 0);
                uint sc = b[74 + g], qh = b[66 + g];
                for (uint l = 0; l < 4; ++l) {
                    float db = d * (0.5f + float(l < 2 ? sc & 15 : sc >> 4)) * 0.25f;
                    gguf_lattice8(y + l * 8, db, grid[b[2 + g * 4 + l] | ((qh << (8 - 2 * l)) & 0x300)], b[34 + g * 4 + l]);
                }
            } else if (Q == 17) {
                float d = gguf_h16(b, 0);
                uint sc = b[66 + g];
                for (uint l = 0; l < 4; ++l) {
                    uint q = gguf_u16(b, 2 + 2 * (4 * g + l));
                    gguf_lattice8(y + l * 8, d * (0.5f + float(l < 2 ? sc & 15 : sc >> 4)) * 0.25f, grid[q & 511], gguf_ksigns[q >> 9]);
                }
            } else if (Q == 16) {
                uint codes = gguf_u32(b, 2 + 8 * g), scale = gguf_u32(b, 6 + 8 * g);
                float db = gguf_h16(b, 0) * (0.5f + float(scale >> 28)) * 0.25f;
                for (uint l = 0; l < 4; ++l) gguf_lattice8(y + l * 8, db, grid[(codes >> (8 * l)) & 255], gguf_ksigns[(scale >> (7 * l)) & 127]);
            } else if (Q == 29) {
                uint s0 = gguf_u16(b, 48), s1 = gguf_u16(b, 50), s2 = gguf_u16(b, 52), s3 = gguf_u16(b, 54);
                float d = float(as_type<half>(ushort((s0 >> 12) | ((s1 >> 8) & 0xf0) | ((s2 >> 4) & 0xf00) | (s3 & 0xf000))));
                uint sc = gguf_u16(b, 48 + 2 * (g / 2)), shift = 6 * (g % 2);
                float dl[2] = {d * float(2 * ((sc >> shift) & 7) + 1), d * float(2 * ((sc >> (shift + 3)) & 7) + 1)};
                uint qh0 = b[32 + g * 2], qh1 = b[33 + g * 2];
                uint index[4] = {b[g * 4] | ((qh0 << 8) & 0x700), b[g * 4 + 1] | ((qh0 << 4) & 0x700), b[g * 4 + 2] | ((qh1 << 8) & 0x700), b[g * 4 + 3] | ((qh1 << 4) & 0x700)};
                float delta[4] = {(qh0 & 0x08) ? -0.125f : 0.125f, (qh0 & 0x80) ? -0.125f : 0.125f, (qh1 & 0x08) ? -0.125f : 0.125f, (qh1 & 0x80) ? -0.125f : 0.125f};
                for (uint l = 0; l < 4; ++l)
                    for (uint j = 0; j < 8; ++j) y[l * 8 + j] = dl[l / 2] * (float(as_type<char>(uchar((grid[index[l]] >> (8 * j)) & 0xff))) + delta[l]);
            }
        }

        """
}
