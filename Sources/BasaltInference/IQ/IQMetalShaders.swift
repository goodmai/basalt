import BasaltCore
import Foundation

enum IQMetalShaders {
    static let simdgroups = 2

    static func header(for type: GGMLType) -> String {
        commonHeader + constantTable(for: type) + formatFunctions(for: type)
    }

    private static func table<T: FixedWidthInteger>(_ name: String, _ type: String, _ values: [T], suffix: String = "") -> String {
        let body = values.map { "0x" + String(UInt64(truncatingIfNeeded: $0) & (T.bitWidth == 64 ? .max : (1 << T.bitWidth) - 1), radix: 16) + suffix }
        return "constant \(type) \(name)[\(values.count)] = {\(body.joined(separator: ","))};\n"
    }

    private static func constantTable(for type: GGMLType) -> String {
        switch type {
        case .iq4NL, .iq4XS:
            "constant float iq_lut_const[16] = {\(ggmlValuesIQ4NL.map { "\($0).0f" }.joined(separator: ","))};\n"
        case .iq3S:
            table("iq_lut_const", "uint", ggmlIQ3SGrid, suffix: "u")
        case .iq3XXS:
            table("iq_lut_const", "uint", ggmlIQ3XXSGrid, suffix: "u") + table("iq_signs_const", "uchar", ggmlSignsIQ2XS)
        case .iq2S:
            table("iq_lut_const", "ulong", ggmlIQ2SGrid, suffix: "ul")
        case .iq2XS:
            table("iq_lut_const", "ulong", ggmlIQ2XSGrid, suffix: "ul") + table("iq_signs_const", "uchar", ggmlSignsIQ2XS)
        case .iq2XXS:
            table("iq_lut_const", "ulong", ggmlIQ2XXSGrid, suffix: "ul") + table("iq_signs_const", "uchar", ggmlSignsIQ2XS)
        case .iq1S, .iq1M:
            table("iq_lut_const", "ulong", ggmlIQ1SGrid, suffix: "ul")
        default:
            ""
        }
    }

    static func matvecSource(for type: GGMLType) -> String {
        let decls = lutDeclarations(for: type)
        let loadCall = lutLoadCall(for: type)
        let lutArgs = lutExtraArgs(for: type)
        return """
            const uint GPR = uint(K) / 32u, SUB = uint(BE) / 32u, ROW = (uint(K) / uint(BE)) * uint(BB);
            const uint N = uint(w_shape[0]);
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint row0 = (threadgroup_position_in_grid.y * 2u + simdgroup_index_in_threadgroup) * 4u;
            \(decls)
            \(loadCall)
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint g = lane; g < GPR; g += 32u) {
                device const T* xs = x + g * 32u;
                float4 xr[8];
                for (uint j = 0u; j < 8u; ++j) {
                    xr[j] = float4(xs[4u * j], xs[4u * j + 1u], xs[4u * j + 2u], xs[4u * j + 3u]);
                }
                uint blkOff = (g / SUB) * uint(BB), sub = g % SUB;
                for (uint r = 0u; r < 4u; ++r) {
                    acc[r] += iq_dot32(w + min(row0 + r, N - 1u) * ROW + blkOff, sub, xr\(lutArgs));
                }
            }
            for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
                float s = simd_sum(acc[r]);
                if (lane == 0u) out[row0 + r] = static_cast<T>(s);
            }
            """
    }

    static func smallMSource(for type: GGMLType) -> String {
        let decls = lutDeclarations(for: type)
        let loadCall = lutLoadCall(for: type)
        let lutArgs = lutExtraArgs(for: type)
        return """
            const uint GPR = uint(K) / 32u, SUB = uint(BE) / 32u, ROW = (uint(K) / uint(BE)) * uint(BB);
            const uint N = uint(w_shape[0]);
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint row0 = (threadgroup_position_in_grid.y * 2u + simdgroup_index_in_threadgroup) * 4u;
            \(decls)
            \(loadCall)
            float acc[4][M];
            for (uint r = 0u; r < 4u; ++r)
                for (uint m = 0u; m < uint(M); ++m) acc[r][m] = 0.0f;
            for (uint g = lane; g < GPR; g += 32u) {
                uint blkOff = (g / SUB) * uint(BB), sub = g % SUB;
                for (uint r = 0u; r < 4u; ++r) {
                    float4 wv[8];
                    float scale = iq_unpack32(w + min(row0 + r, N - 1u) * ROW + blkOff, sub, wv\(lutArgs));
                    for (uint m = 0u; m < uint(M); ++m) {
                        device const T* xs = x + m * uint(K) + g * 32u;
                        float s = 0.0f;
                        for (uint j = 0u; j < 8u; ++j) {
                            s += dot(wv[j], float4(xs[4u * j], xs[4u * j + 1u], xs[4u * j + 2u], xs[4u * j + 3u]));
                        }
                        acc[r][m] += scale * s;
                    }
                }
            }
            for (uint r = 0u; r < 4u && row0 + r < N; ++r) {
                for (uint m = 0u; m < uint(M); ++m) {
                    float sum = simd_sum(acc[r][m]);
                    if (lane == 0u) out[m * N + row0 + r] = static_cast<T>(sum);
                }
            }
            """
    }

    private static let commonHeader = """
        #include <metal_stdlib>
        using namespace metal;
        inline uint iq_u16(device const uchar* b, uint o) { return uint(*(device const ushort*)(b + o)); }
        inline uint iq_u32(device const uchar* b, uint o) {
            ushort2 p = *(device const packed_ushort2*)(b + o);
            return uint(p.x) | (uint(p.y) << 16);
        }
        inline float iq_h16(device const uchar* b, uint o) { return float(*(device const half*)(b + o)); }
        inline float4 iq_signed4(uint g, uint nibble) {
            uint t = ((nibble & 15u) * 0x204081u) & 0x01010101u;
            return float4(as_type<char4>((g ^ (t * 255u)) + t));
        }

        """

    private static func lutDeclarations(for type: GGMLType) -> String {
        switch type {
        case .iq4NL, .iq4XS: "(void)tid;"
        case .iq3S: "threadgroup uint lut_tg[512];"
        case .iq3XXS: "threadgroup uint lut_tg[256];"
        case .iq2S: "threadgroup ulong lut_tg[1024];"
        case .iq2XS: "threadgroup ulong lut_tg[512];"
        case .iq2XXS: "threadgroup ulong lut_tg[256];"
        case .iq1S, .iq1M: "threadgroup ulong lut_tg[2048];"
        default: "(void)tid;"
        }
    }

    private static func lutLoadCall(for type: GGMLType) -> String {
        switch type {
        case .iq4NL, .iq4XS: ""
        default: "iq_load_lut(lut_tg, tid);"
        }
    }

    private static func lutExtraArgs(for type: GGMLType) -> String {
        switch type {
        case .iq4NL, .iq4XS: ""
        default: ", lut_tg"
        }
    }

    private static func formatFunctions(for type: GGMLType) -> String {
        switch type {
        case .iq4NL: iq4NLFunctions
        case .iq4XS: iq4XSFunctions
        case .iq3S: iq3SFunctions
        case .iq3XXS: iq3XXSFunctions
        case .iq2S: iq2SFunctions
        case .iq2XS: iq2XSFunctions
        case .iq2XXS: iq2XXSFunctions
        case .iq1S: iq1SFunctions
        case .iq1M: iq1MFunctions
        default: ""
        }
    }
}

extension IQMetalShaders {
    fileprivate static let iq4NLFunctions = """
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr) {
            (void)g;
            ushort4 q0 = *(device const packed_ushort4*)(b + 2u);
            ushort4 q1 = *(device const packed_ushort4*)(b + 10u);
            uint4 qv = uint4(uint(q0.x) | (uint(q0.y) << 16), uint(q0.z) | (uint(q0.w) << 16),
                             uint(q1.x) | (uint(q1.y) << 16), uint(q1.z) | (uint(q1.w) << 16));
            float s = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 lo = as_type<uchar4>(qv[j] & 0x0f0f0f0fu), hi = as_type<uchar4>((qv[j] >> 4u) & 0x0f0f0f0fu);
                s += dot(float4(iq_lut_const[lo.x], iq_lut_const[lo.y], iq_lut_const[lo.z], iq_lut_const[lo.w]), xr[j]);
                s += dot(float4(iq_lut_const[hi.x], iq_lut_const[hi.y], iq_lut_const[hi.z], iq_lut_const[hi.w]), xr[j + 4u]);
            }
            return iq_h16(b, 0u) * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv) {
            (void)g;
            ushort4 q0 = *(device const packed_ushort4*)(b + 2u);
            ushort4 q1 = *(device const packed_ushort4*)(b + 10u);
            uint4 qv = uint4(uint(q0.x) | (uint(q0.y) << 16), uint(q0.z) | (uint(q0.w) << 16),
                             uint(q1.x) | (uint(q1.y) << 16), uint(q1.z) | (uint(q1.w) << 16));
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 lo = as_type<uchar4>(qv[j] & 0x0f0f0f0fu), hi = as_type<uchar4>((qv[j] >> 4u) & 0x0f0f0f0fu);
                wv[j] = float4(iq_lut_const[lo.x], iq_lut_const[lo.y], iq_lut_const[lo.z], iq_lut_const[lo.w]);
                wv[j + 4u] = float4(iq_lut_const[hi.x], iq_lut_const[hi.y], iq_lut_const[hi.z], iq_lut_const[hi.w]);
            }
            return iq_h16(b, 0u);
        }

        """

    fileprivate static let iq4XSFunctions = """
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr) {
            uint high = iq_u16(b, 2u);
            uint ls = ((uint(b[4u + g / 2u]) >> (4u * (g & 1u))) & 15u) | (((high >> (2u * g)) & 3u) << 4u);
            uint4 qv = *(device const packed_uint4*)(b + 8u + g * 16u);
            float s = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 lo = as_type<uchar4>(qv[j] & 0x0f0f0f0fu), hi = as_type<uchar4>((qv[j] >> 4u) & 0x0f0f0f0fu);
                s += dot(float4(iq_lut_const[lo.x], iq_lut_const[lo.y], iq_lut_const[lo.z], iq_lut_const[lo.w]), xr[j]);
                s += dot(float4(iq_lut_const[hi.x], iq_lut_const[hi.y], iq_lut_const[hi.z], iq_lut_const[hi.w]), xr[j + 4u]);
            }
            return iq_h16(b, 0u) * float(int(ls) - 32) * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv) {
            uint high = iq_u16(b, 2u);
            uint ls = ((uint(b[4u + g / 2u]) >> (4u * (g & 1u))) & 15u) | (((high >> (2u * g)) & 3u) << 4u);
            uint4 qv = *(device const packed_uint4*)(b + 8u + g * 16u);
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 lo = as_type<uchar4>(qv[j] & 0x0f0f0f0fu), hi = as_type<uchar4>((qv[j] >> 4u) & 0x0f0f0f0fu);
                wv[j] = float4(iq_lut_const[lo.x], iq_lut_const[lo.y], iq_lut_const[lo.z], iq_lut_const[lo.w]);
                wv[j + 4u] = float4(iq_lut_const[hi.x], iq_lut_const[hi.y], iq_lut_const[hi.z], iq_lut_const[hi.w]);
            }
            return iq_h16(b, 0u) * float(int(ls) - 32);
        }

        """

    fileprivate static let iq3SFunctions = """
        inline void iq_load_lut(threadgroup uint* grid, uint tid) {
            for (uint i = tid; i < 512u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const uint* grid) {
            uint sc = uint(b[106u + g / 2u]), qh = uint(b[66u + g]), signs4 = iq_u32(b, 74u + g * 4u);
            ushort4 pairs = *(device const packed_ushort4*)(b + 2u + g * 8u);
            float s = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                uint pair = uint(pairs[l]);
                uint first = (pair & 255u) | ((qh << (8u - 2u * l)) & 256u);
                uint second = (pair >> 8u) | ((qh << (7u - 2u * l)) & 256u);
                uint sb = (signs4 >> (8u * l)) & 255u;
                s += dot(iq_signed4(grid[first], sb), xr[2u * l]) + dot(iq_signed4(grid[second], sb >> 4u), xr[2u * l + 1u]);
            }
            return iq_h16(b, 0u) * float(1u + 2u * ((g & 1u) ? (sc >> 4u) : (sc & 15u))) * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const uint* grid) {
            uint sc = uint(b[106u + g / 2u]), qh = uint(b[66u + g]), signs4 = iq_u32(b, 74u + g * 4u);
            ushort4 pairs = *(device const packed_ushort4*)(b + 2u + g * 8u);
            for (uint l = 0u; l < 4u; ++l) {
                uint pair = uint(pairs[l]);
                uint first = (pair & 255u) | ((qh << (8u - 2u * l)) & 256u);
                uint second = (pair >> 8u) | ((qh << (7u - 2u * l)) & 256u);
                uint sb = (signs4 >> (8u * l)) & 255u;
                wv[2u * l] = iq_signed4(grid[first], sb);
                wv[2u * l + 1u] = iq_signed4(grid[second], sb >> 4u);
            }
            return iq_h16(b, 0u) * float(1u + 2u * ((g & 1u) ? (sc >> 4u) : (sc & 15u)));
        }

        """

    fileprivate static let iq3XXSFunctions = """
        inline void iq_load_lut(threadgroup uint* grid, uint tid) {
            for (uint i = tid; i < 256u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const uint* grid) {
            uint aux = iq_u32(b, 66u + 4u * g);
            ushort4 pairs = *(device const packed_ushort4*)(b + 2u + g * 8u);
            float s = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                uint pair = uint(pairs[l]);
                uint sb = uint(iq_signs_const[(aux >> (7u * l)) & 127u]);
                s += dot(iq_signed4(grid[pair & 255u], sb), xr[2u * l]) + dot(iq_signed4(grid[pair >> 8u], sb >> 4u), xr[2u * l + 1u]);
            }
            return iq_h16(b, 0u) * (0.5f + float(aux >> 28u)) * 0.5f * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const uint* grid) {
            uint aux = iq_u32(b, 66u + 4u * g);
            ushort4 pairs = *(device const packed_ushort4*)(b + 2u + g * 8u);
            for (uint l = 0u; l < 4u; ++l) {
                uint pair = uint(pairs[l]);
                uint sb = uint(iq_signs_const[(aux >> (7u * l)) & 127u]);
                wv[2u * l] = iq_signed4(grid[pair & 255u], sb);
                wv[2u * l + 1u] = iq_signed4(grid[pair >> 8u], sb >> 4u);
            }
            return iq_h16(b, 0u) * (0.5f + float(aux >> 28u)) * 0.5f;
        }

        """

    fileprivate static let iq2SFunctions = """
        inline void iq_load_lut(threadgroup ulong* grid, uint tid) {
            for (uint i = tid; i < 1024u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const ulong* grid) {
            uint sc = uint(b[74u + g]), qh = uint(b[66u + g]);
            uint qs4 = iq_u32(b, 2u + g * 4u), signs4 = iq_u32(b, 34u + g * 4u);
            float sLo = 0.0f, sHi = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[((qs4 >> (8u * l)) & 255u) | ((qh << (8u - 2u * l)) & 0x300u)];
                uint sb = (signs4 >> (8u * l)) & 255u;
                float term = dot(iq_signed4(uint(gv), sb), xr[2u * l]) + dot(iq_signed4(uint(gv >> 32u), sb >> 4u), xr[2u * l + 1u]);
                if (l < 2u) sLo += term; else sHi += term;
            }
            return iq_h16(b, 0u) * 0.25f * ((0.5f + float(sc & 15u)) * sLo + (0.5f + float(sc >> 4u)) * sHi);
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const ulong* grid) {
            uint sc = uint(b[74u + g]), qh = uint(b[66u + g]);
            uint qs4 = iq_u32(b, 2u + g * 4u), signs4 = iq_u32(b, 34u + g * 4u);
            float cLo = 0.5f + float(sc & 15u), cHi = 0.5f + float(sc >> 4u);
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[((qs4 >> (8u * l)) & 255u) | ((qh << (8u - 2u * l)) & 0x300u)];
                uint sb = (signs4 >> (8u * l)) & 255u;
                float c = (l < 2u) ? cLo : cHi;
                wv[2u * l] = c * iq_signed4(uint(gv), sb);
                wv[2u * l + 1u] = c * iq_signed4(uint(gv >> 32u), sb >> 4u);
            }
            return iq_h16(b, 0u) * 0.25f;
        }

        """
}

extension IQMetalShaders {
    fileprivate static let iq2XSFunctions = """
        inline void iq_load_lut(threadgroup ulong* grid, uint tid) {
            for (uint i = tid; i < 512u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const ulong* grid) {
            uint sc = uint(b[66u + g]);
            ushort4 qs = *(device const packed_ushort4*)(b + 2u + 8u * g);
            float sLo = 0.0f, sHi = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                uint q = uint(qs[l]);
                ulong gv = grid[q & 511u];
                uint sb = uint(iq_signs_const[q >> 9u]);
                float term = dot(iq_signed4(uint(gv), sb), xr[2u * l]) + dot(iq_signed4(uint(gv >> 32u), sb >> 4u), xr[2u * l + 1u]);
                if (l < 2u) sLo += term; else sHi += term;
            }
            return iq_h16(b, 0u) * 0.25f * ((0.5f + float(sc & 15u)) * sLo + (0.5f + float(sc >> 4u)) * sHi);
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const ulong* grid) {
            uint sc = uint(b[66u + g]);
            ushort4 qs = *(device const packed_ushort4*)(b + 2u + 8u * g);
            float cLo = 0.5f + float(sc & 15u), cHi = 0.5f + float(sc >> 4u);
            for (uint l = 0u; l < 4u; ++l) {
                uint q = uint(qs[l]);
                ulong gv = grid[q & 511u];
                uint sb = uint(iq_signs_const[q >> 9u]);
                float c = (l < 2u) ? cLo : cHi;
                wv[2u * l] = c * iq_signed4(uint(gv), sb);
                wv[2u * l + 1u] = c * iq_signed4(uint(gv >> 32u), sb >> 4u);
            }
            return iq_h16(b, 0u) * 0.25f;
        }

        """

    fileprivate static let iq2XXSFunctions = """
        inline void iq_load_lut(threadgroup ulong* grid, uint tid) {
            for (uint i = tid; i < 256u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const ulong* grid) {
            ushort4 cs = *(device const packed_ushort4*)(b + 2u + 8u * g);
            uint codes = uint(cs.x) | (uint(cs.y) << 16), scale = uint(cs.z) | (uint(cs.w) << 16);
            float s = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[(codes >> (8u * l)) & 255u];
                uint sb = uint(iq_signs_const[(scale >> (7u * l)) & 127u]);
                s += dot(iq_signed4(uint(gv), sb), xr[2u * l]) + dot(iq_signed4(uint(gv >> 32u), sb >> 4u), xr[2u * l + 1u]);
            }
            return iq_h16(b, 0u) * (0.5f + float(scale >> 28u)) * 0.25f * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const ulong* grid) {
            ushort4 cs = *(device const packed_ushort4*)(b + 2u + 8u * g);
            uint codes = uint(cs.x) | (uint(cs.y) << 16), scale = uint(cs.z) | (uint(cs.w) << 16);
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[(codes >> (8u * l)) & 255u];
                uint sb = uint(iq_signs_const[(scale >> (7u * l)) & 127u]);
                wv[2u * l] = iq_signed4(uint(gv), sb);
                wv[2u * l + 1u] = iq_signed4(uint(gv >> 32u), sb >> 4u);
            }
            return iq_h16(b, 0u) * (0.5f + float(scale >> 28u)) * 0.25f;
        }

        """

    fileprivate static let iq1SFunctions = """
        inline void iq_load_lut(threadgroup ulong* grid, uint tid) {
            for (uint i = tid; i < 2048u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const ulong* grid) {
            uint qh = iq_u16(b, 34u + 2u * g), qs4 = iq_u32(b, 2u + g * 4u);
            float delta = (qh & 0x8000u) ? -0.125f : 0.125f, s = 0.0f;
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[((qs4 >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8u)];
                s += dot(float4(as_type<char4>(uint(gv))) + delta, xr[2u * l])
                   + dot(float4(as_type<char4>(uint(gv >> 32u))) + delta, xr[2u * l + 1u]);
            }
            return iq_h16(b, 0u) * float(2u * ((qh >> 12u) & 7u) + 1u) * s;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const ulong* grid) {
            uint qh = iq_u16(b, 34u + 2u * g), qs4 = iq_u32(b, 2u + g * 4u);
            float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
            for (uint l = 0u; l < 4u; ++l) {
                ulong gv = grid[((qs4 >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8u)];
                wv[2u * l] = float4(as_type<char4>(uint(gv))) + delta;
                wv[2u * l + 1u] = float4(as_type<char4>(uint(gv >> 32u))) + delta;
            }
            return iq_h16(b, 0u) * float(2u * ((qh >> 12u) & 7u) + 1u);
        }

        """

    fileprivate static let iq1MFunctions = """
        inline void iq_load_lut(threadgroup ulong* grid, uint tid) {
            for (uint i = tid; i < 2048u; i += 64u) grid[i] = iq_lut_const[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr, threadgroup const ulong* grid) {
            ushort4 sv = *(device const packed_ushort4*)(b + 48u);
            float d = float(as_type<half>(ushort((sv.x >> 12u) | ((sv.y >> 8u) & 0xf0u) | ((sv.z >> 4u) & 0xf00u) | (sv.w & 0xf000u))));
            uint sc = uint(sv[g / 2u]), shift = 6u * (g & 1u);
            float dl0 = d * float(2u * ((sc >> shift) & 7u) + 1u), dl1 = d * float(2u * ((sc >> (shift + 3u)) & 7u) + 1u);
            uint qh = iq_u16(b, 32u + g * 2u), qs4 = iq_u32(b, g * 4u);
            uint qh0 = qh & 255u, qh1 = qh >> 8u;
            ulong g0 = grid[(qs4 & 255u) | ((qh0 << 8u) & 0x700u)], g1 = grid[((qs4 >> 8u) & 255u) | ((qh0 << 4u) & 0x700u)];
            ulong g2 = grid[((qs4 >> 16u) & 255u) | ((qh1 << 8u) & 0x700u)], g3 = grid[(qs4 >> 24u) | ((qh1 << 4u) & 0x700u)];
            float dt0 = (qh0 & 0x08u) ? -0.125f : 0.125f, dt1 = (qh0 & 0x80u) ? -0.125f : 0.125f;
            float dt2 = (qh1 & 0x08u) ? -0.125f : 0.125f, dt3 = (qh1 & 0x80u) ? -0.125f : 0.125f;
            float a = dot(float4(as_type<char4>(uint(g0))) + dt0, xr[0]) + dot(float4(as_type<char4>(uint(g0 >> 32u))) + dt0, xr[1])
                    + dot(float4(as_type<char4>(uint(g1))) + dt1, xr[2]) + dot(float4(as_type<char4>(uint(g1 >> 32u))) + dt1, xr[3]);
            float c = dot(float4(as_type<char4>(uint(g2))) + dt2, xr[4]) + dot(float4(as_type<char4>(uint(g2 >> 32u))) + dt2, xr[5])
                    + dot(float4(as_type<char4>(uint(g3))) + dt3, xr[6]) + dot(float4(as_type<char4>(uint(g3 >> 32u))) + dt3, xr[7]);
            return dl0 * a + dl1 * c;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv, threadgroup const ulong* grid) {
            ushort4 sv = *(device const packed_ushort4*)(b + 48u);
            float d = float(as_type<half>(ushort((sv.x >> 12u) | ((sv.y >> 8u) & 0xf0u) | ((sv.z >> 4u) & 0xf00u) | (sv.w & 0xf000u))));
            uint sc = uint(sv[g / 2u]), shift = 6u * (g & 1u);
            float c0 = float(2u * ((sc >> shift) & 7u) + 1u), c1 = float(2u * ((sc >> (shift + 3u)) & 7u) + 1u);
            uint qh = iq_u16(b, 32u + g * 2u), qs4 = iq_u32(b, g * 4u);
            uint qh0 = qh & 255u, qh1 = qh >> 8u;
            ulong g0 = grid[(qs4 & 255u) | ((qh0 << 8u) & 0x700u)], g1 = grid[((qs4 >> 8u) & 255u) | ((qh0 << 4u) & 0x700u)];
            ulong g2 = grid[((qs4 >> 16u) & 255u) | ((qh1 << 8u) & 0x700u)], g3 = grid[(qs4 >> 24u) | ((qh1 << 4u) & 0x700u)];
            float dt0 = (qh0 & 0x08u) ? -0.125f : 0.125f, dt1 = (qh0 & 0x80u) ? -0.125f : 0.125f;
            float dt2 = (qh1 & 0x08u) ? -0.125f : 0.125f, dt3 = (qh1 & 0x80u) ? -0.125f : 0.125f;
            wv[0] = c0 * (float4(as_type<char4>(uint(g0))) + dt0);
            wv[1] = c0 * (float4(as_type<char4>(uint(g0 >> 32u))) + dt0);
            wv[2] = c0 * (float4(as_type<char4>(uint(g1))) + dt1);
            wv[3] = c0 * (float4(as_type<char4>(uint(g1 >> 32u))) + dt1);
            wv[4] = c1 * (float4(as_type<char4>(uint(g2))) + dt2);
            wv[5] = c1 * (float4(as_type<char4>(uint(g2 >> 32u))) + dt2);
            wv[6] = c1 * (float4(as_type<char4>(uint(g3))) + dt3);
            wv[7] = c1 * (float4(as_type<char4>(uint(g3 >> 32u))) + dt3);
            return d;
        }

        """
}
