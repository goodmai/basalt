import BasaltCore
import Foundation

extension IQMetalShaders {
    static let q2KFunctions = """
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr) {
            float d = iq_h16(b, 80u), dmin = iq_h16(b, 82u);
            uint shift = 2u * (g & 3u), qBase = 16u + (g >> 2u) * 32u;
            uint sc0 = uint(b[2u * g]), sc1 = uint(b[2u * g + 1u]);
            float dl0 = d * float(sc0 & 15u), ml0 = dmin * float(sc0 >> 4u);
            float dl1 = d * float(sc1 & 15u), ml1 = dmin * float(sc1 >> 4u);
            float acc = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 q0 = as_type<uchar4>((iq_u32(b, qBase + 4u * j) >> shift) & 0x03030303u);
                uchar4 q1 = as_type<uchar4>((iq_u32(b, qBase + 16u + 4u * j) >> shift) & 0x03030303u);
                acc += dot(dl0 * float4(q0) - ml0, xr[j]);
                acc += dot(dl1 * float4(q1) - ml1, xr[4u + j]);
            }
            return acc;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv) {
            float d = iq_h16(b, 80u), dmin = iq_h16(b, 82u);
            uint shift = 2u * (g & 3u), qBase = 16u + (g >> 2u) * 32u;
            uint sc0 = uint(b[2u * g]), sc1 = uint(b[2u * g + 1u]);
            float dl0 = d * float(sc0 & 15u), ml0 = dmin * float(sc0 >> 4u);
            float dl1 = d * float(sc1 & 15u), ml1 = dmin * float(sc1 >> 4u);
            for (uint j = 0u; j < 4u; ++j) {
                uchar4 q0 = as_type<uchar4>((iq_u32(b, qBase + 4u * j) >> shift) & 0x03030303u);
                uchar4 q1 = as_type<uchar4>((iq_u32(b, qBase + 16u + 4u * j) >> shift) & 0x03030303u);
                wv[j] = dl0 * float4(q0) - ml0;
                wv[4u + j] = dl1 * float4(q1) - ml1;
            }
            return 1.0f;
        }

        """

    static let q3KFunctions = """
        inline float iq_q3k_scale(device const uchar* b, uint s) {
            uint sc = ((uint(b[96u + (s & 7u)]) >> (4u * (s >> 3u))) & 15u)
                    | (((uint(b[104u + (s & 3u)]) >> (2u * (s >> 2u))) & 3u) << 4u);
            return float(int(sc) - 32);
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr) {
            float d = iq_h16(b, 108u);
            float c0 = iq_q3k_scale(b, 2u * g), c1 = iq_q3k_scale(b, 2u * g + 1u);
            uint shift = 2u * (g & 3u), qBase = 32u + (g >> 2u) * 32u;
            float s0 = 0.0f, s1 = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                uint ql0 = (iq_u32(b, qBase + 4u * j) >> shift) & 0x03030303u;
                uint qh0 = ((iq_u32(b, 4u * j) >> g) & 0x01010101u) << 2u;
                uint ql1 = (iq_u32(b, qBase + 16u + 4u * j) >> shift) & 0x03030303u;
                uint qh1 = ((iq_u32(b, 16u + 4u * j) >> g) & 0x01010101u) << 2u;
                s0 += dot(float4(as_type<uchar4>(ql0 | qh0)) - 4.0f, xr[j]);
                s1 += dot(float4(as_type<uchar4>(ql1 | qh1)) - 4.0f, xr[4u + j]);
            }
            return d * (c0 * s0 + c1 * s1);
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv) {
            float d = iq_h16(b, 108u);
            float c0 = iq_q3k_scale(b, 2u * g), c1 = iq_q3k_scale(b, 2u * g + 1u);
            uint shift = 2u * (g & 3u), qBase = 32u + (g >> 2u) * 32u;
            for (uint j = 0u; j < 4u; ++j) {
                uint ql0 = (iq_u32(b, qBase + 4u * j) >> shift) & 0x03030303u;
                uint qh0 = ((iq_u32(b, 4u * j) >> g) & 0x01010101u) << 2u;
                uint ql1 = (iq_u32(b, qBase + 16u + 4u * j) >> shift) & 0x03030303u;
                uint qh1 = ((iq_u32(b, 16u + 4u * j) >> g) & 0x01010101u) << 2u;
                wv[j] = c0 * (float4(as_type<uchar4>(ql0 | qh0)) - 4.0f);
                wv[4u + j] = c1 * (float4(as_type<uchar4>(ql1 | qh1)) - 4.0f);
            }
            return d;
        }

        """

    static let q4KFunctions = """
        inline float2 iq_q4k_scale_min(device const uchar* b, uint g) {
            uint sc, mn;
            if (g < 4u) {
                sc = uint(b[4u + g]) & 63u;
                mn = uint(b[8u + g]) & 63u;
            } else {
                sc = (uint(b[8u + g]) & 15u) | ((uint(b[g]) >> 6u) << 4u);
                mn = (uint(b[8u + g]) >> 4u) | ((uint(b[4u + g]) >> 6u) << 4u);
            }
            return float2(iq_h16(b, 0u) * float(sc), iq_h16(b, 2u) * float(mn));
        }
        inline float iq_dot32(device const uchar* b, uint g, thread const float4* xr) {
            float2 sm = iq_q4k_scale_min(b, g);
            uint qBase = 16u + (g >> 1u) * 32u, shift = (g & 1u) * 4u;
            float dotq = 0.0f, sumx = 0.0f;
            for (uint j = 0u; j < 8u; ++j) {
                uchar4 nib = as_type<uchar4>((iq_u32(b, qBase + 4u * j) >> shift) & 0x0f0f0f0fu);
                dotq += dot(float4(nib), xr[j]);
                sumx += xr[j].x + xr[j].y + xr[j].z + xr[j].w;
            }
            return sm.x * dotq - sm.y * sumx;
        }
        inline float iq_unpack32(device const uchar* b, uint g, thread float4* wv) {
            float2 sm = iq_q4k_scale_min(b, g);
            uint qBase = 16u + (g >> 1u) * 32u, shift = (g & 1u) * 4u;
            for (uint j = 0u; j < 8u; ++j) {
                uchar4 nib = as_type<uchar4>((iq_u32(b, qBase + 4u * j) >> shift) & 0x0f0f0f0fu);
                wv[j] = sm.x * float4(nib) - sm.y;
            }
            return 1.0f;
        }

        """
}
