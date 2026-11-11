import Foundation

extension GGUFMetalSource {
    static let dots = """
        inline float4 gguf_signed4(uint g, uint nibble) {
            uint t = ((nibble & 15u) * 0x204081u) & 0x01010101u;
            return float4(as_type<char4>((g ^ (t * 255u)) + t));
        }
        inline float gguf_dot8(ulong g, uint signs, thread const float4* x) {
            return dot(gguf_signed4(uint(g), signs), x[0]) + dot(gguf_signed4(uint(g >> 32), signs >> 4), x[1]);
        }
        inline float gguf_dot4x2(ulong first, ulong second, uint signs, thread const float4* x) {
            return dot(gguf_signed4(uint(first), signs), x[0]) + dot(gguf_signed4(uint(second), signs >> 4), x[1]);
        }
        template <int Q> inline float gguf_dot32(device const uchar* b, uint g, thread const float4* x, threadgroup const ulong* grid) {
            if (Q == 21) {
                uint sc = b[106 + g / 2], qh = b[66 + g];
                device const ushort* qs = (device const ushort*)(b + 2 + g * 8);
                device const ushort* signs = (device const ushort*)(b + 74 + g * 4);
                float s = 0.0f;
                for (uint l = 0; l < 4; ++l) {
                    uint pair = qs[l];
                    uint first = (pair & 0xff) | ((qh << (8 - 2 * l)) & 256), second = (pair >> 8) | ((qh << (7 - 2 * l)) & 256);
                    uint sb = (signs[l / 2] >> (8 * (l & 1))) & 0xff;
                    s += gguf_dot4x2(grid[first], grid[second], sb, x + 2 * l);
                }
                return gguf_h16(b, 0) * float(1 + 2 * ((g & 1) ? sc >> 4 : sc & 15)) * s;
            } else if (Q == 18) {
                uint aux = gguf_u32(b, 66 + 4 * g);
                device const ushort* qs = (device const ushort*)(b + 2 + g * 8);
                float s = 0.0f;
                for (uint l = 0; l < 4; ++l) {
                    uint pair = qs[l];
                    s += gguf_dot4x2(grid[pair & 0xff], grid[pair >> 8], gguf_ksigns[(aux >> (7 * l)) & 127], x + 2 * l);
                }
                return gguf_h16(b, 0) * (0.5f + float(aux >> 28)) * 0.5f * s;
            } else if (Q == 22) {
                uint sc = b[74 + g], qh = b[66 + g];
                float low = 0.0f, high = 0.0f;
                for (uint l = 0; l < 4; ++l) {
                    float v = gguf_dot8(grid[b[2 + g * 4 + l] | ((qh << (8 - 2 * l)) & 0x300)], b[34 + g * 4 + l], x + 2 * l);
                    if (l < 2) low += v; else high += v;
                }
                return gguf_h16(b, 0) * 0.25f * ((0.5f + float(sc & 15)) * low + (0.5f + float(sc >> 4)) * high);
            } else if (Q == 17) {
                uint sc = b[66 + g];
                device const ushort* qs = (device const ushort*)(b + 2 + 8 * g);
                float low = 0.0f, high = 0.0f;
                for (uint l = 0; l < 4; ++l) {
                    uint q = qs[l];
                    float v = gguf_dot8(grid[q & 511], gguf_ksigns[q >> 9], x + 2 * l);
                    if (l < 2) low += v; else high += v;
                }
                return gguf_h16(b, 0) * 0.25f * ((0.5f + float(sc & 15)) * low + (0.5f + float(sc >> 4)) * high);
            } else if (Q == 16) {
                uint codes = gguf_u32(b, 2 + 8 * g), scale = gguf_u32(b, 6 + 8 * g);
                float s = 0.0f;
                for (uint l = 0; l < 4; ++l) s += gguf_dot8(grid[(codes >> (8 * l)) & 255], gguf_ksigns[(scale >> (7 * l)) & 127], x + 2 * l);
                return gguf_h16(b, 0) * (0.5f + float(scale >> 28)) * 0.25f * s;
            } else if (Q == 23) {
                uint high = gguf_u16(b, 2);
                uint ls = ((uint(b[4 + g / 2]) >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4);
                device const uchar* q = b + 8 + g * 16;
                float s = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    uint v0 = q[4 * j], v1 = q[4 * j + 1], v2 = q[4 * j + 2], v3 = q[4 * j + 3];
                    s += dot(float4(gguf_kvalues[v0 & 15], gguf_kvalues[v1 & 15], gguf_kvalues[v2 & 15], gguf_kvalues[v3 & 15]), x[j]);
                    s += dot(float4(gguf_kvalues[v0 >> 4], gguf_kvalues[v1 >> 4], gguf_kvalues[v2 >> 4], gguf_kvalues[v3 >> 4]), x[j + 4]);
                }
                return gguf_h16(b, 0) * float(int(ls) - 32) * s;
            } else if (Q == 12) {
                uint s, m;
                if (g < 4) { s = b[4 + g] & 63; m = b[8 + g] & 63; }
                else { s = (b[8 + g] & 15) | ((b[g] >> 6) << 4); m = (b[8 + g] >> 4) | ((b[4 + g] >> 6) << 4); }
                device const uint* q = (device const uint*)(b + 16 + (g / 2) * 32);
                uint shift = (g & 1) * 4;
                float dotq = 0.0f, sumx = 0.0f;
                for (uint j = 0; j < 8; ++j) {
                    uint v = (q[j] >> shift) & 0x0f0f0f0fu;
                    float4 w = as_type<float4>(uint4(v, v >> 8, v >> 16, v >> 24) & 0xffu | 0x4B000000u) - 8388608.0f;
                    dotq += dot(w, x[j]);
                    sumx += x[j].x + x[j].y + x[j].z + x[j].w;
                }
                return gguf_h16(b, 0) * float(s) * dotq - gguf_h16(b, 2) * float(m) * sumx;
            } else {
                float y[32];
                gguf_decode32<Q>(b, g, y, grid);
                float s = 0.0f;
                for (uint j = 0; j < 8; ++j) s += dot(float4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]), x[j]);
                return s;
            }
        }

        """
}
