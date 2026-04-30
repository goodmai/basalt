#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float time;
    float2 resolution;
    int mode;
};

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple HSL to RGB conversion
float3 hsl2rgb(float h, float s, float l) {
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = l - c / 2.0;
    
    float3 rgb;
    if (h < 1.0/6.0) rgb = float3(c, x, 0.0);
    else if (h < 2.0/6.0) rgb = float3(x, c, 0.0);
    else if (h < 3.0/6.0) rgb = float3(0.0, c, x);
    else if (h < 4.0/6.0) rgb = float3(0.0, x, c);
    else if (h < 5.0/6.0) rgb = float3(x, 0.0, c);
    else rgb = float3(c, 0.0, x);
    
    return rgb + m;
}

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant Uniforms &uniforms [[buffer(1)]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fragment float4 fragment_rainbow(VertexOut in [[stage_in]],
                                 constant Uniforms &uniforms [[buffer(0)]]) {
    // Rainbow background effect based on mode
    float speed = 0.2;
    if (uniforms.mode == 1) { // processing
        speed = 1.5;
    } else if (uniforms.mode == 2) { // streaming
        speed = 0.8;
    } else if (uniforms.mode == 3) { // error
        speed = 2.0;
        float hue = 0.0; // Red
        float pulse = sin(uniforms.time * speed) * 0.5 + 0.5;
        float3 rgb = hsl2rgb(hue, 1.0, 0.5 * pulse + 0.2);
        return float4(rgb, 1.0);
    }
    
    float hue = fract(in.uv.x * 2.0 + uniforms.time * speed);
    float3 rgb = hsl2rgb(hue, 0.8, 0.1); // Dark rainbow background (terminal style)
    return float4(rgb, 1.0);
}

struct TextUniforms {
    float2 resolution;
    float2 offset;
    float2 size;
    float4 textColor;
};

vertex VertexOut vertex_text(VertexIn in [[stage_in]],
                             constant TextUniforms &uniforms [[buffer(1)]]) {
    VertexOut out;
    
    // Scale and translate
    float2 pixelPosition = (in.position * uniforms.size) + uniforms.offset;
    
    // Convert to Normalized Device Coordinates (NDC)
    // NDC is -1 to 1, with y pointing up in Metal (or down depending on projection, we'll map top-left to -1, 1)
    // Wait, typical 2D maps 0,0 to top left.
    // Let's assume standard Metal where (-1,-1) is bottom-left, (1,1) is top-right.
    float x = (pixelPosition.x / uniforms.resolution.x) * 2.0 - 1.0;
    float y = 1.0 - (pixelPosition.y / uniforms.resolution.y) * 2.0; // Invert Y
    
    out.position = float4(x, y, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

fragment float4 fragment_text(VertexOut in [[stage_in]],
                              texture2d<float> texture [[texture(0)]],
                              sampler textureSampler [[sampler(0)]]) {
    return texture.sample(textureSampler, in.uv);
}
