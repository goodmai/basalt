import MetalKit
import CoreGraphics
import CoreText
import AppKit

public class TextRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState!
    private var samplerState: MTLSamplerState!
    
    // Shader uniforms
    private struct TextUniforms {
        var resolution: SIMD2<Float>
        var offset: SIMD2<Float>
        var size: SIMD2<Float>
        var textColor: SIMD4<Float>
    }
    
    private let vertices: [Float] = [
        0.0, 0.0,  0.0, 1.0,
        1.0, 0.0,  1.0, 1.0,
        0.0, 1.0,  0.0, 0.0,
        1.0, 1.0,  1.0, 0.0
    ]
    private var vertexBuffer: MTLBuffer!
    
    public init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        setupMetal()
    }
    
    private func setupMetal() {
        guard let library = try? device.makeDefaultLibrary(bundle: Bundle.module) else { return }
        
        let vertexFunction = library.makeFunction(name: "vertex_text")
        let fragmentFunction = library.makeFunction(name: "fragment_text")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Enable blending for text
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.size * 2
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.size * 4
        
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        pipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: .storageModeShared)
        
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerState = device.makeSamplerState(descriptor: samplerDescriptor)
    }
    
    public func createTexture(from text: String, font: NSFont, color: NSColor) -> (MTLTexture?, CGSize) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: color
        ]
        let attrString = NSAttributedString(string: text, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attrString)
        let size = CTFramesetterSuggestFrameSizeWithConstraints(framesetter, CFRangeMake(0, 0), nil, CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude), nil)
        
        let width = max(Int(ceil(size.width)), 1)
        let height = max(Int(ceil(size.height)), 1)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo) else {
            return (nil, size)
        }
        
        // Clear background
        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw text
        context.textMatrix = .identity
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1.0, y: -1.0)
        
        let path = CGPath(rect: CGRect(x: 0, y: 0, width: width, height: height), transform: nil)
        let frame = CTFramesetterCreateFrame(framesetter, CFRangeMake(0, 0), path, nil)
        CTFrameDraw(frame, context)
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        textureDescriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { return (nil, size) }
        
        if let data = context.data {
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: data, bytesPerRow: width * 4)
        }
        
        return (texture, size)
    }
    
    public func draw(texture: MTLTexture, size: CGSize, position: CGPoint, resolution: CGSize, encoder: MTLRenderCommandEncoder) {
        guard let pipelineState = pipelineState, let vertexBuffer = vertexBuffer else { return }
        
        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        
        var uniforms = TextUniforms(
            resolution: SIMD2<Float>(Float(resolution.width), Float(resolution.height)),
            offset: SIMD2<Float>(Float(position.x), Float(position.y)),
            size: SIMD2<Float>(Float(size.width), Float(size.height)),
            textColor: SIMD4<Float>(1, 1, 1, 1) // unused when texture has color
        )
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<TextUniforms>.size, index: 1)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentSamplerState(samplerState, index: 0)
        
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}
