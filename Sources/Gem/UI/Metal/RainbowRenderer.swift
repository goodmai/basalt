import MetalKit
import AppKit

public class RainbowRenderer: NSObject, MTKViewDelegate {
    public var uiState: RainbowUIState
    
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    private var startTime: CFAbsoluteTime
    
    // Shader uniforms matching Metal struct
    private struct Uniforms {
        var time: Float
        var resolution: SIMD2<Float>
        var mode: Int32
    }
    
    // Full screen quad
    private let vertices: [Float] = [
        -1.0, -1.0,  0.0, 1.0,
         1.0, -1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 0.0,
         1.0,  1.0,  1.0, 0.0
    ]
    private var vertexBuffer: MTLBuffer?
    
    // Text rendering
    private var textRenderer: TextRenderer?
    
    // Cached text textures
    private var cachedTextures: [String: (MTLTexture, CGSize)] = [:]
    private var maxCacheSize = 100
    
    public init(state: RainbowUIState) {
        self.uiState = state
        self.startTime = CFAbsoluteTimeGetCurrent()
        super.init()
        self.setupMetal()
    }
    
    private func setupMetal() {
        self.device = MTLCreateSystemDefaultDevice()
        guard let device = device else { return }
        
        self.commandQueue = device.makeCommandQueue()
        
        // Load Metal library — compile from bundled .metal source
        guard let library = Self.loadMetalLibrary(device: device) else {
            print("Failed to load metal library")
            return
        }
        
        // Setup text renderer
        if let commandQueue = commandQueue {
            self.textRenderer = TextRenderer(device: device, commandQueue: commandQueue, library: library)
        }
        
        let vertexFunction = library.makeFunction(name: "vertex_main")
        let fragmentFunction = library.makeFunction(name: "fragment_rainbow")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2 // position
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float2 // uv
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.size * 2
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.size * 4
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create pipeline state: \(error)")
        }
        
        vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Float>.size, options: .storageModeShared)
    }
    
    // MARK: - Texture caching
    
    private func getOrCreateTexture(key: String, text: String, font: NSFont, color: NSColor, backgroundColor: NSColor = .clear) -> (MTLTexture, CGSize)? {
        if let cached = cachedTextures[key] {
            return cached
        }
        guard let textRenderer = textRenderer else { return nil }
        let (tex, size) = textRenderer.createTexture(from: text, font: font, color: color, backgroundColor: backgroundColor)
        guard let texture = tex else { return nil }
        
        // Evict old entries if cache is too big
        if cachedTextures.count >= maxCacheSize {
            cachedTextures.removeAll()
        }
        
        cachedTextures[key] = (texture, size)
        return (texture, size)
    }
    
    private func invalidateCache(for key: String) {
        cachedTextures.removeValue(forKey: key)
    }
    
    // MARK: - MTKViewDelegate
    
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Invalidate all textures on resize since resolution changed
        cachedTextures.removeAll()
    }
    
    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let pipelineState = pipelineState,
              let commandQueue = commandQueue,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor),
              let vertexBuffer = vertexBuffer else {
            return
        }
        
        let currentTime = Float(CFAbsoluteTimeGetCurrent() - startTime)
        let resolution = view.drawableSize
        
        var uniforms = Uniforms(
            time: currentTime,
            resolution: SIMD2<Float>(Float(resolution.width), Float(resolution.height)),
            mode: Int32(uiState.currentMode.rawValue)
        )
        
        // 1. Draw Rainbow Background
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 1)
        renderEncoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        guard let textRenderer = textRenderer else {
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
            return
        }
        
        // Layout constants (in pixels at drawable scale)
        let scale = view.window?.backingScaleFactor ?? 2.0
        let headerHeight: CGFloat = 44 * scale
        let footerHeight: CGFloat = 56 * scale
        let padding: CGFloat = 16 * scale
        let fontSize: CGFloat = 15 * scale
        let smallFontSize: CGFloat = 12 * scale
        
        let headerFont = NSFont.systemFont(ofSize: fontSize, weight: .bold)
        let bodyFont = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
        let smallFont = NSFont.monospacedSystemFont(ofSize: smallFontSize, weight: .regular)
        
        // 2. Draw Header
        let headerText = "🌈 Gemm  •  \(uiState.modelName)"
        if let (tex, size) = getOrCreateTexture(key: "header_\(headerText)", text: headerText, font: headerFont, color: .white) {
            let pos = CGPoint(x: padding, y: padding)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
        }
        
        // Mode indicator on the right
        let modeText: String
        let modeColor: NSColor
        switch uiState.currentMode {
        case .idle:     modeText = "● IDLE";       modeColor = NSColor(calibratedRed: 0.3, green: 0.9, blue: 0.4, alpha: 1.0)
        case .processing: modeText = "◉ PROCESSING"; modeColor = NSColor(calibratedRed: 1.0, green: 0.8, blue: 0.0, alpha: 1.0)
        case .streaming:  modeText = "◉ STREAMING";  modeColor = NSColor(calibratedRed: 0.4, green: 0.7, blue: 1.0, alpha: 1.0)
        case .error:    modeText = "✕ ERROR";      modeColor = NSColor(calibratedRed: 1.0, green: 0.3, blue: 0.3, alpha: 1.0)
        case .finished: modeText = "✓ FINISHED";   modeColor = NSColor(calibratedRed: 0.3, green: 0.9, blue: 0.4, alpha: 1.0)
        }
        if let (tex, size) = getOrCreateTexture(key: "mode_\(modeText)", text: modeText, font: smallFont, color: modeColor) {
            let pos = CGPoint(x: resolution.width - size.width - padding, y: padding + 4 * scale)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
        }
        
        // 3. Draw Chat Messages
        var yOffset = headerHeight + padding - CGFloat(uiState.scrollOffset)
        let maxWidth = resolution.width - 2 * padding
        
        for (_, message) in uiState.messages.enumerated() {
            let prefix: String
            let color: NSColor
            let font: NSFont
            
            switch message.role {
            case .user:
                prefix = "❯ "
                color = NSColor(calibratedRed: 0.6, green: 0.8, blue: 1.0, alpha: 1.0)
                font = bodyFont
            case .assistant:
                prefix = "⟫ "
                color = NSColor(calibratedRed: 0.9, green: 0.9, blue: 0.95, alpha: 1.0)
                font = bodyFont
            case .system:
                prefix = "ℹ "
                color = NSColor(calibratedRed: 0.5, green: 0.5, blue: 0.6, alpha: 1.0)
                font = smallFont
            }
            
            // Split by newlines and handle code blocks/diffs
            let fullText = prefix + message.text
            let rawLines = fullText.components(separatedBy: .newlines)
            var linesToRender: [(text: String, font: NSFont, color: NSColor, bg: NSColor)] = []
            
            var inCodeBlock = false
            for rawLine in rawLines {
                if rawLine.hasPrefix("```") {
                    inCodeBlock.toggle()
                    linesToRender.append((text: rawLine, font: smallFont, color: NSColor(white: 0.7, alpha: 1.0), bg: NSColor(white: 0.15, alpha: 1.0)))
                    continue
                }
                
                let lineFont = inCodeBlock ? smallFont : font
                let bg: NSColor
                let fg: NSColor
                
                if inCodeBlock {
                    bg = NSColor(white: 0.15, alpha: 1.0)
                    fg = NSColor(calibratedRed: 0.8, green: 0.9, blue: 0.8, alpha: 1.0)
                } else if message.role == .assistant && rawLine.hasPrefix("+") { // diff added
                    bg = NSColor(calibratedRed: 0.1, green: 0.4, blue: 0.1, alpha: 0.5)
                    fg = NSColor.white
                } else if message.role == .assistant && rawLine.hasPrefix("-") { // diff removed
                    bg = NSColor(calibratedRed: 0.4, green: 0.1, blue: 0.1, alpha: 0.5)
                    fg = NSColor.white
                } else {
                    bg = .clear
                    fg = color
                }
                
                let wrapped = wordWrap(rawLine, font: lineFont, maxWidth: maxWidth)
                for w in wrapped {
                    linesToRender.append((text: w, font: lineFont, color: fg, bg: bg))
                }
            }
            
            for (lineIndex, lineData) in linesToRender.enumerated() {
                let cacheKey = "msg_\(message.id)_\(lineIndex)_\(lineData.text.hashValue)_\(lineData.bg.hash)"
                if let (tex, size) = getOrCreateTexture(key: cacheKey, text: lineData.text, font: lineData.font, color: lineData.color, backgroundColor: lineData.bg) {
                    let pos = CGPoint(x: padding, y: yOffset)
                    // Only draw if within visible area (below header)
                    if yOffset + size.height > headerHeight {
                        textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
                    }
                    yOffset += size.height + 2 * scale
                }
            }
            yOffset += 8 * scale // gap between messages
            
            // Stop drawing if we've gone past the footer area
            if yOffset > resolution.height - footerHeight - padding {
                break
            }
        }
        
        // 4. Draw Footer (input prompt)
        let footerY = resolution.height - footerHeight
        
        // Divider line (thin text-based)
        let divider = String(repeating: "─", count: 60)
        if let (tex, size) = getOrCreateTexture(key: "divider", text: divider, font: smallFont, color: NSColor(white: 0.4, alpha: 1.0)) {
            let pos = CGPoint(x: padding, y: footerY)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
        }
        
        // Input text or placeholder
        let inputDisplay: String
        let inputColor: NSColor
        if uiState.inputText.isEmpty {
            inputDisplay = "❯ " + uiState.placeholderHint
            inputColor = NSColor(white: 0.45, alpha: 1.0)
        } else {
            inputDisplay = "❯ " + uiState.inputText + "█" // cursor block
            inputColor = .white
        }
        
        // Invalidate input texture every frame since it changes with typing
        invalidateCache(for: "input_field")
        if let (tex, size) = getOrCreateTexture(key: "input_field", text: inputDisplay, font: bodyFont, color: inputColor) {
            let pos = CGPoint(x: padding, y: footerY + 18 * scale)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
        }
        
        // TPS indicator
        if uiState.tokensPerSecond > 0 {
            let tpsText = String(format: "%.1f tok/s", uiState.tokensPerSecond)
            if let (tex, size) = getOrCreateTexture(key: "tps_\(tpsText)", text: tpsText, font: smallFont, color: NSColor(white: 0.5, alpha: 1.0)) {
                let pos = CGPoint(x: resolution.width - size.width - padding, y: footerY + 20 * scale)
                textRenderer.draw(texture: tex, size: size, position: pos, resolution: resolution, encoder: renderEncoder)
            }
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    // MARK: - Word wrapping
    
    private func wordWrap(_ text: String, font: NSFont, maxWidth: CGFloat) -> [String] {
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let fullSize = (text as NSString).size(withAttributes: attributes)
        
        if fullSize.width <= maxWidth {
            return [text]
        }
        
        var lines: [String] = []
        var currentLine = ""
        let words = text.split(separator: " ", omittingEmptySubsequences: false).map(String.init)
        
        for word in words {
            let test = currentLine.isEmpty ? word : currentLine + " " + word
            let testSize = (test as NSString).size(withAttributes: attributes)
            
            if testSize.width > maxWidth && !currentLine.isEmpty {
                lines.append(currentLine)
                currentLine = word
            } else {
                currentLine = test
            }
        }
        
        if !currentLine.isEmpty {
            lines.append(currentLine)
        }
        
        return lines.isEmpty ? [text] : lines
    }
    
    // MARK: - Metal Library Loading
    
    private static func loadMetalLibrary(device: MTLDevice) -> MTLLibrary? {
        // Strategy 1: Try loading from Bundle.module (compiled metallib)
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }
        
        // Strategy 2: Try the default library (if shaders are in default.metallib)
        if let lib = device.makeDefaultLibrary() {
            // Verify our functions exist
            if lib.makeFunction(name: "vertex_main") != nil {
                return lib
            }
        }
        
        // Strategy 3: Compile from .metal source bundled as resource
        if let metalURL = Bundle.module.url(forResource: "RainbowShaders", withExtension: "metal") {
            do {
                let source = try String(contentsOf: metalURL, encoding: .utf8)
                let lib = try device.makeLibrary(source: source, options: nil)
                return lib
            } catch {
                print("Failed to compile Metal shader from source: \(error)")
            }
        }
        
        // Strategy 4: Look for .metal file relative to executable
        let execDir = Bundle.main.bundlePath
        let possiblePaths = [
            "\(execDir)/../Sources/Gem/UI/Metal/RainbowShaders.metal",
            "\(execDir)/../../Sources/Gem/UI/Metal/RainbowShaders.metal",
            "Sources/Gem/UI/Metal/RainbowShaders.metal"
        ]
        
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path) {
                do {
                    let source = try String(contentsOfFile: path, encoding: .utf8)
                    let lib = try device.makeLibrary(source: source, options: nil)
                    return lib
                } catch {
                    print("Failed to compile Metal shader from \(path): \(error)")
                }
            }
        }
        
        print("Could not find Metal shaders in any known location")
        return nil
    }
}
