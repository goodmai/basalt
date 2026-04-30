import MetalKit
import AppKit
import Combine

@MainActor
public class RainbowRenderer: NSObject, MTKViewDelegate {
    public var uiState: RainbowUIState
    public var cancellable: AnyCancellable?
    
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    public var pipelineState: MTLRenderPipelineState?
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
    
    private let logger = GemLogger(module: "RainbowRenderer")
    private var frameCount: Int = 0
    
    public init(state: RainbowUIState) {
        self.uiState = state
        self.startTime = CFAbsoluteTimeGetCurrent()
        super.init()
        logger.debug("RainbowRenderer initialized (waiting for Metal device)")
    }
    
    private func setupMetal(device: MTLDevice) {
        self.device = device
        logger.info("Setting up Metal with device: \(device.name)")
        
        self.commandQueue = device.makeCommandQueue()
        
        // Load Metal library — compile from bundled .metal source
        guard let library = Self.loadMetalLibrary(device: device) else {
            fputs("[Metal] ERROR: Failed to load metal library\n", stderr)
            return
        }
        
        // Setup text renderer
        if let commandQueue = commandQueue {
            self.textRenderer = TextRenderer(device: device, commandQueue: commandQueue, library: library)
            fputs("[Metal] TextRenderer ready\n", stderr)
        }
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertex_main")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragment_rainbow")
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
            fputs("[Metal] Pipeline state created successfully\n", stderr)
        } catch {
            fputs("[Metal] ERROR: Failed to create pipeline state: \(error)\n", stderr)
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
        if pipelineState == nil, let device = view.device {
            setupMetal(device: device)
        }
        
        // Heartbeat log every 60 frames
        frameCount += 1
        if frameCount % 60 == 0 {
            fputs("[Metal] Draw heartbeat frame \(frameCount)\n", stderr)
        }
        
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor else { return }
        
        guard let commandBuffer = commandQueue?.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        
        self.performDraw(in: renderEncoder, viewportSize: view.drawableSize)
        renderEncoder.endEncoding()
        
        // Capture screenshot if requested
        self.captureScreenshotIfNeeded(from: drawable.texture)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    /// Internal drawing logic shared between on-screen and off-screen rendering
    private func performDraw(in renderEncoder: MTLRenderCommandEncoder, viewportSize: CGSize) {
        guard let pipelineState = pipelineState, let vertexBuffer = vertexBuffer else { return }
        
        let currentTime = Float(CFAbsoluteTimeGetCurrent() - startTime)
        var uniforms = Uniforms(
            time: currentTime,
            resolution: SIMD2<Float>(Float(viewportSize.width), Float(viewportSize.height)),
            mode: Int32(uiState.currentMode.rawValue)
        )
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 1)
        renderEncoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        guard let textRenderer = textRenderer else { return }
        // Render UI overlay (messages, input, etc)
        renderUI(in: renderEncoder, textRenderer: textRenderer, viewportSize: viewportSize)
    }
        
    /// Render UI overlay (messages, input, etc)
    private func renderUI(in renderEncoder: MTLRenderCommandEncoder, textRenderer: TextRenderer, viewportSize: CGSize) {
        // Layout constants (in pixels at drawable scale)
        let scale: CGFloat = 2.0 // Assume Retina for headless/default
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
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
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
            let pos = CGPoint(x: viewportSize.width - size.width - padding, y: padding + 4 * scale)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
        }
        
        // 3. Draw Chat Messages
        var yOffset = headerHeight + padding - CGFloat(uiState.scrollOffset)
        let maxWidth = viewportSize.width - 2 * padding
        
        // Diagnostic: log message count periodically
        if frameCount % 120 == 1 {
            fputs("[Metal] renderUI: messages.count=\(uiState.messages.count), mode=\(uiState.currentMode), yOffset=\(yOffset)\n", stderr)
        }
        
        for (_, message) in uiState.messages.enumerated() {
            let prefix: String
            let color: NSColor
            let font: NSFont
            
            switch message.role {
            case .user:
                prefix = "❯ "
                color = .white
                font = bodyFont
            case .assistant:
                prefix = "⟫ "
                color = .white
                font = bodyFont
            case .system:
                prefix = "ℹ "
                color = NSColor(white: 0.7, alpha: 1.0)
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
                } else if rawLine.hasPrefix("![") && rawLine.contains("](") && rawLine.hasSuffix(")") {
                    // Simple markdown image parsing: ![alt](path)
                    if let start = rawLine.firstIndex(of: "("), let end = rawLine.lastIndex(of: ")") {
                        let path = String(rawLine[rawLine.index(after: start)..<end])
                        if let imgTex = loadTexture(from: path) {
                            let imgSize = CGSize(width: CGFloat(imgTex.width) / scale, height: CGFloat(imgTex.height) / scale)
                            let pos = CGPoint(x: padding, y: yOffset)
                            if yOffset + imgSize.height > headerHeight {
                                textRenderer.draw(texture: imgTex, size: imgSize, position: pos, resolution: viewportSize, encoder: renderEncoder)
                            }
                            yOffset += imgSize.height + 2 * scale
                        }
                    }
                    continue
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
                let textHeight = (lineData.text as NSString).size(withAttributes: [.font: lineData.font]).height
                let estimatedHeight = textHeight > 0 ? textHeight : lineData.font.boundingRectForFont.height
                
                if yOffset > viewportSize.height - footerHeight - padding {
                    yOffset += estimatedHeight + 2 * scale
                    continue
                }

                let cacheKey = "msg_\(message.id)_\(lineIndex)_\(lineData.text.hashValue)_\(lineData.bg.hash)"
                
                let isLastAssistantStreaming = (uiState.currentMode == .streaming || uiState.currentMode == .processing) && 
                                               message.role == .assistant && 
                                               message.id == uiState.messages.last?.id

                if isLastAssistantStreaming {
                    invalidateCache(for: cacheKey)
                }

                if let (tex, size) = getOrCreateTexture(key: cacheKey, text: lineData.text, font: lineData.font, color: lineData.color, backgroundColor: lineData.bg) {
                    let pos = CGPoint(x: padding, y: yOffset)
                    // Only draw if within visible area (below header)
                    if yOffset + size.height > headerHeight {
                        textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
                    }
                    yOffset += size.height + 2 * scale
                } else {
                    yOffset += estimatedHeight + 2 * scale
                }
            }
            yOffset += 8 * scale // gap between messages
        }
        
        let maxVisibleY = viewportSize.height - footerHeight - padding
        if yOffset > maxVisibleY && (uiState.currentMode == .streaming || uiState.currentMode == .processing) {
            let overflow = yOffset - maxVisibleY
            DispatchQueue.main.async {
                self.uiState.scrollOffset += overflow
            }
        }
        
        // 4. Draw Footer (input prompt)
        let footerY = viewportSize.height - footerHeight
        
        // Divider line (thin text-based)
        let divider = String(repeating: "─", count: 60)
        if let (tex, size) = getOrCreateTexture(key: "divider", text: divider, font: smallFont, color: NSColor(white: 0.4, alpha: 1.0)) {
            let pos = CGPoint(x: padding, y: footerY)
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
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
            textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
        }
        
        // TPS indicator
        if uiState.tokensPerSecond > 0 {
            let tpsText = String(format: "%.1f tok/s", uiState.tokensPerSecond)
            if let (tex, size) = getOrCreateTexture(key: "tps_\(tpsText)", text: tpsText, font: smallFont, color: NSColor(white: 0.5, alpha: 1.0)) {
                let pos = CGPoint(x: viewportSize.width - size.width - padding, y: footerY + 20 * scale)
                textRenderer.draw(texture: tex, size: size, position: pos, resolution: viewportSize, encoder: renderEncoder)
            }
        }
        
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
    
    public static func loadMetalLibrary(device: MTLDevice) -> MTLLibrary? {
        let logger = GemLogger(module: "MetalLoader")
        
        // 1. Try Bundle.module (compiled .metallib)
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            logger.debug("Loaded Metal library from Bundle.module")
            return lib
        }
        
        // 2. Try default library (if linked)
        if let lib = device.makeDefaultLibrary() {
            logger.debug("Loaded Metal library from default device library")
            return lib
        }

        // 3. Search for .metal source files in development environment
        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        let searchPaths = [
            currentDir.appendingPathComponent("Sources/Gem/UI/Metal/RainbowShaders.metal"),
            currentDir.appendingPathComponent("RainbowShaders.metal"),
            currentDir.appendingPathComponent("GemCore_GemCore.bundle/RainbowShaders.metal"),
            URL(fileURLWithPath: Bundle.main.bundlePath).appendingPathComponent("RainbowShaders.metal")
        ]
        
        for url in searchPaths {
            if fm.fileExists(atPath: url.path) {
                do {
                    let source = try String(contentsOf: url, encoding: .utf8)
                    let lib = try device.makeLibrary(source: source, options: nil)
                    logger.info("Successfully compiled Metal shaders from: \(url.path)")
                    return lib
                } catch {
                    logger.error("Found .metal at \(url.path) but compilation failed: \(error)")
                }
            }
        }
        
        logger.error("CRITICAL: Metal shaders not found. Checked: \(searchPaths.map { $0.path }.joined(separator: ", "))")
        return nil
    }
    
    // MARK: - Image Loading
    
    private lazy var textureLoader: MTKTextureLoader? = {
        guard let device = self.device else { return nil }
        return MTKTextureLoader(device: device)
    }()
    
    private var loadedImages: [String: MTLTexture] = [:]
    
    private func loadTexture(from path: String) -> MTLTexture? {
        if let tex = loadedImages[path] { return tex }
        
        let url: URL
        if path.hasPrefix("file://") {
            url = URL(string: path)!
        } else {
            url = URL(fileURLWithPath: path)
        }
        
        guard let loader = textureLoader else { return nil }
        let options: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: false,
            .SRGB: false
        ]
        
        do {
            let texture = try loader.newTexture(URL: url, options: options)
            loadedImages[path] = texture
            return texture
        } catch {
            print("Failed to load image at \(path): \(error)")
            return nil
        }
    }
    
    // MARK: - Public API
    
    /// Capture a screenshot to a file, even in headless mode
    public func captureScreenshot(to url: URL, width: Int = 1280, height: Int = 720) {
        logger.trace("Manual screenshot requested: \(url.path)")
        
        if pipelineState == nil {
            if let defaultDevice = MTLCreateSystemDefaultDevice() {
                setupMetal(device: defaultDevice)
            }
        }
        
        guard let device = device, let commandQueue = commandQueue else {
            logger.error("Metal not initialized for screenshot and failed to create default device")
            return
        }
        
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            logger.error("Failed to create offscreen texture")
            return
        }
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            logger.error("Failed to create command buffer or encoder")
            return
        }
        
        self.performDraw(in: renderEncoder, viewportSize: CGSize(width: width, height: height))
        renderEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        self.activeScreenshotURL = url
        self.captureScreenshotIfNeeded(from: texture)
    }
    
    // MARK: - Screenshot Capture
    
    public var activeScreenshotURL: URL?
    
    // Capture screenshot is called right before present
    private func captureScreenshotIfNeeded(from texture: MTLTexture) {
        guard let url = activeScreenshotURL else { return }
        logger.trace("Screenshot capture process started for: \(url.lastPathComponent)")
        self.activeScreenshotURL = nil // Only capture once per request
        
        let width = texture.width
        let height = texture.height
        let rowBytes = width * 4
        var rawData = [UInt8](repeating: 0, count: rowBytes * height)
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(&rawData, bytesPerRow: rowBytes, from: region, mipmapLevel: 0)
        
        // bgra8Unorm -> CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        
        guard let context = CGContext(data: &rawData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: bitmapInfo) else {
            logger.error("Failed to create CGContext for screenshot")
            return
        }
        guard let cgImage = context.makeImage() else {
            logger.error("Failed to make CGImage for screenshot")
            return
        }
        
        logger.trace("CGImage created successfully. Converting to PNG...")
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
        if let tiffData = nsImage.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            do {
                try pngData.write(to: url)
                logger.info("Screenshot successfully saved to \(url.path)")
            } catch {
                logger.error("Failed to write PNG data to disk: \(error.localizedDescription)")
            }
        } else {
            logger.error("Failed to create PNG representation for screenshot")
        }
    }
}
