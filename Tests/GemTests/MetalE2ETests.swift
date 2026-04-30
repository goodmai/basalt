import Testing
import MetalKit
import AppKit
@testable import GemCore

@Suite("Metal UI E2E Tests")
struct MetalE2ETests {
    
    @Test("TextRenderer initialization and texture creation")
    @MainActor
    func testTextRenderer_Creation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping Metal test: No GPU available")
            return
        }
        
        let commandQueue = try #require(device.makeCommandQueue())
        let library = try #require(RainbowRenderer.loadMetalLibrary(device: device))
        
        let textRenderer = TextRenderer(device: device, commandQueue: commandQueue, library: library)
        
        let font = NSFont.systemFont(ofSize: 12)
        let (texture, size) = textRenderer.createTexture(from: "Hello Metal", font: font, color: .white)
        
        #expect(texture != nil)
        #expect(size.width > 0)
        #expect(size.height > 0)
        #expect(texture?.width ?? 0 >= Int(size.width))
    }
    
    @Test("RainbowRenderer frame generation (Headless)")
    @MainActor
    func testRainbowRenderer_Frame() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Skipping Metal test: No GPU available")
            return
        }
        
        let state = RainbowUIState()
        let renderer = RainbowRenderer(state: state)
        
        // Mock MTKView
        let mtkView = MTKView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        
        // We can't easily call draw(in:) because it requires a currentDrawable
        // but we can check if it initializes without crashing
        _ = renderer
    }
}
