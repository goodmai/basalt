import XCTest
import MetalKit
@testable import GemCore

@MainActor
final class MetalRenderingTests: XCTestCase {
    
    func testMetalRendererInitialization() throws {
        // Skip on headless environments if no Metal device is available
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available for testing.")
        }
        
        let state = RainbowUIState()
        let renderer = RainbowRenderer(state: state)
        
        let mtkView = RainbowMTKView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), device: device)
        mtkView.delegate = renderer
        
        // Trigger resize and draw to ensure pipeline initializes
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
        
        // We cannot fully test draw(in:) without a currentDrawable, which requires a window.
        // But we can verify that the device is assigned and no immediate crashes occur.
        XCTAssertNotNil(mtkView.device, "MTKView should retain the Metal device.")
    }
    
    func testUIStateModeUpdates() {
        let state = RainbowUIState()
        XCTAssertEqual(state.currentMode, .idle)
        
        state.setMode(.generating)
        XCTAssertEqual(state.currentMode, .generating)
    }
    
    func testMessageAddition() {
        let state = RainbowUIState()
        state.addAssistantMessage("Test message")
        
        XCTAssertEqual(state.messages.count, 1)
        XCTAssertEqual(state.messages[0].text, "Test message")
        XCTAssertEqual(state.messages[0].role, .assistant)
    }
}
