import MetalKit
import SwiftUI

public struct RainbowChatView: NSViewRepresentable {
    @ObservedObject var state: RainbowUIState
    
    public init(state: RainbowUIState) {
        self.state = state
    }
    
    public func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        
        if let defaultDevice = MTLCreateSystemDefaultDevice() {
            mtkView.device = defaultDevice
        }
        
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        
        return mtkView
    }
    
    public func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.uiState = state
    }
    
    public func makeCoordinator() -> RainbowRenderer {
        RainbowRenderer(state: state)
    }
}
