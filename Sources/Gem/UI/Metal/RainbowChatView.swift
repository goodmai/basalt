import MetalKit
import SwiftUI

public struct RainbowChatView: View {
    @ObservedObject var state: RainbowUIState
    var orchestrator: ModelOrchestratorActor?
    var maxTokens: Int
    
    public init(state: RainbowUIState, orchestrator: ModelOrchestratorActor? = nil, maxTokens: Int = 65536) {
        self.state = state
        self.orchestrator = orchestrator
        self.maxTokens = maxTokens
    }
    
    public var body: some View {
        ZStack(alignment: .bottomLeading) {
            MetalViewRepresentable(state: state)
                .edgesIgnoringSafeArea(.all)
                .accessibilityIdentifier("RainbowMetalBackground")
            
            // Transparent text field to capture keyboard input
            TextField("", text: $state.inputText)
                .textFieldStyle(PlainTextFieldStyle())
                .foregroundColor(.clear)
                .background(Color.clear)
                .accentColor(.clear)
                .opacity(0.01)
                .frame(height: 30)
                .padding(.bottom, 10)
                .padding(.horizontal, 16)
                .accessibilityIdentifier("RainbowInputField")
                .onSubmit {
                    submitMessage()
                }
        }
        .onAppear {
            state.setMode(.idle)
        }
        .frame(minWidth: 600, minHeight: 400)
    }
    
    private func submitMessage() {
        let text = state.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        state.addUserMessage(text)
        state.inputText = ""
        state.setMode(.processing)
        
        guard let orchestrator = orchestrator else {
            // Demo mode — no orchestrator connected
            Task {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await MainActor.run {
                    state.addAssistantMessage("🌈 Rainbow UI demo — orchestrator not connected. Your message: \"\(text)\"")
                    state.setMode(.finished)
                }
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                await MainActor.run {
                    state.setMode(.idle)
                }
            }
            return
        }
        
        // Real inference
        Task {
            do {
                let request = GenerationRequest(prompt: text, maxTokens: maxTokens)
                let stream = try await orchestrator.generateStream(request: request)
                
                await MainActor.run {
                    state.setMode(.streaming)
                    state.addAssistantMessage("")
                }
                
                var tps: Double = 0
                for try await chunk in stream {
                    switch chunk {
                    case .text(let t):
                        await MainActor.run {
                            state.appendToLastAssistant(t)
                        }
                    case .metadata(let m):
                        tps = m.tokensPerSecond
                    }
                }
                
                await MainActor.run {
                    state.tokensPerSecond = tps
                    state.setMode(.finished)
                }
                
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                await MainActor.run {
                    state.setMode(.idle)
                }
            } catch {
                await MainActor.run {
                    state.addAssistantMessage("Error: \(error.localizedDescription)")
                    state.setMode(.error)
                }
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                await MainActor.run {
                    state.setMode(.idle)
                }
            }
        }
    }
}

public class RainbowMTKView: MTKView {
    var state: RainbowUIState?
    
    public override func scrollWheel(with event: NSEvent) {
        guard let state = state else { return }
        Task { @MainActor in
            // Basic scroll handling
            state.scrollOffset -= event.scrollingDeltaY
            if state.scrollOffset < 0 {
                state.scrollOffset = 0
            }
        }
    }
    
    // Accessibility tree generation for Agent testing
    public override func accessibilityRole() -> NSAccessibility.Role? {
        return .group
    }
    
    public override func accessibilityLabel() -> String? {
        return "Chat History"
    }
    
    public override func accessibilityChildren() -> [Any]? {
        guard let state = state else { return nil }
        
        var children: [NSAccessibilityElement] = []
        
        // 1. Header Marker
        let header = NSAccessibilityElement()
        header.setAccessibilityRole(.staticText)
        header.setAccessibilityValue("Header: Gemm • \(state.modelName) - Status: \(state.currentMode)")
        // Normally we'd set bounds here (accessibilityFrame)
        children.append(header)
        
        // 2. Message Markers
        for msg in state.messages {
            let msgElement = NSAccessibilityElement()
            msgElement.setAccessibilityRole(.staticText)
            let rolePrefix = msg.role == .user ? "User: " : "Assistant: "
            msgElement.setAccessibilityValue(rolePrefix + msg.text)
            children.append(msgElement)
        }
        
        // 3. Footer Marker
        let footer = NSAccessibilityElement()
        footer.setAccessibilityRole(.staticText)
        footer.setAccessibilityValue("Input field. Current text: \(state.inputText)")
        children.append(footer)
        
        return children
    }
}

public struct MetalViewRepresentable: NSViewRepresentable {
    @ObservedObject var state: RainbowUIState
    
    public init(state: RainbowUIState) {
        self.state = state
    }
    
    public func makeNSView(context: Context) -> RainbowMTKView {
        let mtkView = RainbowMTKView()
        mtkView.state = state
        
        if let defaultDevice = MTLCreateSystemDefaultDevice() {
            mtkView.device = defaultDevice
        }
        
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 60
        
        return mtkView
    }
    
    public func updateNSView(_ nsView: RainbowMTKView, context: Context) {
        context.coordinator.uiState = state
        nsView.state = state
    }
    
    public func makeCoordinator() -> RainbowRenderer {
        RainbowRenderer(state: state)
    }
}
