import MetalKit
import SwiftUI
import AppKit
import ObjectiveC

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
        ZStack(alignment: .bottom) {
            // Metal background — takes full space but does NOT intercept events
            MetalViewRepresentable(state: state)
                .edgesIgnoringSafeArea(.all)
                .accessibilityIdentifier("RainbowMetalBackground")
            
            // Native NSTextField wrapped in NSViewRepresentable.
            // WHY: SwiftUI TextField inside NSHostingView on macOS cannot reliably
            // receive focus via makeFirstResponder — the field editor is never attached.
            // Native NSTextField bypasses this entirely and works like a real mac control.
            NativeInputField(
                text: $state.inputText,
                placeholder: state.currentMode == .idle ? "Введите ваш запрос…" : state.placeholderHint,
                onSubmit: submitMessage
            )
            .frame(height: 44)
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
            .accessibilityIdentifier("RainbowInputField")
        }
        .frame(minWidth: 600, minHeight: 400)
    }
    
    private func submitMessage() {
        var rawText = state.inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawText.isEmpty else { return }

        if rawText.lowercased() == "/clear" {
            state.clearHistory()
            return
        }

        if rawText.hasPrefix("/screenshot ") {            let path = rawText.dropFirst("/screenshot ".count)
            let url = URL(fileURLWithPath: String(path))
            state.captureScreenshotURL = url
            state.inputText = ""
            return
        }
        
        let isParallel = rawText.hasSuffix("/btw")
        if isParallel {
            rawText = String(rawText.dropLast(4)).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            guard state.currentMode == .idle || state.currentMode == .finished else { return }
        }
        
        let text = rawText
        state.addUserMessage(text)
        state.inputText = ""
        if !isParallel {
            state.setMode(.processing)
        }
        
        guard let currentOrchestrator = orchestrator else {
            // Mock mode
            Task {
                await state.coordinator.submit(state: RenderState(content: "🌈 MVI Demo — orchestrator not connected. Your message: \"\(text)\"", isGenerating: true))
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                await state.coordinator.submit(state: RenderState(content: "", isGenerating: false))
            }
            return
        }
        
        let currentMaxTokens = maxTokens
        
        let generationTask = Task.detached {
            fputs("[UI] MVI Intent: queuePrompt (isParallel: \(isParallel)) for: \(text.prefix(40))...\n", stderr)
            do {
                let request = GenerationRequest(prompt: text, maxTokens: currentMaxTokens)
                let stream = try await currentOrchestrator.generateStream(request: request)
                
                var currentAssistantText = ""
                
                for try await chunk in stream {
                    if Task.isCancelled { break }
                    switch chunk {
                    case .text(let t):
                        currentAssistantText += t
                        // Dispatch Event to Coordinator (30 FPS Throttling)
                        await state.coordinator.submit(state: RenderState(
                            content: currentAssistantText,
                            isGenerating: true,
                            fpsTarget: 30
                        ))
                    case .metadata(let m):
                        await MainActor.run { state.tokensPerSecond = m.tokensPerSecond }
                    }
                }
                
                // Final state
                if !Task.isCancelled {
                    await state.coordinator.submit(state: RenderState(
                        content: currentAssistantText,
                        isGenerating: false,
                        fpsTarget: 30
                    ))
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        state.addAssistantMessage("⚠️ Error: \(error.localizedDescription)")
                        state.setMode(.error)
                    }
                }
            }
        }
        
        state.activeGenerationTask = generationTask
    }
}

// MARK: - Native NSTextField wrapper

/// Wraps a native NSTextField so that focus, field editor, and keyboard input work
/// reliably on macOS inside NSHostingView — things SwiftUI TextField cannot guarantee.
public struct NativeInputField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String
    var onSubmit: () -> Void
    
    public func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
    
    public func makeNSView(context: Context) -> NSTextField {
        let tf = NSTextField()
        tf.delegate = context.coordinator
        tf.isBezeled = false
        tf.isBordered = false
        tf.drawsBackground = true
        tf.backgroundColor = NSColor.black.withAlphaComponent(0.45)
        tf.textColor = .white
        tf.font = NSFont.systemFont(ofSize: 16)
        tf.focusRingType = .none
        tf.placeholderAttributedString = NSAttributedString(
            string: placeholder,
            attributes: [
                .foregroundColor: NSColor.white.withAlphaComponent(0.45),
                .font: NSFont.systemFont(ofSize: 16)
            ]
        )
        tf.wantsLayer = true
        tf.layer?.cornerRadius = 8
        tf.layer?.masksToBounds = true
        
        // Give the field a 12pt horizontal inset via a cell subclass trick
        if let cell = tf.cell as? NSTextFieldCell {
            cell.usesSingleLineMode = true
            cell.lineBreakMode = .byTruncatingTail
            cell.wraps = false
        }
        
        // Auto-focus on first display
        DispatchQueue.main.async {
            tf.window?.makeFirstResponder(tf)
        }
        
        return tf
    }
    
    public func updateNSView(_ tf: NSTextField, context: Context) {
        // Safe NSTextField binding sync
        let isEditing = (tf.window?.firstResponder === tf.window?.fieldEditor(false, for: tf))
        
        if tf.stringValue != text {
            if isEditing {
                if text.isEmpty {
                    // To safely clear the field while editing (e.g. after Enter),
                    // we update stringValue and let the coordinator clear the fieldEditor
                    // via standard AppKit methods without forcing it directly to avoid IME deadlocks.
                    tf.stringValue = ""
                    if let editor = tf.window?.fieldEditor(false, for: tf) as? NSTextView {
                        editor.string = ""
                        editor.setSelectedRange(NSRange(location: 0, length: 0))
                    }
                }
            } else {
                tf.stringValue = text
            }
        }
        // Update placeholder when mode changes
        tf.placeholderAttributedString = NSAttributedString(
            string: placeholder,
            attributes: [
                .foregroundColor: NSColor.white.withAlphaComponent(0.45),
                .font: NSFont.systemFont(ofSize: 16)
            ]
        )
    }
    
    public class Coordinator: NSObject, NSTextFieldDelegate {
        var parent: NativeInputField
        init(parent: NativeInputField) { self.parent = parent }
        
        // Called on every keystroke → sync to binding
        public func controlTextDidChange(_ obj: Notification) {
            guard let tf = obj.object as? NSTextField else { return }
            parent.text = tf.stringValue
        }
        
        // Called when user presses Return
        public func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            if selector == #selector(NSResponder.insertNewline(_:)) {
                parent.onSubmit()
                return true
            }
            return false
        }
    }
}

public class RainbowMTKView: MTKView {
    var state: RainbowUIState?
    
    public override init(frame frameRect: CGRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
    }
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    // Do NOT accept first responder, let SwiftUI's TextField handle keyboard input
    public override var acceptsFirstResponder: Bool { false }
    
    // Forward mouse down so SwiftUI's tap gesture can catch it
    public override func mouseDown(with event: NSEvent) {
        super.mouseDown(with: event)
        self.nextResponder?.mouseDown(with: event)
    }
    
    private var redrawTimer: Timer?
    
    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
        if window != nil {
            redrawTimer?.invalidate()
            redrawTimer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self = self else { return }
                    self.delegate?.draw(in: self)
                }
            }
            // Add to common modes so it fires during UI interactions (scrolling, tracking)
            if let timer = redrawTimer {
                RunLoop.main.add(timer, forMode: .common)
            }
        }
    }
    
    public override func removeFromSuperview() {
        redrawTimer?.invalidate()
        redrawTimer = nil
        super.removeFromSuperview()
    }
    
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
        let logger = GemLogger(module: "MetalRepresentable")
        logger.info("makeNSView called - creating MTKView")
        
        let mtkView = RainbowMTKView()
        if let defaultDevice = MTLCreateSystemDefaultDevice() {
            mtkView.device = defaultDevice
            fputs("[Metal] Default device attached\n", stderr)
        } else {
            fputs("[Metal] ERROR: No default device found\n", stderr)
        }
        
        mtkView.state = state
        mtkView.delegate = context.coordinator
        
        // Standard Metal config
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.wantsLayer = true
        mtkView.layer?.isOpaque = true
        
        // Best practice for SwiftUI + MTKView on macOS when manually scheduling redraws
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true
        mtkView.preferredFramesPerSecond = 60
        mtkView.autoResizeDrawable = true
        
        context.coordinator.cancellable = state.objectWillChange.sink { [weak mtkView] _ in
            DispatchQueue.main.async {
                mtkView?.needsDisplay = true
            }
        }
        
        return mtkView
    }
    
    public func updateNSView(_ nsView: RainbowMTKView, context: Context) {
        context.coordinator.uiState = state
        nsView.state = state
        nsView.needsDisplay = true
        
        if let window = nsView.window {
            window.contentView?.wantsLayer = true
        }
        
        // Force at least one draw if pipeline is missing
        if context.coordinator.pipelineState == nil && nsView.drawableSize.width > 0 {
            fputs("[Metal] Forcing initial draw call from updateNSView\n", stderr)
            nsView.delegate?.draw(in: nsView)
        }
        
        if let url = state.captureScreenshotURL {
            context.coordinator.captureScreenshot(to: url)
            Task { @MainActor in
                state.captureScreenshotURL = nil
            }
        }
    }
    
    public func makeCoordinator() -> RainbowRenderer {
        RainbowRenderer(state: state)
    }
}
