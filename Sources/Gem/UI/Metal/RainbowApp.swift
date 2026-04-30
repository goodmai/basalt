import AppKit
import SwiftUI

@available(macOS 10.15, *)
@MainActor
public class RainbowAppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var state: RainbowUIState
    var orchestrator: ModelOrchestratorActor?
    var maxTokens: Int
    private let logger = GemLogger(module: "RainbowApp")
    
    public init(state: RainbowUIState, orchestrator: ModelOrchestratorActor? = nil, maxTokens: Int = 65536) {
        self.state = state
        self.orchestrator = orchestrator
        self.maxTokens = maxTokens
        super.init()
    }
    
    public func applicationDidFinishLaunching(_ aNotification: Notification) {
        logger.info("Rainbow Chat UI launched. Mode: \(state.isAgentMode ? "AGENT" : "INTERACTIVE")")
        logger.trace("Initializing chat view with \(orchestrator == nil ? "no" : "active") orchestrator")
        let chatView = RainbowChatView(state: state, orchestrator: orchestrator, maxTokens: maxTokens)
        let hostingView = NSHostingView(rootView: chatView.edgesIgnoringSafeArea(.all))
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 650),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.center()
        window.setFrameAutosaveName("RainbowChatWindow")
        window.contentView = hostingView
        window.makeKeyAndOrderFront(nil)
        window.title = "Gemm Rainbow Chat"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.backgroundColor = .black
        window.minSize = NSSize(width: 600, height: 400)
        
        // Dark appearance
        window.appearance = NSAppearance(named: .darkAqua)
        
        // Bring app to front
        NSApp.activate(ignoringOtherApps: true)
        
        // Register Global Event Bus for Shortcuts (MVI Intent)
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self = self else { return event }
            return self.handleGlobalShortcut(event)
        }
        
        // Automated Agent Suite
        if state.isAgentMode {
            startAgentSuite()
        }
    }
    
    @objc
    public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    private func handleGlobalShortcut(_ event: NSEvent) -> NSEvent? {
        let isControl = event.modifierFlags.contains(.control)
        let isShift = event.modifierFlags.contains(.shift)
        
        switch event.keyCode {
        case 53: // Esc
            logger.info("MVI Intent: Cancel (Esc)")
            state.cancelGeneration()
            return nil
        case 8 where isControl: // Ctrl+C
            logger.info("MVI Intent: Cancel (Ctrl+C)")
            state.cancelGeneration()
            return nil
        case 2 where isControl: // Ctrl+D
            logger.info("MVI Intent: Exit (Ctrl+D)")
            NSApplication.shared.terminate(self)
            return nil
        case 31 where isControl: // Ctrl+O
            logger.info("MVI Intent: Focus Terminal (Ctrl+O)")
            // TODO: switch focus
            return nil
        case 48 where isShift: // Shift+Tab
            logger.info("MVI Intent: Toggle Auto-Confirm (Shift+Tab)")
            // TODO: toggle confirm
            return nil
        default:
            return event
        }
    }
    
    private func startAgentSuite() {
        guard let orchestrator = orchestrator else {
            fputs("[Agent] ERROR: No orchestrator for real suite\n", stderr)
            Darwin.exit(1)
        }
        
        let currentMaxTokens = maxTokens
        let fm = FileManager.default
        let imagesDir = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("images")
        try? fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)
        
        let agentLogger = { (msg: String) in fputs("[chat] \(msg)\n", stderr) }
        
        Task { @MainActor in
            _ = agentLogger("[Agent] Starting REAL agentic test suite within GUI event loop...")
            
            let tasks = [
                ("Arithmetic", "Calculate 123 * 456 + 789 / 3"),
                ("Algebra", "Solve for x: 2x + 5 = 15. Show steps."),
                ("Translation", "Translate 'The quick brown fox jumps over the lazy dog' to Russian.")
            ]
            
            for (name, prompt) in tasks {
                _ = agentLogger("[Agent] Task: \(name) -> \(prompt)")
                state.addUserMessage(prompt)
                state.setMode(.processing)
                
                do {
                    let request = GenerationRequest(prompt: prompt, maxTokens: currentMaxTokens)
                    let stream = try await orchestrator.generateStream(request: request)
                    
                    state.setMode(.streaming)
                    state.addAssistantMessage("")
                    
                    for try await chunk in stream {
                        if case .text(let t) = chunk {
                            state.appendToLastAssistant(t)
                        }
                    }
                    
                    state.setMode(.finished)
                    let screenshotPath = imagesDir.appendingPathComponent("real_test_\(name.lowercased()).png")
                    state.captureScreenshotURL = screenshotPath
                    _ = agentLogger("[Agent] Task \(name) finished. Screenshot queued to \(screenshotPath.lastPathComponent)")
                    
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    state.setMode(.idle)
                } catch {
                    _ = agentLogger("[Agent] ERROR in task \(name): \(error.localizedDescription)")
                    state.setMode(.error)
                }
            }
            
            _ = agentLogger("[Agent] ALL TASKS COMPLETE. Exiting.")
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            Darwin.exit(0)
        }
    }
    

}

@MainActor
private var globalDelegate: RainbowAppDelegate?

@MainActor
@available(macOS 10.15, *)
public func launchRainbowUI(state: RainbowUIState, orchestrator: ModelOrchestratorActor? = nil, maxTokens: Int = 65536) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    
    let delegate = RainbowAppDelegate(state: state, orchestrator: orchestrator, maxTokens: maxTokens)
    globalDelegate = delegate // Keep alive!
    app.delegate = delegate
    
    fputs("[App] NSApp.run() starting...\n", stderr)
    app.run()
}
