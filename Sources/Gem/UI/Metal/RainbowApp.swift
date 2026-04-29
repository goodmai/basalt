import AppKit
import SwiftUI

@available(macOS 10.15, *)
public class RainbowAppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var state: RainbowUIState
    var orchestrator: ModelOrchestratorActor?
    var maxTokens: Int
    
    public init(state: RainbowUIState, orchestrator: ModelOrchestratorActor? = nil, maxTokens: Int = 65536) {
        self.state = state
        self.orchestrator = orchestrator
        self.maxTokens = maxTokens
        super.init()
    }
    
    public func applicationDidFinishLaunching(_ aNotification: Notification) {
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
    }
    
    public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@MainActor
@available(macOS 10.15, *)
public func launchRainbowUI(state: RainbowUIState, orchestrator: ModelOrchestratorActor? = nil, maxTokens: Int = 65536) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = RainbowAppDelegate(state: state, orchestrator: orchestrator, maxTokens: maxTokens)
    app.delegate = delegate
    app.run()
}
