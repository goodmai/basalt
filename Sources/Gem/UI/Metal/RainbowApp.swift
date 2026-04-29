import AppKit
import SwiftUI

@available(macOS 10.15, *)
public class RainbowAppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var state: RainbowUIState
    
    public init(state: RainbowUIState) {
        self.state = state
        super.init()
    }
    
    public func applicationDidFinishLaunching(_ aNotification: Notification) {
        let chatView = RainbowChatView(state: state)
        // Add ignore safe area to fill the whole window
        let hostingView = NSHostingView(rootView: chatView.edgesIgnoringSafeArea(.all))
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.center()
        window.setFrameAutosaveName("RainbowChatWindow")
        window.contentView = hostingView
        window.makeKeyAndOrderFront(nil)
        window.title = "Gemm Rainbow Chat"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        
        // Bring app to front
        NSApp.activate(ignoringOtherApps: true)
    }
    
    public func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@MainActor
@available(macOS 10.15, *)
public func launchRainbowUI(state: RainbowUIState) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = RainbowAppDelegate(state: state)
    app.delegate = delegate
    app.run()
}
