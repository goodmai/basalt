import Foundation
import AppKit
import SwiftUI
@testable import GemCore

@MainActor
func runUITest() {
    print("Starting UI Test...")
    
    let state = RainbowUIState()
    state.modelName = "Agent Test Model"
    state.addUserMessage("Show me a code block and a diff.")
    state.addAssistantMessage("Here is some code:\n```\nfunc test() {\n    print(\"Hello\")\n}\n```\nAnd a diff:\n+ added line\n- removed line")
    
    // Create view
    let view = RainbowChatView(state: state)
    let hostingView = NSHostingView(rootView: view)
    hostingView.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
    
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered,
                          defer: false)
    window.contentView = hostingView
    window.makeKeyAndOrderFront(nil)
    
    // Trigger screenshot
    let desktopURL = URL(fileURLWithPath: "/Users/alexeiboklag/projects/mlx/test_screenshot.png")
    state.captureScreenshotURL = desktopURL
    
    // Wait for render loop
    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
        print("Screenshot should be saved to \(desktopURL.path)")
        NSApplication.shared.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)

DispatchQueue.main.async {
    runUITest()
}

app.run()
