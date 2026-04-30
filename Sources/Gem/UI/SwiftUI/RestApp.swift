import SwiftUI
import AppKit

@available(macOS 14.0, *)
@MainActor
public func launchRESTUI(baseURL: String) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    
    let delegate = RestAppDelegate(baseURL: baseURL)
    // Keep a strong reference if necessary, though NSApp retains it temporarily during run
    app.delegate = delegate
    app.run()
}

@available(macOS 14.0, *)
class RestAppDelegate: NSObject, NSApplicationDelegate {
    let baseURL: String
    var window: NSWindow!
    
    init(baseURL: String) {
        self.baseURL = baseURL
    }
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        let appState = AppState(baseURL: baseURL)
        let contentView = ContentView()
            .environment(appState)
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 700),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.center()
        window.setFrameAutosaveName("RestChatWindow")
        window.contentView = NSHostingView(rootView: contentView)
        window.makeKeyAndOrderFront(nil)
        window.title = "Gemm REST Chat"
        
        NSApp.activate(ignoringOtherApps: true)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@available(macOS 14.0, *)
@MainActor
@Observable
class AppState {
    let baseURL: String
    var token: String?
    var isAuthenticated: Bool { token != nil }
    var restClient: RESTClient
    
    init(baseURL: String) {
        self.baseURL = baseURL
        self.restClient = RESTClient(baseURL: baseURL)
    }
    
    func setToken(_ newToken: String) {
        self.token = newToken
        Task {
            await self.restClient.setToken(newToken)
        }
    }
    
    func logout() {
        self.token = nil
        Task {
            await self.restClient.setToken(nil)
        }
    }
}

@available(macOS 14.0, *)
struct ContentView: View {
    @Environment(AppState.self) private var appState
    
    var body: some View {
        Group {
            if appState.isAuthenticated {
                ChatView()
            } else {
                LoginView()
            }
        }
        .frame(minWidth: 600, minHeight: 500)
    }
}
