import AppKit
import BasaltApp
import SwiftUI

final class QuickAskPanel: NSPanel {
    static let frame = NSRect(x: 0, y: 0, width: 640, height: 80)
    static let style: NSWindow.StyleMask = [.nonactivatingPanel, .titled, .fullSizeContentView]

    init() {
        super.init(contentRect: Self.frame, styleMask: Self.style, backing: .buffered, defer: true)
        (isFloatingPanel, level, hidesOnDeactivate, isReleasedWhenClosed) = (true, .floating, false, false)
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        (titleVisibility, titlebarAppearsTransparent, isMovableByWindowBackground) = (.hidden, true, true)
        (backgroundColor, hasShadow) = (.clear, true)
        standardWindowButton(.closeButton)?.isHidden = true
        standardWindowButton(.miniaturizeButton)?.isHidden = true
        standardWindowButton(.zoomButton)?.isHidden = true
    }

    override var canBecomeKey: Bool { true }

    override func cancelOperation(_ sender: Any?) { close() }
}

@MainActor
final class QuickAskController: NSObject, NSWindowDelegate {
    let model: AppModel
    let openConversation: @MainActor () -> Void
    private var panel: QuickAskPanel?

    init(model: AppModel, openConversation: @escaping @MainActor () -> Void) { (self.model, self.openConversation) = (model, openConversation) }

    func toggle() {
        if let panel, panel.isVisible { return panel.close() }
        let panel = QuickAskPanel()
        panel.delegate = self
        panel.contentView = NSHostingView(rootView: QuickAskView(store: model.openQuickAsk()) { [weak self] in self?.promote() })
        panel.center()
        panel.setFrameOrigin(NSPoint(x: panel.frame.minX, y: (NSScreen.main?.visibleFrame.maxY ?? 800) * 0.72))
        panel.makeKeyAndOrderFront(nil)
        self.panel = panel
    }

    func promote() {
        Task {
            await model.promoteQuickAsk()
            panel?.close()
            openConversation()
        }
    }

    func windowWillClose(_ notification: Notification) {
        model.dismissQuickAsk()
        panel = nil
    }
}
