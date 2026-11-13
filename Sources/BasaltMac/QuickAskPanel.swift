import AppKit
import BasaltApp
import SwiftUI

final class QuickAskPanel: NSPanel {
    static let frame = NSRect(x: 0, y: 0, width: 664, height: 92)
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

    func resizeKeepingTopEdge(to targetHeight: CGFloat) {
        let clamped = min(max(targetHeight, Self.frame.height), 460)
        guard abs(frame.height - clamped) > 1 else { return }
        let topY = frame.maxY
        let next = NSRect(x: frame.minX, y: topY - clamped, width: Self.frame.width, height: clamped)
        let animate = !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        setFrame(next, display: true, animate: animate)
    }
}

@MainActor
final class QuickAskController: NSObject, NSWindowDelegate {
    let model: AppModel
    let openConversation: @MainActor () -> Void
    private var panel: QuickAskPanel?

    init(model: AppModel, openConversation: @escaping @MainActor () -> Void) {
        (self.model, self.openConversation) = (model, openConversation)
    }

    func toggle() {
        if let panel, panel.isVisible { return panel.close() }
        let panel = QuickAskPanel()
        panel.delegate = self
        let store = model.openQuickAsk()
        let host = NSHostingView(rootView: QuickAskView(store: store) { [weak self] in self?.promote() })
        host.sizingOptions = [.intrinsicContentSize]
        panel.contentView = host
        panel.center()
        let topAnchor = (NSScreen.main?.visibleFrame.maxY ?? 800) * 0.78
        panel.setFrameTopLeftPoint(NSPoint(x: panel.frame.minX, y: topAnchor))
        panel.resizeKeepingTopEdge(to: host.fittingSize.height)
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
