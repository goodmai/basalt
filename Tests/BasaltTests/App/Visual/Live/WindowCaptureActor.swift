import CoreGraphics
import Foundation
import ScreenCaptureKit

public actor WindowCaptureActor {
    public init() {}

    public func captureWindow(windowID: CGWindowID, scale: Int = 2) async throws(VisualTestError) -> RenderedBitmap {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        } catch {
            throw .screenCaptureUnavailable(error.localizedDescription)
        }
        guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
            throw .windowNotFound(title: "\(windowID)")
        }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        let pixelScale = max(1, scale)
        config.width = max(1, Int(window.frame.width)) * pixelScale
        config.height = max(1, Int(window.frame.height)) * pixelScale
        config.showsCursor = false
        config.capturesShadowsOnly = false
        let cgImage: CGImage
        do {
            cgImage = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        } catch {
            throw .screenCaptureUnavailable(error.localizedDescription)
        }
        return try BitmapCodec.fromCGImage(cgImage)
    }
}
