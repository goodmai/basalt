import Foundation
import ScreenCaptureKit
import Vision
import CoreGraphics
import XCTest

@MainActor
@available(macOS 12.3, *)
public struct VisualValidator {
    
    public enum AutomationError: Error {
        case windowNotFound
        case captureFailed
    }
    
    public static func captureWindow(bundleIdentifier: String = "com.apple.dt.xctest.tool") async throws -> CGImage {
        // SCShareableContent gets all shareable windows
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        
        // Find the window belonging to the app under test. 
        // Note: For SPM CLI tests, the app might run under xctest or a generic process, so we can fallback to the active window or search by title.
        guard let window = content.windows.first(where: { 
            $0.owningApplication?.bundleIdentifier == bundleIdentifier || 
            $0.owningApplication?.applicationName.contains("Gemm") == true ||
            $0.title?.contains("Gemm") == true
        }) ?? content.windows.first else {
            throw AutomationError.windowNotFound
        }
        
        let config = SCStreamConfiguration()
        config.width = Int(window.frame.width)
        config.height = Int(window.frame.height)
        
        let image = try await SCScreenshotManager.captureImage(contentFilter: SCContentFilter(desktopIndependentWindow: window), configuration: config)
        return image
    }
    
    public static func validateTextPresence(in image: CGImage, expectedText: String) async -> Bool {
        return await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                guard let observations = request.results as? [VNRecognizedTextObservation] else {
                    continuation.resume(returning: false)
                    return
                }
                
                let foundTexts = observations.compactMap { $0.topCandidates(1).first?.string }
                let contains = foundTexts.contains(where: { $0.localizedCaseInsensitiveContains(expectedText) })
                continuation.resume(returning: contains)
            }
            
            request.recognitionLevel = .accurate
            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(returning: false)
            }
        }
    }
}
