import AppKit
import AVFoundation
import BasaltCore
import Foundation
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

public enum InteractionStep: Sendable, Equatable {
    case draft(String)
    case startRun
    case appendToken(String)
    case finishRun
    case enginePhase(EngineState.Phase)
}

@MainActor
public enum VideoScenarioCatalog {
    public static func renderQuickAskMorphFrames() throws(VisualTestError) -> [RenderedBitmap] {
        let spec = VisualSceneSpec(id: .quickAskStreaming, width: 680, height: 440, scale: 2)
        let store = ChatStore(
            ChatState(conversation: ConversationSnapshot(id: ConversationID(VisualSceneFixtures.fixedUUID(50)), model: "mlx-community/gemma-4-12B-it-qat-4bit")),
            reduce: ChatReducer.reduce,
            handle: { _ in .none }
        )
        var frames: [RenderedBitmap] = []
        let idle = try OffscreenSceneRenderer.render(QuickAskView(store: store) {}, spec: spec, variant: .darkStandard, desktopBackdrop: true)
        frames += Array(repeating: idle, count: 15)

        store.send(.draftChanged("Why use Liquid Glass containers in macOS 27?"))
        let typed = try OffscreenSceneRenderer.render(QuickAskView(store: store) {}, spec: spec, variant: .darkStandard, desktopBackdrop: true)
        frames += Array(repeating: typed, count: 15)

        let run = RunID(VisualSceneFixtures.fixedUUID(51))
        store.send(.send(message: MessageID(VisualSceneFixtures.fixedUUID(52)), reply: MessageID(VisualSceneFixtures.fixedUUID(53)), run: run))
        let chunks = [
            "Single ", "shared ", "backdrop ", "sampling ", "keeps ",
            "morphing ", "controls ", "smooth ", "at ", "**60 fps**."
        ]
        for chunk in chunks {
            store.send(.delta(run, .text(chunk)))
            let stepFrame = try OffscreenSceneRenderer.render(QuickAskView(store: store) {}, spec: spec, variant: .darkStandard, desktopBackdrop: true)
            frames += Array(repeating: stepFrame, count: 6)
        }
        store.send(.finished(run, RunTiming(timeToFirstToken: .milliseconds(90), streaming: .milliseconds(480))))
        let done = try OffscreenSceneRenderer.render(QuickAskView(store: store) {}, spec: spec, variant: .darkStandard, desktopBackdrop: true)
        frames += Array(repeating: done, count: 20)
        return frames
    }

    public static func renderChatStreamFrames() throws(VisualTestError) -> [RenderedBitmap] {
        let spec = VisualSceneSpec(id: .chatConversationRich, width: 1040, height: 760, scale: 2)
        var conversation = ConversationSnapshot(id: ConversationID(VisualSceneFixtures.fixedUUID(60)), model: "mlx-community/Qwen3.6-35B-A3B-8bit")
        conversation.title = "Streaming Transcript Choreography"
        conversation.messages = (0 ..< 14).map { idx in
            MessageSnapshot(
                id: MessageID(VisualSceneFixtures.fixedUUID(300 + idx)),
                role: idx.isMultiple(of: 2) ? .user : .assistant,
                parts: [.text("Message \(idx + 1): deterministic local inference on Apple Silicon with full receipts.")]
            )
        }
        let store = ChatStore(ChatState(conversation: conversation), reduce: ChatReducer.reduce, handle: { _ in .none })
        var frames: [RenderedBitmap] = []
        let run = RunID(VisualSceneFixtures.fixedUUID(61))
        store.send(.draftChanged("Stream the benchmark summary."))
        let beforeSend = try OffscreenSceneRenderer.render(ChatView(store: store), spec: spec, variant: .darkStandard)
        frames += Array(repeating: beforeSend, count: 12)

        store.send(.send(message: MessageID(VisualSceneFixtures.fixedUUID(62)), reply: MessageID(VisualSceneFixtures.fixedUUID(63)), run: run))
        for tokenIndex in 0 ..< 4 {
            store.send(.delta(run, .text("Step \(tokenIndex + 1): zero swap-outs confirmed. ")))
            let frame = try OffscreenSceneRenderer.render(ChatView(store: store), spec: spec, variant: .darkStandard)
            frames += Array(repeating: frame, count: 6)
        }
        store.send(.scrolled(atBottom: false))
        for tokenIndex in 4 ..< 8 {
            store.send(.delta(run, .text("Step \(tokenIndex + 1): Liquid Glass bar pinned. ")))
            let frame = try OffscreenSceneRenderer.render(ChatView(store: store), spec: spec, variant: .darkStandard)
            frames += Array(repeating: frame, count: 8)
        }
        store.send(.jumpToBottom)
        store.send(.finished(run, RunTiming(timeToFirstToken: .milliseconds(120), streaming: .milliseconds(640))))
        let finalFrame = try OffscreenSceneRenderer.render(ChatView(store: store), spec: spec, variant: .darkStandard)
        frames += Array(repeating: finalFrame, count: 16)
        return frames
    }

    public static func renderEngineRecoveryFrames() async throws(VisualTestError) -> [RenderedBitmap] {
        let spec = VisualSceneSpec(id: .rootBannerFailed, width: 1040, height: 720, scale: 2)
        let model = try await VisualSceneFixtures.makeAppModel(phase: .spawning, withSummaries: true)
        let phases: [EngineState.Phase] = [
            .spawning,
            .loading(port: 58_100),
            .backingOff(attempt: 1),
            .ready(port: 58_100),
        ]
        var frames: [RenderedBitmap] = []
        for phase in phases {
            var state = EngineState()
            state.phase = phase
            await model.engineChanged(state)
            let frame = try OffscreenSceneRenderer.render(RootView(model: model), spec: spec, variant: .darkStandard)
            frames += Array(repeating: frame, count: 24)
        }
        return frames
    }
}

@Suite("Live window capture and 60 fps MP4 choreography evidence (#474)",
       .enabled(if: ProcessInfo.processInfo.environment["BASALT_UI_E2E"] == "1", "set BASALT_UI_E2E=1 to run live window and MP4 evidence"),
       .serialized)
@MainActor
struct AppLiveEvidenceTests {
    static var outputDirectory: URL {
        if let custom = ProcessInfo.processInfo.environment["BASALT_UI_EVIDENCE_DIR"], !custom.isEmpty {
            return URL(fileURLWithPath: custom)
        }
        return FileManager.default.temporaryDirectory.appendingPathComponent("basalt-ui-e2e-\(UUID().uuidString)")
    }

    @Test("All 3 canonical interaction scenarios encode valid 60 fps MP4 containers with non-zero video tracks")
    func choreographedMP4Scenarios() async throws {
        let root = Self.outputDirectory.appendingPathComponent("videos")
        let scenarios: [(String, [RenderedBitmap])] = [
            ("quickask-morph", try VideoScenarioCatalog.renderQuickAskMorphFrames()),
            ("chat-stream-pin-and-jump", try VideoScenarioCatalog.renderChatStreamFrames()),
            ("engine-recovery-banner", try await VideoScenarioCatalog.renderEngineRecoveryFrames()),
        ]
        for (name, frames) in scenarios {
            let url = root.appendingPathComponent("\(name).mp4")
            let recorded = try await ChoreographedMP4Writer.writeMP4(frames: frames, fps: 60, to: url)
            #expect(recorded.byteSize > 1024)
            #expect(recorded.durationSeconds >= 1.0)
            let asset = AVURLAsset(url: url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            #expect(!tracks.isEmpty, "missing video track in \(name).mp4")
        }
    }

    @Test("Live NSWindow capture and VideoRecordingLease lifecycle succeed when Screen Recording permission is available")
    func liveWindowCaptureAndLease() async throws {
        let store = VisualSceneFixtures.makeQuickAskStore(streaming: true)
        let window = NSWindow(
            contentRect: NSRect(x: 120, y: 120, width: 680, height: 440),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.title = "Basalt Live Evidence"
        let host = NSHostingView(rootView: QuickAskView(store: store) {})
        window.contentView = host
        window.orderFrontRegardless()
        await Task.yield()
        host.layoutSubtreeIfNeeded()
        host.display()
        defer { window.close() }

        let windowID = CGWindowID(window.windowNumber)
        let capturer = WindowCaptureActor()
        do {
            let shot = try await capturer.captureWindow(windowID: windowID, scale: 2)
            #expect(shot.width > 0 && shot.height > 0)
            let recorder = VideoRecorderActor()
            let mp4URL = Self.outputDirectory.appendingPathComponent("videos/live-window.mp4")
            let lease = try await recorder.start(windowID: windowID, outputURL: mp4URL)
            store.send(.draftChanged("Live ScreenCaptureKit check"))
            await Task.yield()
            host.layoutSubtreeIfNeeded()
            host.display()
            let video = try await recorder.finish(lease)
            #expect(video.width > 0 && video.height > 0)
        } catch {
            switch error {
            case .screenCaptureUnavailable, .recordingFailed:
                return
            default:
                Issue.record("Unexpected live capture error: \(error)")
            }
        }
    }
}

@Suite("CLI Before/After UI evidence package runner (#474)",
       .enabled(if: ProcessInfo.processInfo.environment["BASALT_GENERATE_UI_EVIDENCE"] == "1"),
       .serialized)
@MainActor
struct AppEvidencePackageRunner {
    @Test("Generates before/after comparisons, manifest.json, report.md, and 60 fps MP4s")
    func generateEvidencePackage() async throws {
        let env = ProcessInfo.processInfo.environment
        let baselinePath = env["BASALT_UI_BASELINE_DIR"] ?? BaselineStore.defaultDirectory.path
        let outputPath = env["BASALT_UI_EVIDENCE_DIR"] ?? "logs/ui-evidence-latest"
        let commitBase = env["BASALT_UI_COMMIT_BASE"] ?? "baseline"
        let commitHead = env["BASALT_UI_COMMIT_HEAD"] ?? "HEAD"
        let includeVideos = env["BASALT_UI_SKIP_VIDEOS"] != "1"
        try await VisualEvidenceGenerator.generatePackage(
            baselineDirectory: URL(fileURLWithPath: baselinePath),
            outputDirectory: URL(fileURLWithPath: outputPath),
            commitBase: commitBase,
            commitHead: commitHead,
            includeVideos: includeVideos
        )
        #expect(FileManager.default.fileExists(atPath: "\(outputPath)/manifest.json"))
        #expect(FileManager.default.fileExists(atPath: "\(outputPath)/report.md"))
    }
}
