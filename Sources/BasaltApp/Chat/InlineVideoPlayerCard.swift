import AVFoundation
import AVKit
import AppKit
import BasaltAppCore
import BasaltCore
import SwiftUI
import UniformTypeIdentifiers

public struct GeneratedVideoPart: Sendable, Codable, Equatable, Identifiable {
    public var id: String { assetID }
    public var assetID: String
    public var prompt: String
    public var model: String
    public var filePath: String?
    public var posterPNG: Data?
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Int
    public var durationSeconds: Double
    public var hasAudio: Bool
    public var executionMode: VideoExecutionMode
    public var currentPhase: VideoExecutionPhase?
    public var currentStep: Int
    public var totalSteps: Int
    public var isGenerating: Bool

    public init(
        assetID: String,
        prompt: String,
        model: String,
        dimensions: VideoDimensions = .defaultWidescreen,
        temporal: VideoTemporalSpec = .init(),
        executionMode: VideoExecutionMode = .strictPhasedTeardown
    ) {
        self.assetID = assetID
        self.prompt = prompt
        self.model = model
        self.filePath = nil
        self.posterPNG = nil
        self.width = dimensions.width
        self.height = dimensions.height
        self.frames = temporal.frames
        self.fps = temporal.fps
        self.durationSeconds = Double(temporal.frames) / Double(max(1, temporal.fps))
        self.hasAudio = true
        self.executionMode = executionMode
        self.currentPhase = nil
        self.currentStep = temporal.steps
        self.totalSteps = temporal.steps
        self.isGenerating = false
    }
}

public struct InlineVideoPlayerCard: View {
    public let part: GeneratedVideoPart
    public var onExport: ((GeneratedVideoPart) -> Void)?

    @State private var player: AVPlayer?
    @State private var isPlaying = false
    @State private var isMuted = false
    @State private var playheadSeconds: Double = 0.0

    public init(
        part: GeneratedVideoPart,
        initialPlayheadSeconds: Double = 0.0,
        onExport: ((GeneratedVideoPart) -> Void)? = nil
    ) {
        self.part = part
        self._playheadSeconds = State(initialValue: max(0.0, min(initialPlayheadSeconds, part.durationSeconds)))
        self.onExport = onExport
    }

    private var aspectRatio: CGFloat {
        CGFloat(max(part.width, 1)) / CGFloat(max(part.height, 1))
    }

    private var posterImage: NSImage? {
        InlineVideoIO.loadPosterImage(part: part)
    }

    private var currentFrameIndex: Int {
        guard part.durationSeconds > 0 else { return 1 }
        let ratio = min(1.0, max(0.0, playheadSeconds / part.durationSeconds))
        return min(part.frames, max(1, Int((ratio * Double(part.frames - 1)).rounded()) + 1))
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            viewportSurface
            if !part.isGenerating {
                timelineBar
                actionBar
            }
        }
        .frame(maxWidth: 580)
        .contextMenu { contextMenuItems }
        .onDisappear {
            player?.pause()
            isPlaying = false
        }
    }

    private var viewportSurface: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
            if part.isGenerating {
                PhasedVideoGeneratingOverlay(part: part, previewImage: posterImage)
            } else if isPlaying, let player {
                VideoPlayer(player: player)
            } else {
                stillFrameOverlay
            }
        }
        .aspectRatio(aspectRatio, contentMode: .fit)
        .frame(maxWidth: 580, maxHeight: 340)
        .clipShape(.rect(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.primary.opacity(0.14), lineWidth: 0.8)
        }
    }

    private var stillFrameOverlay: some View {
        ZStack {
            if let posterImage {
                Image(nsImage: posterImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
            } else {
                LinearGradient(
                    colors: [
                        Color(red: 0.09, green: 0.14, blue: 0.24),
                        Color(red: 0.18, green: 0.12, blue: 0.28),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }

            VStack {
                HStack(spacing: 6) {
                    if part.executionMode == .strictPhasedTeardown {
                        Label("Phased UMA", systemImage: "memorychip")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .adaptiveGlass(.chromeRegular, in: .capsule)
                    }
                    Spacer()
                    Text(verbatim: "\(part.width)×\(part.height) · \(part.fps) fps")
                        .font(.caption2.monospacedDigit().weight(.medium))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .adaptiveGlass(.chromeRegular, in: .capsule)
                }
                .padding(10)
                Spacer()
            }

            Button {
                togglePlayback()
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.title3.weight(.bold))
                    .padding(14)
                    .adaptiveGlass(.chromeInteractive, in: .circle)
            }
            .buttonStyle(.plain)
            .help(isPlaying ? Text("Pause") : Text("Play"))
        }
    }

    private var timelineBar: some View {
        HStack(spacing: 8) {
            Button {
                togglePlayback()
            } label: {
                Label(isPlaying ? "Pause" : "Play", systemImage: isPlaying ? "pause.fill" : "play.fill")
                    .labelStyle(.iconOnly)
                    .font(.caption.weight(.semibold))
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)

            Slider(
                value: Binding(
                    get: { playheadSeconds },
                    set: { seek(to: $0) }
                ),
                in: 0.0 ... max(part.durationSeconds, 0.1)
            ) {
                Text("Timeline")
            }
            .controlSize(.small)

            Text(verbatim: Self.formatTimecode(playheadSeconds, total: part.durationSeconds, frame: currentFrameIndex, totalFrames: part.frames))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.primary.opacity(0.05), in: .rect(cornerRadius: 10))
    }

    private var actionBar: some View {
        HStack(spacing: 6) {
            pillButton(
                isPlaying ? "Pause" : "Play",
                systemImage: isPlaying ? "pause.fill" : "play.fill"
            ) {
                togglePlayback()
            }
            if part.hasAudio {
                pillButton(
                    isMuted ? "Muted" : "Audio",
                    systemImage: isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill"
                ) {
                    isMuted.toggle()
                    player?.isMuted = isMuted
                }
            }
            pillButton("Copy Video", systemImage: "doc.on.doc") {
                InlineVideoIO.copyVideoToPasteboard(part: part)
            }
            pillButton("Export MP4…", systemImage: "square.and.arrow.down", tint: .accentColor) {
                exportCurrentVideo()
            }
            Spacer(minLength: 4)
            Text(verbatim: part.model)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        Button(isPlaying ? "Pause" : "Play", systemImage: isPlaying ? "pause.fill" : "play.fill") {
            togglePlayback()
        }
        Button("Copy Video", systemImage: "doc.on.doc") {
            InlineVideoIO.copyVideoToPasteboard(part: part)
        }
        Button("Export MP4…", systemImage: "square.and.arrow.down") {
            exportCurrentVideo()
        }
    }
}

extension InlineVideoPlayerCard {
    private func exportCurrentVideo() {
        if let onExport {
            onExport(part)
        } else {
            InlineVideoIO.exportMP4WithPanel(part: part)
        }
    }
    private func pillButton(
        _ title: LocalizedStringKey,
        systemImage: String,
        tint: Color? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.caption.weight(.medium))
                .foregroundStyle(tint ?? .primary)
                .lineLimit(1)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background((tint ?? .primary).opacity(tint == nil ? 0.08 : 0.18), in: .capsule)
                .overlay {
                    Capsule().strokeBorder((tint ?? .primary).opacity(tint == nil ? 0.16 : 0.45), lineWidth: 0.7)
                }
        }
        .buttonStyle(.plain)
        .fixedSize(horizontal: true, vertical: false)
    }

    private func ensurePlayer() -> AVPlayer? {
        if let player { return player }
        guard let path = part.filePath, FileManager.default.fileExists(atPath: path) else { return nil }
        let created = AVPlayer(url: URL(fileURLWithPath: path))
        created.isMuted = isMuted
        player = created
        return created
    }

    private func togglePlayback() {
        guard let active = ensurePlayer() else {
            isPlaying.toggle()
            return
        }
        if isPlaying {
            active.pause()
            isPlaying = false
        } else {
            active.play()
            isPlaying = true
        }
    }

    private func seek(to seconds: Double) {
        let clamped = max(0.0, min(seconds, max(part.durationSeconds, 0.1)))
        playheadSeconds = clamped
        if let active = ensurePlayer() {
            let target = CMTime(seconds: clamped, preferredTimescale: 600)
            active.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        }
    }

    static func formatTimecode(_ current: Double, total: Double, frame: Int, totalFrames: Int) -> String {
        let curStr = String(format: "%02d:%04.1f", Int(current) / 60, current.truncatingRemainder(dividingBy: 60))
        let totStr = String(format: "%02d:%04.1f", Int(total) / 60, total.truncatingRemainder(dividingBy: 60))
        return "\(curStr) / \(totStr) · F\(frame)/\(totalFrames)"
    }
}

struct PhasedVideoGeneratingOverlay: View {
    let part: GeneratedVideoPart
    let previewImage: NSImage?

    private var phaseTitleKey: LocalizedStringKey {
        switch part.currentPhase ?? .denoising {
        case .textConditioning: "Phase 1 · Text Conditioning"
        case .denoising: "Phase 2 · DiT Denoising"
        case .vaeAndAudioMuxing: "Phase 3 · 3D-VAE & MP4 Muxing"
        }
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            if let previewImage {
                Image(nsImage: previewImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
                    .blur(radius: 2.5)
            } else {
                LinearGradient(
                    colors: [
                        Color(red: 0.10, green: 0.15, blue: 0.26),
                        Color(red: 0.22, green: 0.11, blue: 0.29),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }

            VStack(spacing: 6) {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.mini)
                        .accessibilityLabel(Text("Generating video…"))
                    Text(phaseTitleKey)
                        .font(.caption.weight(.semibold))
                    Text(verbatim: "(\(part.currentStep)/\(max(1, part.totalSteps)))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .adaptiveGlass(.chromeRegular, in: .capsule)

                ProgressView(
                    value: Double(min(max(part.currentStep, 0), max(part.totalSteps, 1))),
                    total: Double(max(part.totalSteps, 1))
                )
                .progressViewStyle(.linear)
                .frame(maxWidth: 280)
            }
            .padding(14)
        }
    }
}

@MainActor
enum InlineVideoIO {
    static func loadPosterImage(part: GeneratedVideoPart) -> NSImage? {
        if let png = part.posterPNG, !png.isEmpty, let img = NSImage(data: png) {
            return img
        }
        return nil
    }

    static func copyVideoToPasteboard(part: GeneratedVideoPart) {
        guard let path = part.filePath, FileManager.default.fileExists(atPath: path) else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.writeObjects([NSURL(fileURLWithPath: path)])
    }

    static func exportMP4WithPanel(part: GeneratedVideoPart) {
        guard let path = part.filePath,
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = part.assetID.isEmpty ? "basalt-video.mp4" : part.assetID
        if panel.runModal() == .OK, let outURL = panel.url {
            try? data.write(to: outURL, options: .atomic)
        }
    }
}
