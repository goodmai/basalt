import AppKit
import BasaltAppCore
import SwiftUI
import UniformTypeIdentifiers

public struct GeneratedImageCard: View {
    public let part: GeneratedImagePart
    public var onEdit: (GeneratedImagePart) -> Void = { _ in }

    private let isStreaming: Bool
    @State private var showCheckerboard = true
    @State private var showBoxes = true
    @State private var splitRatio: Double = 0.52
    @State private var showingLightbox = false

    public init(part: GeneratedImagePart, isStreaming: Bool = true, onEdit: @escaping (GeneratedImagePart) -> Void = { _ in }) {
        self.part = part
        self.isStreaming = isStreaming
        self.onEdit = onEdit
    }

    private var activeImage: NSImage? {
        GeneratedImageIO.loadImage(filePath: part.filePath, fallbackPNG: part.previewPNG)
    }

    private var beforeImage: NSImage? {
        part.beforeFilePath.flatMap { GeneratedImageIO.loadImage(filePath: $0, fallbackPNG: nil) }
    }

    private var aspectRatio: CGFloat {
        CGFloat(max(part.width, 1)) / CGFloat(max(part.height, 1))
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            canvasSurface
            if !part.isStreamingPreview {
                actionBar
            }
        }
        .contextMenu { contextMenuItems }
        .sheet(isPresented: $showingLightbox) {
            ImageLightboxSheet(
                part: part,
                image: activeImage,
                showsCheckerboard: part.hasAlpha && showCheckerboard
            )
        }
    }

    private var canvasSurface: some View {
        ZStack {
            backdropLayer
            mediaLayer
            if !part.boxes.isEmpty, showBoxes, !part.isStreamingPreview {
                ImageBoundingBoxOverlay(
                    boxes: part.boxes,
                    imageWidth: part.width,
                    imageHeight: part.height
                )
                .allowsHitTesting(false)
            }
        }
        .aspectRatio(aspectRatio, contentMode: .fit)
        .frame(maxWidth: 560, maxHeight: 336)
        .clipShape(.rect(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.primary.opacity(0.14), lineWidth: 0.8)
        }
    }

    @ViewBuilder
    private var backdropLayer: some View {
        if part.hasAlpha, showCheckerboard {
            ImageCheckerboardBackdrop()
        } else {
            Color(nsColor: .underPageBackgroundColor)
        }
    }

    @ViewBuilder
    private var mediaLayer: some View {
        if isStreaming && part.isStreamingPreview {
            DenoisingPreviewOverlay(part: part, previewImage: activeImage)
        } else if let beforeImage, let activeImage {
            BeforeAfterComparisonOverlay(
                beforeImage: beforeImage,
                afterImage: activeImage,
                splitRatio: $splitRatio
            )
        } else if let activeImage {
            Image(nsImage: activeImage)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
        } else {
            ProgressView()
                .controlSize(.regular)
                .accessibilityLabel("Generating image…")
        }
    }

    private var actionBar: some View {
        HStack(spacing: 6) {
            toolbarButton("Zoom", systemImage: "arrow.up.left.and.arrow.down.right") {
                showingLightbox = true
            }
            toolbarButton("Copy Image", systemImage: "doc.on.doc", iconOnly: true) {
                GeneratedImageIO.copyToPasteboard(part: part)
            }
            toolbarButton("Save…", systemImage: "square.and.arrow.down") {
                GeneratedImageIO.saveWithPanel(part: part)
            }
            toolbarButton("Edit Image", systemImage: "wand.and.stars", tint: .accentColor) {
                onEdit(part)
            }
            if part.hasAlpha {
                toolbarButton(
                    showCheckerboard ? "Checkerboard" : "Solid",
                    systemImage: showCheckerboard ? "square.grid.2x2" : "square.fill",
                    iconOnly: true
                ) {
                    showCheckerboard.toggle()
                }
            }
            if !part.boxes.isEmpty {
                toolbarButton("BBox", systemImage: "viewfinder", tint: showBoxes ? .cyan : nil) {
                    showBoxes.toggle()
                }
            }
            Spacer(minLength: 4)
            Text(verbatim: "\(part.width)×\(part.height)" + (part.hasAlpha ? " · RGBA" : ""))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
        }
        .frame(maxWidth: 560)
    }

    private func toolbarButton(
        _ title: LocalizedStringKey,
        systemImage: String,
        iconOnly: Bool = false,
        tint: Color? = nil,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Group {
                if iconOnly {
                    Label(title, systemImage: systemImage).labelStyle(.iconOnly)
                } else {
                    Label(title, systemImage: systemImage).labelStyle(.titleAndIcon)
                }
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(tint ?? .primary)
            .lineLimit(1)
            .padding(.horizontal, iconOnly ? 8 : 10)
            .padding(.vertical, 5)
            .background((tint ?? .primary).opacity(tint == nil ? 0.08 : 0.18), in: .capsule)
            .overlay {
                Capsule().strokeBorder((tint ?? .primary).opacity(tint == nil ? 0.16 : 0.45), lineWidth: 0.7)
            }
        }
        .buttonStyle(.plain)
        .fixedSize(horizontal: true, vertical: false)
        .help(Text(title))
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        Button("Zoom", systemImage: "arrow.up.left.and.arrow.down.right") {
            showingLightbox = true
        }
        Button("Copy Image", systemImage: "doc.on.doc") {
            GeneratedImageIO.copyToPasteboard(part: part)
        }
        Button("Save…", systemImage: "square.and.arrow.down") {
            GeneratedImageIO.saveWithPanel(part: part)
        }
        if !part.isStreamingPreview {
            Button("Edit Image", systemImage: "wand.and.stars") {
                onEdit(part)
            }
        }
    }
}

struct DenoisingPreviewOverlay: View {
    let part: GeneratedImagePart
    let previewImage: NSImage?

    var body: some View {
        ZStack(alignment: .bottom) {
            if let previewImage {
                Image(nsImage: previewImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
                    .blur(radius: 2.5)
                    .scaleEffect(1.02)
            } else {
                LinearGradient(
                    colors: [
                        Color(red: 0.11, green: 0.15, blue: 0.25),
                        Color(red: 0.20, green: 0.12, blue: 0.28),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }

            VStack(spacing: 6) {
                HStack(spacing: 6) {
                    ProgressView()
                        .controlSize(.mini)
                    Text("Step \(part.currentStep)/\(part.totalSteps) · \(part.model)")
                        .font(.caption.monospacedDigit().weight(.medium))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .adaptiveGlass(.chromeRegular, in: .capsule)

                ProgressView(
                    value: Double(min(max(part.currentStep, 0), max(part.totalSteps, 1))),
                    total: Double(max(part.totalSteps, 1))
                )
                .progressViewStyle(.linear)
                .frame(maxWidth: 260)
            }
            .padding(14)
        }
    }
}

public struct ImageCheckerboardBackdrop: View {
    public var tileSize: CGFloat = 12
    @Environment(\.colorScheme) private var colorScheme

    public init(tileSize: CGFloat = 12) {
        self.tileSize = tileSize
    }

    public var body: some View {
        let primary = colorScheme == .dark ? Color(white: 0.15) : Color(white: 0.94)
        let alternate = colorScheme == .dark ? Color(white: 0.22) : Color(white: 0.86)
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(primary))
            let step = max(tileSize, 4)
            let cols = Int(ceil(size.width / step))
            let rows = Int(ceil(size.height / step))
            var squares = Path()
            for row in 0 ..< rows {
                for col in 0 ..< cols where (row + col).isMultiple(of: 2) {
                    let rect = CGRect(x: CGFloat(col) * step, y: CGFloat(row) * step, width: step, height: step)
                    squares.addRect(rect)
                }
            }
            context.fill(squares, with: .color(alternate))
        }
    }
}

public struct BeforeAfterComparisonOverlay: View {
    public let beforeImage: NSImage
    public let afterImage: NSImage
    @Binding public var splitRatio: Double

    public init(beforeImage: NSImage, afterImage: NSImage, splitRatio: Binding<Double>) {
        self.beforeImage = beforeImage
        self.afterImage = afterImage
        self._splitRatio = splitRatio
    }

    public var body: some View {
        GeometryReader { geo in
            let clamped = min(max(splitRatio, 0.05), 0.95)
            let splitX = geo.size.width * clamped
            ZStack(alignment: .topLeading) {
                Image(nsImage: afterImage)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: geo.size.width, height: geo.size.height)

                Image(nsImage: beforeImage)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .mask(alignment: .leading) {
                        Rectangle().frame(width: splitX)
                    }

                Rectangle()
                    .fill(Color.white.opacity(0.92))
                    .frame(width: 2, height: geo.size.height)
                    .shadow(color: .black.opacity(0.45), radius: 3)
                    .position(x: splitX, y: geo.size.height * 0.5)

                Image(systemName: "arrow.left.and.right")
                    .font(.caption2.weight(.bold))
                    .padding(7)
                    .adaptiveGlass(.chromeInteractive, in: .circle)
                    .position(x: splitX, y: geo.size.height * 0.5)

                HStack {
                    Text("Before")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .adaptiveGlass(.chromeRegular, in: .capsule)
                    Spacer()
                    Text("After")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .adaptiveGlass(.chromeRegular, in: .capsule)
                }
                .padding(10)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let raw = Double(value.location.x / max(geo.size.width, 1))
                        splitRatio = min(max(raw, 0.05), 0.95)
                    }
            )
        }
    }
}

struct ImageBoundingBoxOverlay: View {
    let boxes: [ChatImageBox]
    let imageWidth: Int
    let imageHeight: Int

    var body: some View {
        GeometryReader { geo in
            ForEach(Array(boxes.enumerated()), id: \.offset) { _, box in
                let rect = boxRect(box, in: geo.size)
                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.cyan.opacity(0.08))
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(Color.cyan, lineWidth: 1.75)
                        }
                    Text(verbatim: box.label)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(Color.black.opacity(0.9))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.cyan, in: .capsule)
                        .padding(4)
                }
                .frame(width: max(rect.width, 24), height: max(rect.height, 20))
                .position(x: rect.midX, y: rect.midY)
            }
        }
    }

    private func boxRect(_ box: ChatImageBox, in canvas: CGSize) -> CGRect {
        let isNormalized = box.x <= 1.0 && box.y <= 1.0 && box.width <= 1.0 && box.height <= 1.0
        let normX = isNormalized ? box.x : box.x / Double(max(imageWidth, 1))
        let normY = isNormalized ? box.y : box.y / Double(max(imageHeight, 1))
        let normW = isNormalized ? box.width : box.width / Double(max(imageWidth, 1))
        let normH = isNormalized ? box.height : box.height / Double(max(imageHeight, 1))
        return CGRect(
            x: CGFloat(normX) * canvas.width,
            y: CGFloat(normY) * canvas.height,
            width: CGFloat(normW) * canvas.width,
            height: CGFloat(normH) * canvas.height
        )
    }
}

public struct ImageLightboxSheet: View {
    public let part: GeneratedImagePart
    public let image: NSImage?
    public var showsCheckerboard = false
    @Environment(\.dismiss) private var dismiss

    public init(part: GeneratedImagePart, image: NSImage?, showsCheckerboard: Bool = false) {
        self.part = part
        self.image = image
        self.showsCheckerboard = showsCheckerboard
    }

    public var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(verbatim: part.prompt)
                        .font(.headline)
                        .lineLimit(1)
                    Text(verbatim: "\(part.width)×\(part.height) · \(part.model)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Copy Image", systemImage: "doc.on.doc") {
                    GeneratedImageIO.copyToPasteboard(part: part)
                }
                Button("Save…", systemImage: "square.and.arrow.down") {
                    GeneratedImageIO.saveWithPanel(part: part)
                }
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)

            ZStack {
                if showsCheckerboard {
                    ImageCheckerboardBackdrop(tileSize: 16)
                } else {
                    Color(nsColor: .underPageBackgroundColor)
                }
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .scaledToFit()
                        .padding(12)
                }
            }
            .clipShape(.rect(cornerRadius: 14))
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
        .frame(minWidth: 720, minHeight: 560)
    }
}

@MainActor
enum GeneratedImageIO {
    private static var imageCache: [String: NSImage] = [:]

    static func loadImage(filePath: String?, fallbackPNG: Data?) -> NSImage? {
        if let filePath, !filePath.isEmpty {
            if let cached = imageCache[filePath] { return cached }
            if let image = NSImage(contentsOfFile: filePath) {
                if imageCache.count >= 24 { imageCache.removeAll(keepingCapacity: true) }
                imageCache[filePath] = image
                return image
            }
        }
        if let fallbackPNG, !fallbackPNG.isEmpty {
            return NSImage(data: fallbackPNG)
        }
        return nil
    }

    static func rawPNGData(for part: GeneratedImagePart) -> Data? {
        if let path = part.filePath, let data = try? Data(contentsOf: URL(fileURLWithPath: path)) {
            return data
        }
        return part.previewPNG
    }

    static func copyToPasteboard(part: GeneratedImagePart) {
        let pb = NSPasteboard.general
        pb.clearContents()
        if let data = rawPNGData(for: part) {
            pb.setData(data, forType: .png)
        } else if let image = loadImage(filePath: part.filePath, fallbackPNG: part.previewPNG) {
            pb.writeObjects([image])
        }
    }

    static func saveWithPanel(part: GeneratedImagePart) {
        guard let data = rawPNGData(for: part) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = part.assetID.isEmpty ? "basalt-image.png" : part.assetID
        if panel.runModal() == .OK, let url = panel.url {
            try? data.write(to: url, options: .atomic)
        }
    }
}
