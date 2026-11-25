import BasaltCore
import Foundation

public struct ChatAttachment: Sendable, Equatable, Identifiable {
    public static let byteLimit = 32 << 20
    static let kinds: [String: String] = [
        "pdf": "pdf", "png": "image", "jpg": "image", "jpeg": "image", "heic": "image", "tiff": "image", "gif": "image", "webp": "image",
        "wav": "audio", "mp3": "audio", "m4a": "audio", "aac": "audio", "flac": "audio", "caf": "audio",
    ]

    public let id: UUID
    public var name: String
    public var kind: String
    public var format: String
    public var data: Data
    public var sourcePath: String?

    public init?(name: String, data: Data, id: UUID = UUID(), sourcePath: String? = nil) {
        let format = (name as NSString).pathExtension.lowercased()
        guard let kind = Self.kinds[format], !data.isEmpty, data.count <= Self.byteLimit else { return nil }
        (self.id, self.name, self.kind, self.format, self.data, self.sourcePath) = (id, name, kind, format, data, sourcePath)
    }

    private init(id: UUID, name: String, kind: String, format: String, data: Data, sourcePath: String?) {
        (self.id, self.name, self.kind, self.format, self.data, self.sourcePath) = (id, name, kind, format, data, sourcePath)
    }

    public static func deferredReference(name: String, sourcePath: String, id: UUID = UUID()) -> ChatAttachment? {
        let format = (name as NSString).pathExtension.lowercased()
        guard let kind = Self.kinds[format], !sourcePath.isEmpty else { return nil }
        return ChatAttachment(id: id, name: name, kind: kind, format: format, data: Data(), sourcePath: sourcePath)
    }

    public static func accepts(_ name: String) -> Bool { kinds[(name as NSString).pathExtension.lowercased()] != nil }

    public static func load(from url: URL) -> ChatAttachment? {
        guard url.isFileURL, accepts(url.lastPathComponent) else { return nil }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let target = url.resolvingSymlinksInPath()
        guard let values = try? target.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let size = values.fileSize,
              size > 0, size <= byteLimit else { return nil }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return ChatAttachment(name: url.lastPathComponent, data: data, sourcePath: url.path)
    }

    @concurrent
    public static func loadAsync(from url: URL) async -> ChatAttachment? {
        load(from: url)
    }
}

public struct PipelineCall: Sendable, Equatable {
    public var attachments: [ChatAttachment]
    public var task: String

    public var body: Data {
        let attachments = attachments.map { ["kind": $0.kind, "format": $0.format, "data": $0.data.base64EncodedString()] }
        return (try? JSONSerialization.data(withJSONObject: ["attachments": attachments, "task": task])) ?? Data()
    }
}

public struct PipelineResult: Sendable, Equatable {
    public var perceived: [String]
    public var answer: String
    public var receipt: Receipt

    public init(perceived: [String], answer: String, receipt: Receipt) { (self.perceived, self.answer, self.receipt) = (perceived, answer, receipt) }

    public init(_ response: PipelineResponse, count: Int) {
        let pages = response.perceived.pages ?? []
        var perceived = (0 ..< count).map { index in pages.filter { $0.attachment == index }.map(\.markdown).joined(separator: "\n\n") }
        if let transcript = response.perceived.transcript, let empty = perceived.firstIndex(of: "") { perceived[empty] = transcript }
        var receipt = Receipt(model: response.system2?.model)
        receipt.stages = response.receipt.stages.map { StageReceipt(stage: $0.stage.rawValue, model: $0.model, ms: $0.ms, cost: $0.cost) }
        receipt.completionTokens = response.receipt.stages.last?.tokens ?? 0
        self.init(perceived: perceived, answer: response.system2?.text ?? response.answers.values.compactMap(\.text).first ?? "", receipt: receipt)
    }
}
