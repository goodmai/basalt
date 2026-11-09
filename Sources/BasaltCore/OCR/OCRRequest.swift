import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum OCRTask: String, Sendable, CaseIterable, Encodable {
    case document, text, table, formula, chart
}

public enum OCRFormat: String, Sendable, CaseIterable {
    case markdown, json
}

public enum OCRDocument: Sendable {
    case pdf(Data)
    case image(ImageInput)

    public static let maxBytes = ImageInput.maxTotalBytes

    public init(data: Data) throws(BasaltError) {
        guard !data.isEmpty, data.count <= Self.maxBytes else {
            throw .invalidRequestStructure(details: "the document must hold 1 byte to \(Self.maxBytes >> 20) MB; this one has \(data.count) bytes")
        }
        if data.starts(with: Data("%PDF-".utf8)) {
            self = .pdf(data)
            return
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), let type = CGImageSourceGetType(source),
              let mime = UTType(type as String)?.preferredMIMEType else {
            throw .invalidRequestStructure(details: "the document is neither a PDF nor an image")
        }
        self = .image(try ImageInput(data: data, mime: mime))
    }

    public init(encoded: String) throws(BasaltError) {
        let payload = encoded.hasPrefix("data:") ? encoded.split(separator: ",", maxSplits: 1).last.map(String.init) ?? "" : encoded
        guard payload.utf8.count <= (Self.maxBytes + 2) / 3 * 4 else {
            throw .invalidRequestStructure(details: "the document is over the limit of \(Self.maxBytes >> 20) MB")
        }
        guard let data = Data(base64Encoded: payload, options: .ignoreUnknownCharacters) else {
            throw .invalidRequestStructure(details: "file must be base64 or a data:<mime>;base64,<data> URI")
        }
        try self.init(data: data)
    }

    func pages(_ selection: OCRPageSelection?, dpi: Double) throws(BasaltError) -> [(number: Int, image: ImageInput)] {
        switch self {
        case .image(let image):
            guard selection == nil || selection?.numbers == [1] else {
                throw .invalidRequestStructure(details: "an image has one page; leave pages out or send 1")
            }
            return [(1, image)]
        case .pdf(let data):
            return try OCRImage.render(pdf: data, pages: selection, dpi: dpi)
        }
    }
}

public struct OCRPageSelection: Sendable, Equatable {
    public static let maxPages = 32

    public let numbers: [Int]

    public init(_ text: String) throws(BasaltError) {
        var numbers = Set<Int>()
        for item in text.split(separator: ",", omittingEmptySubsequences: false) {
            let bounds = item.split(separator: "-", omittingEmptySubsequences: false).map { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard bounds.count <= 2, let first = bounds.first ?? nil, let last = bounds.last ?? nil, 1 <= first, first <= last else {
                throw .invalidRequestStructure(details: "pages must be page numbers and ranges from 1, such as 1-3,5; got '\(text.prefix(64))'")
            }
            guard last - first < Self.maxPages else { throw Self.tooMany }
            numbers.formUnion(first ... last)
            guard numbers.count <= Self.maxPages else { throw Self.tooMany }
        }
        self.numbers = numbers.sorted()
    }

    static let tooMany = BasaltError.invalidRequestStructure(details: "a request reads at most \(maxPages) pages; select them with pages, such as 1-\(maxPages)")

    static func resolve(_ selection: OCRPageSelection?, pageCount: Int) throws(BasaltError) -> [Int] {
        guard let selection else {
            guard pageCount <= maxPages else {
                throw .invalidRequestStructure(details: "the document has \(pageCount) pages and a request reads at most \(maxPages); select them with pages")
            }
            return Array(1 ... max(1, pageCount))
        }
        guard let past = selection.numbers.first(where: { $0 > pageCount }) else { return selection.numbers }
        throw .invalidRequestStructure(details: "page \(past) is past the end of the document, which has \(pageCount) pages")
    }
}

public struct OCRRequest: Sendable {
    public let document: OCRDocument
    public let pages: OCRPageSelection?
    public let task: OCRTask
    public let format: OCRFormat
    public let model: String?

    public init(document: OCRDocument, pages: String?, task: String?, format: String?, model: String?) throws(BasaltError) {
        guard let parsedTask = OCRTask(rawValue: task ?? OCRTask.document.rawValue) else {
            throw .invalidRequestStructure(details: "task must be one of \(OCRTask.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        guard let parsedFormat = OCRFormat(rawValue: format ?? OCRFormat.markdown.rawValue) else {
            throw .invalidRequestStructure(details: "format must be markdown or json")
        }
        guard (model?.count ?? 0) <= 256 else { throw .invalidRequestStructure(details: "model must be at most 256 characters") }
        self.document = document
        self.pages = try pages.map { text throws(BasaltError) in try OCRPageSelection(text) }
        self.task = parsedTask
        self.format = parsedFormat
        self.model = model?.isEmpty == true ? nil : model
    }
}

public struct OCRBox: Sendable, Equatable, Encodable {
    public let x1: Double
    public let y1: Double
    public let x2: Double
    public let y2: Double

    public static let page = OCRBox(x1: 0, y1: 0, x2: 1, y2: 1)

    public init(x1: Double, y1: Double, x2: Double, y2: Double) {
        (self.x1, self.y1, self.x2, self.y2) = (x1, y1, x2, y2)
    }

    var area: Double { max(0, x2 - x1) * max(0, y2 - y1) }

    func intersection(_ other: OCRBox) -> Double {
        OCRBox(x1: max(x1, other.x1), y1: max(y1, other.y1), x2: min(x2, other.x2), y2: min(y2, other.y2)).area
    }

    func coverage(by outer: OCRBox) -> Double {
        area == 0 ? 0 : intersection(outer) / area
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(contentsOf: [x1, y1, x2, y2])
    }
}

public struct OCRBlock: Sendable, Equatable, Encodable {
    public let type: String
    public let bbox: OCRBox
    public let text: String
}

public struct OCRPageResult: Sendable, Equatable, Encodable {
    public let page: Int
    public let markdown: String
    public var blocks: [OCRBlock]?
    public let tokens: Int
    public let ms: Int
}

public struct OCRResult: Sendable, Encodable {
    public let model: String
    public let family: String
    public let task: OCRTask
    public let pages: [OCRPageResult]

    init(model: String, family: OCRRecipe.Family, task: OCRTask, pages: [OCRPageResult], format: OCRFormat) {
        self.model = model
        self.family = family.rawValue
        self.task = task
        self.pages = pages.map { page in
            var page = page
            if format == .markdown { page.blocks = nil }
            return page
        }
    }
}
