import Foundation

public protocol DocumentTranscriber: Sendable {
    func transcribe(_ image: ImageInput) async throws(BasaltError) -> DocumentMarkdown
}

public struct DocumentRegion: Sendable, Equatable {
    public struct Cell: Sendable, Equatable {
        public let row: Int
        public let column: Int
        public let text: String

        public init(row: Int, column: Int, text: String) {
            (self.row, self.column, self.text) = (row, column, text)
        }
    }

    public enum Element: Sendable, Equatable {
        case title(String)
        case paragraph(String)
        case list(ordered: Bool, items: [String])
        case table([Cell])
    }

    public let element: Element
    public let box: OCRBox

    public init(_ element: Element, box: OCRBox) {
        self.element = element
        self.box = box
    }

    var isParagraph: Bool {
        if case .paragraph = element { true } else { false }
    }

    var blockType: String {
        switch element {
        case .title: "title"
        case .paragraph: "text"
        case .list: "list"
        case .table: "table"
        }
    }

    var markdown: String {
        switch element {
        case .title(let text): Self.line(text).isEmpty ? "" : "# \(Self.line(text))"
        case .paragraph(let text): text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .list(let ordered, let items):
            items.map(Self.line).filter { !$0.isEmpty }.enumerated()
                .map { ordered ? "\($0.offset + 1). \($0.element)" : "- \($0.element)" }.joined(separator: "\n")
        case .table(let cells): Self.table(cells)
        }
    }

    static func line(_ text: String) -> String {
        text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")
    }

    static func table(_ cells: [Cell]) -> String {
        let cells = cells.filter { $0.row >= 0 && $0.column >= 0 }
        guard let rows = cells.map(\.row).max(), let columns = cells.map(\.column).max(), cells.contains(where: { !line($0.text).isEmpty }) else {
            return ""
        }
        var grid = Array(repeating: Array(repeating: "", count: columns + 1), count: rows + 1)
        for cell in cells { grid[cell.row][cell.column] = line(cell.text).replacingOccurrences(of: "|", with: "\\|") }
        let lines = grid.map { "| \($0.joined(separator: " | ")) |" }
        return ([lines[0], "|" + String(repeating: " --- |", count: columns + 1)] + lines.dropFirst()).joined(separator: "\n")
    }
}

public struct DocumentMarkdown: Sendable, Equatable {
    public static let coveredFraction = 0.5

    public let markdown: String
    public let blocks: [OCRBlock]

    public init(_ regions: [DocumentRegion]) {
        let structures = regions.filter { !$0.isParagraph }
        let paragraphs = regions.filter { region in
            region.isParagraph && !structures.contains { region.box.coverage(by: $0.box) >= Self.coveredFraction }
        }
        blocks = Self.merged(paragraphs, structures.sorted { $0.box.y1 < $1.box.y1 }).compactMap { region in
            let text = region.markdown
            return text.isEmpty ? nil : OCRBlock(type: region.blockType, bbox: region.box, text: text)
        }
        markdown = blocks.map(\.text).joined(separator: "\n\n")
    }

    static func merged(_ paragraphs: [DocumentRegion], _ structures: [DocumentRegion]) -> [DocumentRegion] {
        var pending = structures[...]
        var ordered: [DocumentRegion] = []
        for paragraph in paragraphs {
            while let next = pending.first, next.box.y1 <= paragraph.box.y1 {
                ordered.append(next)
                pending.removeFirst()
            }
            ordered.append(paragraph)
        }
        return ordered + pending
    }
}
