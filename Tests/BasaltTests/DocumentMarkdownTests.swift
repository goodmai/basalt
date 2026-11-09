import Testing
import Foundation
@testable import BasaltCore

@Suite("Document Markdown: Vision regions to paragraphs, lists and tables in reading order (#403)")
struct DocumentMarkdownTests {
    static func box(_ top: Double, _ bottom: Double, left: Double = 0.1, right: Double = 0.9) -> OCRBox {
        OCRBox(x1: left, y1: top, x2: right, y2: bottom)
    }

    static func cells(_ rows: [[String]]) -> [DocumentRegion.Cell] {
        rows.enumerated().flatMap { row in row.element.enumerated().map { DocumentRegion.Cell(row: row.offset, column: $0.offset, text: $0.element) } }
    }

    @Test("Paragraphs keep the order Vision gave them; the title, lists and tables go in before the first paragraph below their top edge")
    func readingOrder() {
        let page = DocumentMarkdown([
            DocumentRegion(.paragraph("Intro"), box: Self.box(0.10, 0.15)),
            DocumentRegion(.paragraph("Middle"), box: Self.box(0.50, 0.55)),
            DocumentRegion(.paragraph("End"), box: Self.box(0.90, 0.95)),
            DocumentRegion(.list(ordered: false, items: ["a", "b"]), box: Self.box(0.60, 0.70)),
            DocumentRegion(.table(Self.cells([["H1", "H2"], ["1", "2"]])), box: Self.box(0.20, 0.40)),
            DocumentRegion(.title("Report"), box: Self.box(0.01, 0.05)),
        ])
        #expect(page.markdown == "# Report\n\nIntro\n\n| H1 | H2 |\n| --- | --- |\n| 1 | 2 |\n\nMiddle\n\n- a\n- b\n\nEnd")
        #expect(page.blocks.map(\.type) == ["title", "text", "table", "text", "list", "text"])
        #expect(page.blocks[2].bbox == Self.box(0.20, 0.40))
    }

    @Test("A structure below every paragraph goes last, and two structures keep their top-to-bottom order")
    func trailingStructures() {
        let page = DocumentMarkdown([
            DocumentRegion(.list(ordered: true, items: ["late"]), box: Self.box(0.80, 0.85)),
            DocumentRegion(.paragraph("Only"), box: Self.box(0.10, 0.15)),
            DocumentRegion(.list(ordered: true, items: ["early"]), box: Self.box(0.30, 0.35)),
        ])
        #expect(page.markdown == "Only\n\n1. early\n\n1. late")
    }

    @Test("A paragraph mostly inside a title, list or table repeats its text and is dropped; one mostly outside stays")
    func dropsCoveredParagraphs() {
        let page = DocumentMarkdown([
            DocumentRegion(.paragraph("Report"), box: Self.box(0.01, 0.05)),
            DocumentRegion(.paragraph("1. First"), box: Self.box(0.30, 0.34)),
            DocumentRegion(.paragraph("Name Count"), box: Self.box(0.52, 0.56)),
            DocumentRegion(.paragraph("Caption"), box: Self.box(0.68, 0.76)),
            DocumentRegion(.title("Report"), box: Self.box(0.01, 0.05)),
            DocumentRegion(.list(ordered: true, items: ["First"]), box: Self.box(0.29, 0.35)),
            DocumentRegion(.table(Self.cells([["Name", "Count"]])), box: Self.box(0.50, 0.70)),
        ])
        #expect(page.blocks.map(\.text) == ["# Report", "1. First", "| Name | Count |\n| --- | --- |", "Caption"])
        #expect(DocumentMarkdown.coveredFraction == 0.5)
    }

    @Test("List items are numbered when ordered and bulleted otherwise; line breaks inside an item fold to spaces and empty items go")
    func lists() {
        #expect(DocumentRegion(.list(ordered: true, items: ["one", "", "two\n words "]), box: .page).markdown == "1. one\n2. two words")
        #expect(DocumentRegion(.list(ordered: false, items: ["x", "y"]), box: .page).markdown == "- x\n- y")
        #expect(DocumentRegion(.list(ordered: false, items: [" ", "\n"]), box: .page).markdown.isEmpty)
    }

    @Test("A table becomes a GFM pipe table from its first row, with missing cells empty, pipes escaped and cell line breaks folded")
    func tables() {
        let cells = [
            DocumentRegion.Cell(row: 0, column: 0, text: "Name"), DocumentRegion.Cell(row: 0, column: 2, text: "Note"),
            DocumentRegion.Cell(row: 1, column: 0, text: "a|b"), DocumentRegion.Cell(row: 1, column: 1, text: "two\nlines"),
            DocumentRegion.Cell(row: -1, column: 0, text: "ignored"),
        ]
        #expect(DocumentRegion(.table(cells), box: .page).markdown == "| Name |  | Note |\n| --- | --- | --- |\n| a\\|b | two lines |  |")
        #expect(DocumentRegion(.table([]), box: .page).markdown.isEmpty)
        #expect(DocumentRegion(.table(Self.cells([[" ", ""]])), box: .page).markdown.isEmpty)
    }

    @Test("Empty regions leave no block; a paragraph keeps its own line breaks; an empty page is empty Markdown")
    func emptyRegions() {
        let page = DocumentMarkdown([
            DocumentRegion(.title("  "), box: Self.box(0.0, 0.1)),
            DocumentRegion(.paragraph("\n"), box: Self.box(0.2, 0.3)),
            DocumentRegion(.paragraph(" line one\nline two "), box: Self.box(0.4, 0.5)),
        ])
        #expect(page.blocks == [OCRBlock(type: "text", bbox: Self.box(0.4, 0.5), text: "line one\nline two")])
        #expect(page.markdown == "line one\nline two")
        #expect(DocumentMarkdown([]).markdown.isEmpty)
    }

    @Test("Image fallback: each image becomes a numbered [image N, OCR] block after its message's text, counted across the conversation")
    func replacesImagesWithTranscripts() throws {
        let image = try ImageInput(data: Data("png".utf8), mime: "image/png")
        let messages = [
            ChatMessageInput(role: .system, text: "Be brief."),
            ChatMessageInput(role: .user, text: "Compare these.", images: [image]),
            ChatMessageInput(role: .assistant, text: "Send more."),
            ChatMessageInput(role: .user, text: "", images: [image, image]),
        ]
        let replaced = ChatMessageInput.replacingImages(in: messages, with: ["# Invoice", "Total 42", ""])
        #expect(replaced.allSatisfy { $0.images.isEmpty })
        #expect(replaced.map(\.role) == [.system, .user, .assistant, .user])
        #expect(replaced[0] == messages[0] && replaced[2] == messages[2])
        #expect(replaced[1].text == "Compare these.\n\n[image 1, OCR]\n# Invoice")
        #expect(replaced[3].text == "[image 2, OCR]\nTotal 42\n\n[image 3, OCR]\n(no text recognized)")
    }
}
