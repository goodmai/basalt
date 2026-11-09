import Testing
import Foundation
@testable import BasaltCore

@Suite("MinerU2.5 layout output, crops' prompts and block assembly (#400)")
struct MinerULayoutTests {
    static let page = """
    <|box_start|>100 50 900 120<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>
    <|box_start|>100 150 900 400<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
    <|box_start|>100 410 900 600<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>txt_contd_tgt
    <|box_start|>100 620 900 900<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
    <|box_start|>150 650 400 700<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
    <|box_start|>950 980 920 940<|box_end|><|ref_start|>page_number<|ref_end|><|rotate_left|>
    <|box_start|>0 0 1001 10<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
    <|box_start|>10 10 10 20<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
    <|box_start|>1 1 5 5<|box_end|><|ref_start|>inline_formula<|ref_end|><|rotate_up|>
    <|box_start|>1 1 5 5<|box_end|><|ref_start|>sticker<|ref_end|><|rotate_up|>
    <|box_start|>10 920 60 990<|box_end|><|ref_start|>Unknown<|ref_end|>
    <|box_start|>100 150 900 401<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
    """

    @Test("The stage-1 output parses in the mineru-vl-utils format: 0-1000 boxes normalized, swapped corners fixed, rotation tokens read")
    func parsesLayout() {
        let blocks = MinerULayout(parsing: Self.page).blocks
        #expect(blocks.map(\.type) == ["title", "text", "text", "table", "page_number", "image"])
        #expect(blocks[0].box == OCRBox(x1: 0.1, y1: 0.05, x2: 0.9, y2: 0.12))
        #expect(blocks[4].box == OCRBox(x1: 0.92, y1: 0.94, x2: 0.95, y2: 0.98))
        #expect(blocks.map(\.angle) == [0, 0, 0, 0, 270, nil])
        #expect(blocks.map(\.continues) == [false, false, true, false, false, false])
    }

    @Test("Blocks parse the same without newlines between them, and output that is not layout yields no blocks")
    func parsesRunTogether() {
        let joined = Self.page.replacingOccurrences(of: "\n", with: "")
        #expect(MinerULayout(parsing: joined).blocks == MinerULayout(parsing: Self.page).blocks)
        #expect(MinerULayout(parsing: "I cannot see a document.").blocks.isEmpty)
    }

    @Test("Text inside a table and a caption inside its image are dropped; duplicates of one type and angle collapse")
    func filtersCoveredAndDuplicates() {
        let output = """
        <|box_start|>0 0 500 500<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>\
        <|box_start|>10 400 490 490<|box_end|><|ref_start|>image_caption<|ref_end|><|rotate_up|>\
        <|box_start|>10 510 490 560<|box_end|><|ref_start|>image_caption<|ref_end|><|rotate_up|>\
        <|box_start|>0 600 1000 1000<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>\
        <|box_start|>10 610 200 700<|box_end|><|ref_start|>equation<|ref_end|><|rotate_up|>\
        <|box_start|>0 0 500 500<|box_end|><|ref_start|>image<|ref_end|><|rotate_right|>
        """
        let blocks = MinerULayout(parsing: output).blocks
        #expect(blocks.map(\.type) == ["image", "image_caption", "table", "image"])
        #expect(blocks.map(\.angle) == [0, 0, 0, 90])
    }

    @Test("Each block type gets its crop prompt; images, charts and containers are not recognized",
          arguments: [("table", "\nTable Recognition:"), ("equation", "\nFormula Recognition:"), ("text", "\nText Recognition:"),
                      ("title", "\nText Recognition:"), ("page_number", "\nText Recognition:"), ("image", nil), ("chart", nil),
                      ("list", nil), ("equation_block", nil), ("image_block", nil)])
    func cropPrompts(_ type: String, _ prompt: String?) {
        #expect(MinerULayout.prompt(forBlock: type) == prompt)
    }

    @Test("Blocks assemble to Markdown like json2md: paragraphs joined by a blank line, continued text merged, empty blocks kept only in blocks")
    func assembles() {
        func block(_ type: String, continues: Bool = false) -> MinerULayout.Block {
            MinerULayout.Block(type: type, box: .page, angle: 0, continues: continues)
        }
        let assembled = MinerULayout.assemble([
            (block("title"), "# Title"), (block("text"), "first half"), (block("image"), ""),
            (block("text", continues: true), "second half"), (block("text", continues: true), "中文"), (block("table"), "<table></table>"),
        ])
        #expect(assembled.markdown == "# Title\n\nfirst half second half中文\n\n<table></table>")
        #expect(assembled.blocks.map(\.type) == ["title", "text", "image", "text", "text", "table"])
        #expect(assembled.blocks[2].text.isEmpty)
        #expect(MinerULayout.assemble([(block("text", continues: true), "orphan")]).markdown == "orphan")
    }

    @Test("OTSL tables convert to HTML with column and row spans, padded rows and escaped text, as otsl2html does")
    func convertsOTSL() {
        #expect(OTSLTable.html("<fcel>A<fcel>B<nl><fcel>C<lcel><nl>")
            == "<table><tr><td>A</td><td>B</td></tr><tr><td colspan=\"2\">C</td></tr></table>")
        #expect(OTSLTable.html("<fcel>A<fcel>B<nl><ucel><fcel>C<nl>")
            == "<table><tr><td rowspan=\"2\">A</td><td>B</td></tr><tr><td>C</td></tr></table>")
        #expect(OTSLTable.html("<fcel>A<lcel><nl><ucel><xcel><nl><fcel>B<nl>")
            == "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td></tr><tr></tr><tr><td>B</td><td></td></tr></table>")
        #expect(OTSLTable.html("<fcel>a < b & \"c\"<ecel>ignored<nl>")
            == "<table><tr><td>a &lt; b &amp; &quot;c&quot;</td><td></td></tr></table>")
        #expect(OTSLTable.html("<table><tr><td>x</td></tr></table>") == "<table><tr><td>x</td></tr></table>")
        #expect(OTSLTable.html("| a | b |") == "| a | b |")
    }
}
