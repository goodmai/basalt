import Testing
@testable import BasaltUI

@Suite("Table layout value core")
struct TableLayoutTests {
    @Test("Column widths take the longest line of a multi-line cell and ignore cells past the headers")
    func columnWidths() {
        #expect(TableLayout.columnWidths(headers: ["ab", "c"], rows: [["x\nlonger", "y", "ignored-extra"], ["abc"]]) == [6, 1])
    }

    @Test("A cell is flattened to one line, truncated to the width, and padded by its alignment")
    func padCell() {
        #expect(TableLayout.padCell("a\nb", width: 5, alignment: .left) == "a b  ")
        #expect(TableLayout.padCell("ab", width: 5, alignment: .right) == "   ab")
        #expect(TableLayout.padCell("ab", width: 5, alignment: .center) == " ab  ")
        #expect(TableLayout.padCell("abcdef", width: 3, alignment: .right) == "abc")
        #expect(TableLayout.padCell("abc", width: 3, alignment: .center) == "abc")
    }

    @Test("Rows drop cells past the widths, and short rows are filled with blank cells; alignment defaults to left")
    func cellsAndFill() {
        let layout = TableLayout(widths: [2, 3], alignments: [.right])
        #expect(layout.header(["a", "b", "extra"]) == [" a", "b  ", "extra"])
        #expect(layout.cells(["a", "b", "c"]) == [" a", "b  "])
        #expect(layout.cells(["a"]) == [" a"])
        #expect(layout.fill(["a"]) == ["   "])
        #expect(layout.fill(["a", "b"]).isEmpty)
    }

    @Test("Border rules span each column plus its padding")
    func rules() {
        #expect(TableBorders.ascii.rule(TableBorders.ascii.top, widths: [1, 2]) == "+---+----+\n")
        #expect(TableBorders.unicode.rule(TableBorders.unicode.bottom, widths: [1]) == "╰───╯\n")
        #expect(TableBorders.unicode.line(["a", "bc"]) == "│ a │ bc │\n")
    }

    @Test("The minimal style separates columns by two spaces up to the last header")
    func minimal() {
        let layout = TableLayout(widths: [1, 2], alignments: [])
        #expect(layout.minimal(headers: ["a", "b"], rows: [["x"]]) == "a  b \n-  --\nx    \n")
    }
}
