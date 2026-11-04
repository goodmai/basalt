import Foundation
import BasaltCore

public struct TableRenderer: Sendable {
    public enum TableStyle: Sendable {
        case ascii
        case unicode
        case minimal
    }

    public enum ColumnAlignment: Sendable {
        case left
        case center
        case right
    }

    public static func render(
        headers: [String],
        rows: [[String]],
        alignment: [ColumnAlignment]? = nil,
        widths: [Int]? = nil,
        style: TableStyle = .unicode,
        colorHeaders: Bool = false,
        alternateRowColors: Bool = false
    ) -> String {
        guard !headers.isEmpty else { return "" }
        let layout = TableLayout(
            widths: widths ?? TableLayout.columnWidths(headers: headers, rows: rows),
            alignments: alignment ?? Array(repeating: .left, count: headers.count)
        )
        switch style {
        case .unicode:
            return layout.bordered(headers: headers, rows: rows, borders: .unicode, colorHeaders: colorHeaders, alternateRowColors: alternateRowColors)
        case .ascii:
            return layout.bordered(headers: headers, rows: rows, borders: .ascii, colorHeaders: colorHeaders, alternateRowColors: alternateRowColors)
        case .minimal:
            return layout.minimal(headers: headers, rows: rows)
        }
    }
}

struct TableLayout: Sendable, Equatable {
    let widths: [Int]
    let alignments: [TableRenderer.ColumnAlignment]

    static func columnWidths(headers: [String], rows: [[String]]) -> [Int] {
        var widths = headers.map(\.count)
        for row in rows {
            for (index, cell) in row.enumerated() where index < widths.count {
                let longestLine = cell.split(separator: "\n", omittingEmptySubsequences: false).map(\.count).max() ?? 0
                widths[index] = max(widths[index], longestLine)
            }
        }
        return widths
    }

    static func padCell(_ text: String, width: Int, alignment: TableRenderer.ColumnAlignment) -> String {
        let cleanText = text.replacingOccurrences(of: "\n", with: " ")
        guard cleanText.count < width else { return String(cleanText.prefix(width)) }
        let padding = width - cleanText.count
        switch alignment {
        case .left:
            return cleanText + String(repeating: " ", count: padding)
        case .right:
            return String(repeating: " ", count: padding) + cleanText
        case .center:
            return String(repeating: " ", count: padding / 2) + cleanText + String(repeating: " ", count: padding - padding / 2)
        }
    }

    func alignment(_ column: Int) -> TableRenderer.ColumnAlignment {
        column < alignments.count ? alignments[column] : .left
    }

    func header(_ headers: [String]) -> [String] {
        headers.enumerated().map { index, header in
            Self.padCell(header, width: index < widths.count ? widths[index] : header.count, alignment: alignment(index))
        }
    }

    func cells(_ row: [String]) -> [String] {
        row.prefix(widths.count).enumerated().map { index, cell in
            Self.padCell(cell, width: widths[index], alignment: alignment(index))
        }
    }

    func fill(_ row: [String]) -> [String] {
        (row.count..<widths.count).map { String(repeating: " ", count: widths[$0]) }
    }
}

struct TableBorders: Sendable, Equatable {
    let horizontal: String
    let vertical: String
    let top: [String]
    let separator: [String]
    let bottom: [String]

    static let unicode = TableBorders(
        horizontal: "─", vertical: "│", top: ["╭", "┬", "╮"], separator: ["├", "┼", "┤"], bottom: ["╰", "┴", "╯"]
    )
    static let ascii = TableBorders(
        horizontal: "-", vertical: "|", top: ["+", "+", "+"], separator: ["+", "+", "+"], bottom: ["+", "+", "+"]
    )

    func rule(_ glyphs: [String], widths: [Int]) -> String {
        glyphs[0] + widths.map { String(repeating: horizontal, count: $0 + 2) }.joined(separator: glyphs[1]) + glyphs[2] + "\n"
    }

    func line(_ cells: [String]) -> String {
        vertical + cells.map { " \($0) \(vertical)" }.joined() + "\n"
    }
}

extension TableLayout {
    func bordered(
        headers: [String], rows: [[String]], borders: TableBorders, colorHeaders: Bool, alternateRowColors: Bool
    ) -> String {
        var output = borders.rule(borders.top, widths: widths)
        output += borders.line(header(headers).map { colorHeaders ? TerminalUI.bold($0) : $0 })
        output += borders.rule(borders.separator, widths: widths)
        for (rowIndex, row) in rows.enumerated() {
            let dimmed = alternateRowColors && rowIndex % 2 == 1
            output += borders.line(cells(row).map { dimmed ? TerminalUI.dim($0) : $0 } + fill(row))
        }
        return output + borders.rule(borders.bottom, widths: widths)
    }

    func minimal(headers: [String], rows: [[String]]) -> String {
        let gapped: ([String]) -> String = { items in
            items.enumerated().map { $0.element + ($0.offset < headers.count - 1 ? "  " : "") }.joined() + "\n"
        }
        var output = gapped(header(headers))
        output += gapped(headers.indices.map { String(repeating: "-", count: $0 < widths.count ? widths[$0] : 0) })
        for row in rows {
            output += gapped(cells(row) + fill(row))
        }
        return output
    }
}

/// Builder pattern for constructing tables
public final class TableBuilder {
    private var headers: [String] = []
    private var rows: [[String]] = []
    private var alignments: [TableRenderer.ColumnAlignment] = []
    private var widths: [Int]?
    private var style: TableRenderer.TableStyle = .unicode
    private var colorHeaders = false
    private var alternateRowColors = false
    
    public init() {}
    
    /// Add a header column
    @discardableResult
    public func addHeader(_ header: String) -> TableBuilder {
        headers.append(header)
        return self
    }
    
    /// Add a column with configuration
    @discardableResult
    public func addColumn(
        _ header: String,
        alignment: TableRenderer.ColumnAlignment = .left,
        width: Int? = nil
    ) -> TableBuilder {
        headers.append(header)
        alignments.append(alignment)
        if let width = width {
            if widths == nil {
                widths = Array(repeating: 0, count: headers.count - 1)
            }
            widths?.append(width)
        }
        return self
    }
    
    /// Add a data row
    @discardableResult
    public func addRow(_ row: [String]) -> TableBuilder {
        rows.append(row)
        return self
    }
    
    /// Set table style
    @discardableResult
    public func setStyle(_ style: TableRenderer.TableStyle) -> TableBuilder {
        self.style = style
        return self
    }
    
    /// Enable colored headers
    @discardableResult
    public func withColoredHeaders() -> TableBuilder {
        self.colorHeaders = true
        return self
    }
    
    /// Enable alternating row colors
    @discardableResult
    public func withAlternatingRowColors() -> TableBuilder {
        self.alternateRowColors = true
        return self
    }
    
    /// Build and render the table
    public func build() -> String {
        return TableRenderer.render(
            headers: headers,
            rows: rows,
            alignment: alignments.isEmpty ? nil : alignments,
            widths: widths,
            style: style,
            colorHeaders: colorHeaders,
            alternateRowColors: alternateRowColors
        )
    }
}
