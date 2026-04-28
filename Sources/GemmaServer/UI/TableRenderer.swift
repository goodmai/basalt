import Foundation

/// Epic 16.2: Table Rendering System
/// Provides ASCII/Unicode table rendering without ConsoleKit dependency
public struct TableRenderer: Sendable {
    
    // MARK: - Table Style
    
    /// Visual style for table borders
    public enum TableStyle: Sendable {
        case ascii      // Uses -, +, |
        case unicode    // Uses box-drawing characters ─, │, ┌, etc.
        case minimal    // No borders, just spacing
    }
    
    /// Column alignment
    public enum ColumnAlignment: Sendable {
        case left
        case center
        case right
    }
    
    // MARK: - Main Rendering API
    
    /// Render a table with headers and rows
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
        
        // Calculate column widths
        let columnWidths = widths ?? calculateColumnWidths(headers: headers, rows: rows)
        
        // Determine alignments
        let columnAlignments = alignment ?? Array(repeating: .left, count: headers.count)
        
        var output = ""
        
        // Render based on style
        switch style {
        case .unicode:
            output = renderUnicodeTable(
                headers: headers,
                rows: rows,
                widths: columnWidths,
                alignments: columnAlignments,
                colorHeaders: colorHeaders,
                alternateRowColors: alternateRowColors
            )
        case .ascii:
            output = renderASCIITable(
                headers: headers,
                rows: rows,
                widths: columnWidths,
                alignments: columnAlignments,
                colorHeaders: colorHeaders,
                alternateRowColors: alternateRowColors
            )
        case .minimal:
            output = renderMinimalTable(
                headers: headers,
                rows: rows,
                widths: columnWidths,
                alignments: columnAlignments
            )
        }
        
        return output
    }
    
    // MARK: - Private Helpers
    
    private static func calculateColumnWidths(headers: [String], rows: [[String]]) -> [Int] {
        var widths = headers.map { $0.count }
        
        for row in rows {
            for (index, cell) in row.enumerated() {
                guard index < widths.count else { continue }
                // Handle newlines - take max line length
                let cellLines = cell.split(separator: "\n", omittingEmptySubsequences: false)
                let maxLineLength = cellLines.map { $0.count }.max() ?? 0
                widths[index] = max(widths[index], maxLineLength)
            }
        }
        
        return widths
    }
    
    private static func padCell(
        _ text: String,
        width: Int,
        alignment: ColumnAlignment
    ) -> String {
        // Remove newlines for single-line display
        let cleanText = text.replacingOccurrences(of: "\n", with: " ")
        
        guard cleanText.count < width else {
            // Truncate if too long
            return String(cleanText.prefix(width))
        }
        
        let padding = width - cleanText.count
        
        switch alignment {
        case .left:
            return cleanText + String(repeating: " ", count: padding)
        case .right:
            return String(repeating: " ", count: padding) + cleanText
        case .center:
            let leftPad = padding / 2
            let rightPad = padding - leftPad
            return String(repeating: " ", count: leftPad) + cleanText + String(repeating: " ", count: rightPad)
        }
    }
    
    // MARK: - Unicode Table Rendering
    
    private static func renderUnicodeTable(
        headers: [String],
        rows: [[String]],
        widths: [Int],
        alignments: [ColumnAlignment],
        colorHeaders: Bool,
        alternateRowColors: Bool
    ) -> String {
        var output = ""
        
        // Top border: ╭─┬─╮
        output += "╭"
        output += widths.map { String(repeating: "─", count: $0 + 2) }.joined(separator: "┬")
        output += "╮\n"
        
        // Headers: │ Name │ Age │
        output += "│"
        for (index, header) in headers.enumerated() {
            let width = index < widths.count ? widths[index] : header.count
            let alignment = index < alignments.count ? alignments[index] : .left
            let paddedHeader = padCell(header, width: width, alignment: alignment)
            
            if colorHeaders {
                output += " \(TerminalUI.bold(paddedHeader)) │"
            } else {
                output += " \(paddedHeader) │"
            }
        }
        output += "\n"
        
        // Header separator: ├─┼─┤
        output += "├"
        output += widths.map { String(repeating: "─", count: $0 + 2) }.joined(separator: "┼")
        output += "┤\n"
        
        // Data rows
        for (rowIndex, row) in rows.enumerated() {
            output += "│"
            for (colIndex, cell) in row.enumerated() {
                guard colIndex < widths.count else { continue }
                let width = widths[colIndex]
                let alignment = colIndex < alignments.count ? alignments[colIndex] : .left
                let paddedCell = padCell(cell, width: width, alignment: alignment)
                
                if alternateRowColors && rowIndex % 2 == 1 {
                    output += " \(TerminalUI.dim(paddedCell)) │"
                } else {
                    output += " \(paddedCell) │"
                }
            }
            
            // Fill missing columns with empty cells
            for colIndex in row.count..<widths.count {
                let width = widths[colIndex]
                output += " \(String(repeating: " ", count: width)) │"
            }
            
            output += "\n"
        }
        
        // Bottom border: ╰─┴─╯
        output += "╰"
        output += widths.map { String(repeating: "─", count: $0 + 2) }.joined(separator: "┴")
        output += "╯\n"
        
        return output
    }
    
    // MARK: - ASCII Table Rendering
    
    private static func renderASCIITable(
        headers: [String],
        rows: [[String]],
        widths: [Int],
        alignments: [ColumnAlignment],
        colorHeaders: Bool,
        alternateRowColors: Bool
    ) -> String {
        var output = ""
        
        // Top border: +---+---+
        output += "+"
        output += widths.map { String(repeating: "-", count: $0 + 2) }.joined(separator: "+")
        output += "+\n"
        
        // Headers
        output += "|"
        for (index, header) in headers.enumerated() {
            let width = index < widths.count ? widths[index] : header.count
            let alignment = index < alignments.count ? alignments[index] : .left
            let paddedHeader = padCell(header, width: width, alignment: alignment)
            
            if colorHeaders {
                output += " \(TerminalUI.bold(paddedHeader)) |"
            } else {
                output += " \(paddedHeader) |"
            }
        }
        output += "\n"
        
        // Header separator
        output += "+"
        output += widths.map { String(repeating: "-", count: $0 + 2) }.joined(separator: "+")
        output += "+\n"
        
        // Data rows
        for (rowIndex, row) in rows.enumerated() {
            output += "|"
            for (colIndex, cell) in row.enumerated() {
                guard colIndex < widths.count else { continue }
                let width = widths[colIndex]
                let alignment = colIndex < alignments.count ? alignments[colIndex] : .left
                let paddedCell = padCell(cell, width: width, alignment: alignment)
                
                if alternateRowColors && rowIndex % 2 == 1 {
                    output += " \(TerminalUI.dim(paddedCell)) |"
                } else {
                    output += " \(paddedCell) |"
                }
            }
            
            // Fill missing columns
            for colIndex in row.count..<widths.count {
                let width = widths[colIndex]
                output += " \(String(repeating: " ", count: width)) |"
            }
            
            output += "\n"
        }
        
        // Bottom border
        output += "+"
        output += widths.map { String(repeating: "-", count: $0 + 2) }.joined(separator: "+")
        output += "+\n"
        
        return output
    }
    
    // MARK: - Minimal Table Rendering
    
    private static func renderMinimalTable(
        headers: [String],
        rows: [[String]],
        widths: [Int],
        alignments: [ColumnAlignment]
    ) -> String {
        var output = ""
        
        // Headers
        for (index, header) in headers.enumerated() {
            let width = index < widths.count ? widths[index] : header.count
            let alignment = index < alignments.count ? alignments[index] : .left
            output += padCell(header, width: width, alignment: alignment)
            if index < headers.count - 1 {
                output += "  "  // Two spaces between columns
            }
        }
        output += "\n"
        
        // Separator (dashes under headers)
        for (index, _) in headers.enumerated() {
            let width = index < widths.count ? widths[index] : 0
            output += String(repeating: "-", count: width)
            if index < headers.count - 1 {
                output += "  "
            }
        }
        output += "\n"
        
        // Data rows
        for row in rows {
            for (colIndex, cell) in row.enumerated() {
                guard colIndex < widths.count else { continue }
                let width = widths[colIndex]
                let alignment = colIndex < alignments.count ? alignments[colIndex] : .left
                output += padCell(cell, width: width, alignment: alignment)
                if colIndex < headers.count - 1 {
                    output += "  "
                }
            }
            
            // Fill missing columns
            for colIndex in row.count..<widths.count {
                let width = widths[colIndex]
                output += String(repeating: " ", count: width)
                if colIndex < headers.count - 1 {
                    output += "  "
                }
            }
            
            output += "\n"
        }
        
        return output
    }
}

// MARK: - Table Builder

/// Builder pattern for constructing tables
public class TableBuilder: @unchecked Sendable {
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
