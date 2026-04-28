import Testing
import Foundation
@testable import GemmaServerCore

/// Epic 16.2: Table Rendering System - Unit Tests
/// Following TDD: Write tests FIRST, then implement
@Suite("TableRenderer Tests - Epic 16.2")
struct TableRendererTests {
    
    // MARK: - Basic Table Rendering Tests
    
    @Test("Render simple 2x2 table")
    func testSimpleTable() {
        let headers = ["Name", "Age"]
        let rows = [
            ["Alice", "30"],
            ["Bob", "25"]
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        // Verify headers are present
        #expect(output.contains("Name"))
        #expect(output.contains("Age"))
        
        // Verify data rows are present
        #expect(output.contains("Alice"))
        #expect(output.contains("30"))
        #expect(output.contains("Bob"))
        #expect(output.contains("25"))
        
        // Verify it's not empty
        #expect(!output.isEmpty)
    }
    
    @Test("Render table with column alignment")
    func testTableWithAlignment() {
        let headers = ["Left", "Center", "Right"]
        let rows = [
            ["A", "B", "C"]
        ]
        let alignment: [TableRenderer.ColumnAlignment] = [.left, .center, .right]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            alignment: alignment
        )
        
        #expect(output.contains("Left"))
        #expect(output.contains("Center"))
        #expect(output.contains("Right"))
    }
    
    @Test("Render empty table")
    func testEmptyTable() {
        let headers = ["Col1", "Col2"]
        let rows: [[String]] = []
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        // Headers should still be present
        #expect(output.contains("Col1"))
        #expect(output.contains("Col2"))
    }
    
    @Test("Render table with long content")
    func testTableWithLongContent() {
        let headers = ["ID", "Description"]
        let rows = [
            ["1", "This is a very long description that should be handled properly"],
            ["2", "Short"]
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        #expect(output.contains("This is a very long description"))
        #expect(output.contains("Short"))
    }
    
    // MARK: - Table Style Tests
    
    @Test("Render table with ASCII style")
    func testASCIIStyle() {
        let headers = ["A", "B"]
        let rows = [["1", "2"]]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            style: .ascii
        )
        
        // ASCII style uses -, +, |
        #expect(output.contains("-"))
        #expect(output.contains("|"))
    }
    
    @Test("Render table with Unicode style")
    func testUnicodeStyle() {
        let headers = ["A", "B"]
        let rows = [["1", "2"]]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            style: .unicode
        )
        
        // Unicode style uses box-drawing characters
        // Just verify it renders something
        #expect(!output.isEmpty)
        #expect(output.contains("A"))
        #expect(output.contains("B"))
    }
    
    @Test("Render table with minimal style")
    func testMinimalStyle() {
        let headers = ["A", "B"]
        let rows = [["1", "2"]]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            style: .minimal
        )
        
        // Minimal style - no borders, just spacing
        #expect(output.contains("A"))
        #expect(output.contains("B"))
        #expect(output.contains("1"))
        #expect(output.contains("2"))
    }
    
    // MARK: - Column Width Tests
    
    @Test("Auto-sizing columns based on content")
    func testAutoSizing() {
        let headers = ["Short", "VeryLongHeader"]
        let rows = [
            ["X", "A"],
            ["Y", "LongValue"]
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        // Verify all content is present
        #expect(output.contains("Short"))
        #expect(output.contains("VeryLongHeader"))
        #expect(output.contains("LongValue"))
    }
    
    @Test("Fixed column widths")
    func testFixedWidths() {
        let headers = ["A", "B", "C"]
        let rows = [["1", "2", "3"]]
        let widths = [5, 10, 15]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            widths: widths
        )
        
        #expect(output.contains("A"))
        #expect(output.contains("B"))
        #expect(output.contains("C"))
    }
    
    // MARK: - Edge Cases
    
    @Test("Handle mismatched row lengths")
    func testMismatchedRowLengths() {
        let headers = ["A", "B", "C"]
        let rows = [
            ["1", "2"],        // Missing one column
            ["3", "4", "5"],   // Full row
            ["6"]              // Missing two columns
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        // Should still render without crashing
        #expect(!output.isEmpty)
        #expect(output.contains("A"))
    }
    
    @Test("Handle special characters in cells")
    func testSpecialCharacters() {
        let headers = ["Name", "Symbol"]
        let rows = [
            ["Test", "✓"],
            ["Check", "→"],
            ["Emoji", "🎉"]
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        #expect(output.contains("✓"))
        #expect(output.contains("→"))
        #expect(output.contains("🎉"))
    }
    
    @Test("Handle newlines in cells")
    func testNewlinesInCells() {
        let headers = ["Single", "Multi"]
        let rows = [
            ["One", "Line1\nLine2"]
        ]
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        // Newlines should be replaced with space or handled gracefully
        #expect(output.contains("One"))
        #expect(output.contains("Line1") || output.contains("Line2"))
    }
    
    // MARK: - Color Support Tests
    
    @Test("Render table with colored headers")
    func testColoredHeaders() {
        let headers = ["Success", "Error"]
        let rows = [["OK", "Failed"]]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            colorHeaders: true
        )
        
        // Just verify it renders
        #expect(!output.isEmpty)
        #expect(output.contains("Success"))
        #expect(output.contains("Error"))
    }
    
    @Test("Render table with alternating row colors")
    func testAlternatingRowColors() {
        let headers = ["A"]
        let rows = [["1"], ["2"], ["3"]]
        
        let output = TableRenderer.render(
            headers: headers,
            rows: rows,
            alternateRowColors: true
        )
        
        #expect(!output.isEmpty)
        #expect(output.contains("1"))
        #expect(output.contains("2"))
        #expect(output.contains("3"))
    }
}

// MARK: - TableBuilder Tests

@Suite("TableBuilder Tests - Epic 16.2")
struct TableBuilderTests {
    
    @Test("Build table using builder pattern")
    func testBuilderPattern() {
        let table = TableBuilder()
            .addHeader("Name")
            .addHeader("Age")
            .addRow(["Alice", "30"])
            .addRow(["Bob", "25"])
            .build()
        
        #expect(!table.isEmpty)
        #expect(table.contains("Name"))
        #expect(table.contains("Alice"))
    }
    
    @Test("Build table with column configuration")
    func testBuilderWithColumnConfig() {
        let table = TableBuilder()
            .addColumn("ID", alignment: .right, width: 5)
            .addColumn("Name", alignment: .left, width: 20)
            .addRow(["1", "Test"])
            .build()
        
        #expect(!table.isEmpty)
        #expect(table.contains("ID"))
        #expect(table.contains("Name"))
    }
    
    @Test("Build table with style")
    func testBuilderWithStyle() {
        let table = TableBuilder()
            .setStyle(.unicode)
            .addHeader("A")
            .addRow(["1"])
            .build()
        
        #expect(!table.isEmpty)
    }
}

// MARK: - Performance Tests

@Suite("Table Performance Tests")
struct TablePerformanceTests {
    
    @Test("Render large table (100 rows)")
    func testLargeTable() {
        let headers = ["ID", "Name", "Status", "Value"]
        var rows: [[String]] = []
        
        for i in 1...100 {
            rows.append([
                "\(i)",
                "Item \(i)",
                i % 2 == 0 ? "Active" : "Inactive",
                "\(i * 100)"
            ])
        }
        
        let output = TableRenderer.render(headers: headers, rows: rows)
        
        #expect(!output.isEmpty)
        #expect(output.contains("ID"))
        #expect(output.contains("100"))
    }
    
    @Test("Render wide table (20 columns)")
    func testWideTable() {
        let headers = (1...20).map { "Col\($0)" }
        let row = (1...20).map { "\($0)" }
        
        let output = TableRenderer.render(headers: headers, rows: [row])
        
        #expect(!output.isEmpty)
        #expect(output.contains("Col1"))
        #expect(output.contains("Col20"))
    }
}
