import Testing
import Foundation
@testable import GemCore

/// Epic 16.3: Markdown & Syntax Highlighting - Unit Tests
/// Following TDD: Write tests FIRST, then implement
@Suite("MarkdownRenderer Tests - Epic 16.3")
struct MarkdownRendererTests {
    
    // MARK: - Basic Markdown Tests
    
    @Test("Render plain text")
    func testPlainText() {
        let markdown = "Hello, world!"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Hello, world!"))
    }
    
    @Test("Render bold text")
    func testBoldText() {
        let markdown = "This is **bold** text"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("bold"))
        // Bold should be visually distinct (ANSI or plain)
        #expect(!output.isEmpty)
    }
    
    @Test("Render italic text")
    func testItalicText() {
        let markdown = "This is *italic* text"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("italic"))
    }
    
    @Test("Render inline code")
    func testInlineCode() {
        let markdown = "Use `print()` function"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("print()"))
    }
    
    // MARK: - Header Tests
    
    @Test("Render H1 header")
    func testH1Header() {
        let markdown = "# Main Title"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Main Title"))
    }
    
    @Test("Render H2 header")
    func testH2Header() {
        let markdown = "## Subtitle"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Subtitle"))
    }
    
    @Test("Render multiple headers")
    func testMultipleHeaders() {
        let markdown = """
        # Title
        ## Section
        ### Subsection
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Title"))
        #expect(output.contains("Section"))
        #expect(output.contains("Subsection"))
    }
    
    // MARK: - List Tests
    
    @Test("Render unordered list")
    func testUnorderedList() {
        let markdown = """
        - Item 1
        - Item 2
        - Item 3
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Item 1"))
        #expect(output.contains("Item 2"))
        #expect(output.contains("Item 3"))
    }
    
    @Test("Render ordered list")
    func testOrderedList() {
        let markdown = """
        1. First
        2. Second
        3. Third
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("First"))
        #expect(output.contains("Second"))
        #expect(output.contains("Third"))
    }
    
    // MARK: - Code Block Tests
    
    @Test("Render code block without language")
    func testCodeBlockNoLanguage() {
        let markdown = """
        ```
        let x = 42
        print(x)
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("let x = 42"))
        #expect(output.contains("print(x)"))
    }
    
    @Test("Render Swift code block")
    func testSwiftCodeBlock() {
        let markdown = """
        ```swift
        func hello() {
            print("Hello")
        }
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("func hello()"))
        #expect(output.contains("print"))
    }
    
    @Test("Render Python code block")
    func testPythonCodeBlock() {
        let markdown = """
        ```python
        def hello():
            print("Hello")
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("def hello()"))
    }
    
    @Test("Render JavaScript code block")
    func testJavaScriptCodeBlock() {
        let markdown = """
        ```javascript
        function hello() {
            console.log("Hello");
        }
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("function hello()"))
    }
    
    @Test("Render JSON code block")
    func testJSONCodeBlock() {
        let markdown = """
        ```json
        {
            "name": "test",
            "value": 42
        }
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("name"))
        #expect(output.contains("test"))
    }
    
    // MARK: - Link Tests
    
    @Test("Render inline link")
    func testInlineLink() {
        let markdown = "Visit [GitHub](https://github.com)"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("GitHub"))
    }
    
    @Test("Render reference link")
    func testReferenceLink() {
        let markdown = """
        See [docs][1]
        
        [1]: https://example.com
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("docs"))
    }
    
    // MARK: - Blockquote Tests
    
    @Test("Render blockquote")
    func testBlockquote() {
        let markdown = "> This is a quote"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("This is a quote"))
    }
    
    @Test("Render multi-line blockquote")
    func testMultiLineBlockquote() {
        let markdown = """
        > Line 1
        > Line 2
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Line 1"))
        #expect(output.contains("Line 2"))
    }
    
    // MARK: - Complex Markdown Tests
    
    @Test("Render mixed content")
    func testMixedContent() {
        let markdown = """
        # Title
        
        This is **bold** and *italic*.
        
        - Item 1
        - Item 2
        
        ```swift
        let x = 42
        ```
        """
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Title"))
        #expect(output.contains("bold"))
        #expect(output.contains("italic"))
        #expect(output.contains("Item 1"))
        #expect(output.contains("let x = 42"))
    }
    
    // MARK: - Edge Cases
    
    @Test("Handle empty markdown")
    func testEmptyMarkdown() {
        let output = MarkdownRenderer.render("")
        #expect(output.isEmpty || output == "\n")
    }
    
    @Test("Handle special characters")
    func testSpecialCharacters() {
        let markdown = "Test < > & \" '"
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Test"))
    }
}

// MARK: - Syntax Highlighter Tests

@Suite("SyntaxHighlighter Tests - Epic 16.3")
struct SyntaxHighlighterTests {
    
    @Test("Highlight Swift code")
    func testSwiftHighlighting() {
        let code = """
        func hello() {
            print("World")
        }
        """
        
        let highlighted = SyntaxHighlighter.highlight(code, language: .swift)
        
        #expect(!highlighted.isEmpty)
        #expect(highlighted.contains("func") || highlighted.contains("hello"))
    }
    
    @Test("Highlight Python code")
    func testPythonHighlighting() {
        let code = """
        def hello():
            print("World")
        """
        
        let highlighted = SyntaxHighlighter.highlight(code, language: .python)
        
        #expect(!highlighted.isEmpty)
        #expect(highlighted.contains("def") || highlighted.contains("hello"))
    }
    
    @Test("Highlight JavaScript code")
    func testJavaScriptHighlighting() {
        let code = """
        function hello() {
            console.log("World");
        }
        """
        
        let highlighted = SyntaxHighlighter.highlight(code, language: .javascript)
        
        #expect(!highlighted.isEmpty)
    }
    
    @Test("Highlight JSON")
    func testJSONHighlighting() {
        let code = """
        {
            "key": "value"
        }
        """
        
        let highlighted = SyntaxHighlighter.highlight(code, language: .json)
        
        #expect(!highlighted.isEmpty)
    }
    
    @Test("Highlight Bash")
    func testBashHighlighting() {
        let code = """
        #!/bin/bash
        echo "Hello"
        """
        
        let highlighted = SyntaxHighlighter.highlight(code, language: .bash)
        
        #expect(!highlighted.isEmpty)
    }
    
    @Test("Handle plain text (no highlighting)")
    func testPlainText() {
        let code = "Just plain text"
        let highlighted = SyntaxHighlighter.highlight(code, language: .plain)
        
        #expect(highlighted == code || highlighted.contains("Just plain text"))
    }
}

// MARK: - Integration Tests

@Suite("Markdown + Syntax Integration Tests")
struct MarkdownSyntaxIntegrationTests {
    
    @Test("Render markdown with syntax-highlighted code")
    func testMarkdownWithHighlightedCode() {
        let markdown = """
        # Example
        
        Here's some Swift code:
        
        ```swift
        func test() {
            print("Hello")
        }
        ```
        
        And that is it!
        """
        
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Example"))
        #expect(output.contains("Swift code"))
        #expect(output.contains("func test()"))
        #expect(output.contains("that is it"))
    }
    
    @Test("Render multiple code blocks with different languages")
    func testMultipleCodeBlocks() {
        let markdown = """
        ## Swift
        ```swift
        let x = 42
        ```
        
        ## Python
        ```python
        x = 42
        ```
        """
        
        let output = MarkdownRenderer.render(markdown)
        
        #expect(output.contains("Swift"))
        #expect(output.contains("Python"))
        #expect(output.contains("let x = 42") || output.contains("x = 42"))
    }
}
