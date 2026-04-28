import Foundation
import Markdown
import Splash

/// Epic 16.3: Markdown Rendering & Syntax Highlighting
/// Renders markdown with syntax-highlighted code blocks for terminal output
public struct MarkdownRenderer: Sendable {
    
    // MARK: - Main API
    
    /// Render markdown to terminal-formatted text
    public static func render(_ markdown: String, colorize: Bool = true) -> String {
        guard !markdown.isEmpty else { return "" }
        
        let document = Document(parsing: markdown)
        var visitor = TerminalMarkdownVisitor(colorize: colorize)
        return visitor.visit(document)
    }
}

// MARK: - Terminal Markdown Visitor

/// Visits markdown AST and converts to terminal-formatted text
struct TerminalMarkdownVisitor: MarkupWalker {
    let colorize: Bool
    var output = ""
    var listLevel = 0
    var listItemNumber = 1
    
    init(colorize: Bool = true) {
        self.colorize = colorize
    }
    
    mutating func visit(_ document: Document) -> String {
        output = ""
        for child in document.children {
            visitAny(child)
        }
        return output
    }
    
    // MARK: - Block Elements
    
    mutating func visitHeading(_ heading: Heading) -> () {
        let text = extractPlainText(from: heading)
        
        switch heading.level {
        case 1:
            if colorize {
                output += "\n" + TerminalUI.heading(text).underline + "\n\n"
            } else {
                output += "\n# \(text)\n\n"
            }
        case 2:
            if colorize {
                output += "\n" + TerminalUI.info(text).bold + "\n\n"
            } else {
                output += "\n## \(text)\n\n"
            }
        case 3:
            if colorize {
                output += "\n" + TerminalUI.bold(text) + "\n\n"
            } else {
                output += "\n### \(text)\n\n"
            }
        default:
            output += "\n\(text)\n\n"
        }
    }
    
    mutating func visitParagraph(_ paragraph: Paragraph) -> () {
        for child in paragraph.children {
            visitAny(child)
        }
        output += "\n\n"
    }
    
    mutating func visitCodeBlock(_ codeBlock: CodeBlock) -> () {
        let code = codeBlock.code
        
        if let language = codeBlock.language?.lowercased() {
            // Syntax highlighted code
            if colorize {
                output += "\n" + TerminalUI.dim("```\(language)") + "\n"
                let highlighted = SyntaxHighlighter.highlight(code, language: .from(string: language))
                output += highlighted
                output += TerminalUI.dim("```") + "\n\n"
            } else {
                output += "\n```\(language)\n\(code)\n```\n\n"
            }
        } else {
            // Plain code block
            if colorize {
                output += "\n" + TerminalUI.dim("```") + "\n"
                output += code
                output += TerminalUI.dim("```") + "\n\n"
            } else {
                output += "\n```\n\(code)\n```\n\n"
            }
        }
    }
    
    mutating func visitBlockQuote(_ blockQuote: BlockQuote) -> () {
        for child in blockQuote.children {
            if colorize {
                output += TerminalUI.dim("│ ")
            } else {
                output += "> "
            }
            visitAny(child)
        }
    }
    
    mutating func visitUnorderedList(_ unorderedList: UnorderedList) -> () {
        listLevel += 1
        for item in unorderedList.listItems {
            visitListItem(item, ordered: false)
        }
        listLevel -= 1
        if listLevel == 0 {
            output += "\n"
        }
    }
    
    mutating func visitOrderedList(_ orderedList: OrderedList) -> () {
        listLevel += 1
        listItemNumber = Int(orderedList.startIndex)
        for item in orderedList.listItems {
            visitListItem(item, ordered: true)
            listItemNumber += 1
        }
        listLevel -= 1
        if listLevel == 0 {
            output += "\n"
        }
    }
    
    mutating func visitListItem(_ listItem: ListItem, ordered: Bool) {
        let indent = String(repeating: "  ", count: listLevel - 1)
        let bullet = ordered ? "\(listItemNumber)." : "•"
        
        output += "\(indent)\(bullet) "
        
        for child in listItem.children {
            visitAny(child)
        }
        output += "\n"
    }
    
    // MARK: - Inline Elements
    
    mutating func visitText(_ text: Text) -> () {
        output += text.string
    }
    
    mutating func visitStrong(_ strong: Strong) -> () {
        let text = extractPlainText(from: strong)
        if colorize {
            output += TerminalUI.bold(text)
        } else {
            output += "**\(text)**"
        }
    }
    
    mutating func visitEmphasis(_ emphasis: Emphasis) -> () {
        let text = extractPlainText(from: emphasis)
        if colorize {
            output += TerminalUI.italic(text)
        } else {
            output += "*\(text)*"
        }
    }
    
    mutating func visitInlineCode(_ inlineCode: InlineCode) -> () {
        if colorize {
            output += TerminalUI.code(inlineCode.code)
        } else {
            output += "`\(inlineCode.code)`"
        }
    }
    
    mutating func visitLink(_ link: Link) -> () {
        let text = extractPlainText(from: link)
        if let destination = link.destination {
            if colorize {
                output += TerminalUI.info(text).underline
                output += TerminalUI.dim(" [\(destination)]")
            } else {
                output += "\(text) (\(destination))"
            }
        } else {
            output += text
        }
    }
    
    mutating func visitSoftBreak(_ softBreak: SoftBreak) -> () {
        output += " "
    }
    
    mutating func visitLineBreak(_ lineBreak: LineBreak) -> () {
        output += "\n"
    }
    
    // MARK: - Helper
    
    mutating func visitAny(_ markup: Markup) {
        markup.accept(&self)
    }
    
    private func extractPlainText(from markup: Markup) -> String {
        var text = ""
        for child in markup.children {
            if let textNode = child as? Text {
                text += textNode.string
            } else {
                text += extractPlainText(from: child)
            }
        }
        return text
    }
}

// MARK: - Syntax Highlighter

public struct SyntaxHighlighter: Sendable {
    
    /// Supported programming languages
    public enum Language: String, Sendable {
        case swift
        case python
        case javascript
        case json
        case bash
        case plain
        
        static func from(string: String) -> Language {
            switch string.lowercased() {
            case "swift": return .swift
            case "python", "py": return .python
            case "javascript", "js", "typescript", "ts": return .javascript
            case "json": return .json
            case "bash", "sh", "shell": return .bash
            default: return .plain
            }
        }
    }
    
    /// Highlight code with syntax coloring
    public static func highlight(_ code: String, language: Language) -> String {
        guard TerminalUI.colorsEnabled else {
            return code
        }
        
        switch language {
        case .swift:
            return highlightSwift(code)
        case .python:
            return highlightPython(code)
        case .javascript:
            return highlightJavaScript(code)
        case .json:
            return highlightJSON(code)
        case .bash:
            return highlightBash(code)
        case .plain:
            return code
        }
    }
    
    // MARK: - Language-Specific Highlighters
    
    private static func highlightSwift(_ code: String) -> String {
        // Use Splash for Swift syntax highlighting
        let highlighter = Splash.SyntaxHighlighter(format: ANSIOutputFormat())
        return highlighter.highlight(code)
    }
    
    private static func highlightPython(_ code: String) -> String {
        // Simple regex-based highlighting for Python
        var highlighted = code
        
        // Keywords
        let keywords = ["def", "class", "import", "from", "return", "if", "else", "elif", "for", "while", "try", "except", "with", "as"]
        for keyword in keywords {
            highlighted = highlighted.replacingOccurrences(
                of: "\\b\(keyword)\\b",
                with: TerminalUI.bold(keyword),
                options: .regularExpression
            )
        }
        
        // Strings
        highlighted = highlighted.replacingOccurrences(
            of: "\"([^\"]+)\"",
            with: TerminalUI.success("\"$1\""),
            options: .regularExpression
        )
        
        return highlighted
    }
    
    private static func highlightJavaScript(_ code: String) -> String {
        var highlighted = code
        
        // Keywords
        let keywords = ["function", "const", "let", "var", "return", "if", "else", "for", "while", "class", "async", "await"]
        for keyword in keywords {
            highlighted = highlighted.replacingOccurrences(
                of: "\\b\(keyword)\\b",
                with: TerminalUI.bold(keyword),
                options: .regularExpression
            )
        }
        
        return highlighted
    }
    
    private static func highlightJSON(_ code: String) -> String {
        var highlighted = code
        
        // Keys (quoted strings before colon)
        highlighted = highlighted.replacingOccurrences(
            of: "\"([^\"]+)\":",
            with: TerminalUI.info("\"$1\"") + ":",
            options: .regularExpression
        )
        
        // String values
        highlighted = highlighted.replacingOccurrences(
            of: ":\\s*\"([^\"]+)\"",
            with: ": " + TerminalUI.success("\"$1\""),
            options: .regularExpression
        )
        
        return highlighted
    }
    
    private static func highlightBash(_ code: String) -> String {
        var highlighted = code
        
        // Commands
        let commands = ["echo", "cd", "ls", "grep", "cat", "git", "npm", "brew"]
        for command in commands {
            highlighted = highlighted.replacingOccurrences(
                of: "\\b\(command)\\b",
                with: TerminalUI.bold(command),
                options: .regularExpression
            )
        }
        
        return highlighted
    }
}

// MARK: - Splash ANSI Output Format

struct ANSIOutputFormat: Splash.OutputFormat {
    func makeBuilder() -> ANSIOutputBuilder {
        ANSIOutputBuilder()
    }
}

struct ANSIOutputBuilder: Splash.OutputBuilder {
    private var output = ""
    
    mutating func addToken(_ token: String, ofType type: TokenType) {
        switch type {
        case .keyword:
            output += TerminalUI.bold(token)
        case .string:
            output += TerminalUI.success(token)
        case .number:
            output += TerminalUI.info(token)
        case .comment:
            output += TerminalUI.dim(token)
        case .type:
            output += TerminalUI.warning(token)
        case .call:
            output += TerminalUI.info(token)
        case .property:
            output += token
        case .dotAccess:
            output += token
        case .preprocessing:
            output += TerminalUI.dim(token)
        default:
            output += token
        }
    }
    
    mutating func addPlainText(_ text: String) {
        output += text
    }
    
    mutating func addWhitespace(_ whitespace: String) {
        output += whitespace
    }
    
    func build() -> String {
        return output
    }
}
