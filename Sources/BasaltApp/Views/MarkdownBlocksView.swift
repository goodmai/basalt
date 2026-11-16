import AppKit
import Foundation
import Markdown
import SwiftUI

public enum MarkdownBlock: Sendable, Equatable {
    public enum ColumnAlignment: Sendable, Equatable {
        case leading
        case center
        case trailing
    }

    public struct TableModel: Sendable, Equatable {
        public var headers: [String]
        public var rows: [[String]]
        public var alignments: [ColumnAlignment]

        public init(headers: [String], rows: [[String]], alignments: [ColumnAlignment] = []) {
            (self.headers, self.rows, self.alignments) = (headers, rows, alignments)
        }
    }

    case heading(level: Int, text: String)
    case paragraph(String)
    case codeBlock(language: String?, code: String)
    case table(TableModel)
    case blockQuote(String)
    case list(ordered: Bool, start: Int, items: [String])
    case thematicBreak
}

public enum MarkdownBlockParser {
    private static let blockPrefixes = ["#", "> ", "- ", "* ", "1. "]
    private static let blockSubstrings = ["```", "~~~", "\n\n", "\n#", "\n> ", "\n- ", "\n* ", "\n1. "]

    public static func hasStructuredBlocks(_ source: String) -> Bool {
        if blockSubstrings.contains(where: source.contains) { return true }
        let trimmed = source.trimmingCharacters(in: .whitespaces)
        if blockPrefixes.contains(where: trimmed.hasPrefix) { return true }
        return source.contains("|") && source.contains("---")
    }

    public static func parse(_ source: String) -> [MarkdownBlock] {
        guard !source.isEmpty else { return [] }
        let document = Markdown::Document(parsing: source)
        return document.children.compactMap(convertBlock)
    }

    private static func convertBlock(_ markup: any Markup) -> MarkdownBlock? {
        switch markup {
        case let heading as Heading:
            return .heading(level: heading.level, text: inlineMarkdown(heading, collapseBreaks: true))
        case let paragraph as Paragraph:
            return .paragraph(inlineMarkdown(paragraph, collapseBreaks: false))
        case let block as CodeBlock:
            let language = block.language.flatMap { raw -> String? in
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            }
            let code = block.code.hasSuffix("\n") ? String(block.code.dropLast()) : block.code
            return .codeBlock(language: language, code: code)
        case let table as Markdown::Table:
            return .table(convertTable(table))
        case let quote as BlockQuote:
            let body = quote.children.map { inlineMarkdown($0, collapseBreaks: false) }.filter { !$0.isEmpty }.joined(separator: "\n")
            return .blockQuote(body)
        case let unordered as UnorderedList:
            let items = Array(unordered.listItems.map { listItemText($0) })
            return .list(ordered: false, start: 1, items: items)
        case let ordered as OrderedList:
            let items = Array(ordered.listItems.map { listItemText($0) })
            return .list(ordered: true, start: Int(ordered.startIndex), items: items)
        case is ThematicBreak:
            return .thematicBreak
        default:
            let fallback = inlineMarkdown(markup, collapseBreaks: false).trimmingCharacters(in: .whitespacesAndNewlines)
            return fallback.isEmpty ? nil : .paragraph(fallback)
        }
    }

    private static func convertTable(_ table: Markdown::Table) -> MarkdownBlock.TableModel {
        let headers = Array(table.head.cells.map { inlineMarkdown($0, collapseBreaks: true) })
        let rows = Array(table.body.rows.map { row in Array(row.cells.map { inlineMarkdown($0, collapseBreaks: true) }) })
        let alignments = Array(table.columnAlignments.map { alignment -> MarkdownBlock.ColumnAlignment in
            switch alignment {
            case .center?: .center
            case .right?: .trailing
            case .left?, nil: .leading
            }
        })
        return MarkdownBlock.TableModel(headers: headers, rows: rows, alignments: alignments)
    }

    private static func listItemText(_ item: ListItem) -> String {
        item.children.map { inlineMarkdown($0, collapseBreaks: true) }.filter { !$0.isEmpty }.joined(separator: " ")
    }

    private static func inlineMarkdown(_ markup: any Markup, collapseBreaks: Bool) -> String {
        var result = ""
        for child in markup.children {
            switch child {
            case let text as Markdown::Text:
                result += text.string
            case let strong as Strong:
                result += "**\(inlineMarkdown(strong, collapseBreaks: collapseBreaks))**"
            case let emphasis as Emphasis:
                result += "*\(inlineMarkdown(emphasis, collapseBreaks: collapseBreaks))*"
            case let strikethrough as Strikethrough:
                result += "~~\(inlineMarkdown(strikethrough, collapseBreaks: collapseBreaks))~~"
            case let code as InlineCode:
                result += "`\(code.code)`"
            case let link as Markdown::Link:
                let label = inlineMarkdown(link, collapseBreaks: collapseBreaks)
                result += link.destination.map { "[\(label)](\($0))" } ?? label
            case is SoftBreak:
                result += collapseBreaks ? " " : "\n"
            case is LineBreak:
                result += "\n"
            default:
                result += inlineMarkdown(child, collapseBreaks: collapseBreaks)
            }
        }
        return result
    }
}

public struct MarkdownBlocksView: View {
    public let blocks: [MarkdownBlock]
    public var onCopy: @Sendable (String) -> Void

    public init(
        markdown: String,
        onCopy: @escaping @Sendable (String) -> Void = { text in
            Task { @MainActor in
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            }
        }
    ) {
        self.blocks = MarkdownBlockParser.parse(markdown)
        self.onCopy = onCopy
    }

    public init(
        blocks: [MarkdownBlock],
        onCopy: @escaping @Sendable (String) -> Void = { text in
            Task { @MainActor in
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            }
        }
    ) {
        (self.blocks, self.onCopy) = (blocks, onCopy)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            inlineText(text)
                .font(Self.headingFont(level))
                .fontWeight(.semibold)
        case .paragraph(let text):
            inlineText(text)
        case .codeBlock(let language, let code):
            MarkdownCodeCard(language: language, code: code, onCopy: onCopy)
        case .table(let table):
            MarkdownTableGrid(table: table)
        case .blockQuote(let text):
            HStack(spacing: 10) {
                Rectangle().fill(.secondary.opacity(0.45)).frame(width: 3)
                inlineText(text).foregroundStyle(.secondary)
            }
        case .list(let ordered, let start, let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { offset, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        SwiftUI::Text(verbatim: ordered ? "\(start + offset)." : "•").foregroundStyle(.secondary)
                        inlineText(item)
                    }
                }
            }
        case .thematicBreak:
            Divider()
        }
    }

    private func inlineText(_ markdown: String) -> some View {
        let attributed = (try? AttributedString(markdown: markdown, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(markdown)
        return SwiftUI::Text(attributed).textSelection(.enabled)
    }

    private static func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: .title2
        case 2: .title3
        case 3: .headline
        default: .subheadline
        }
    }
}

struct MarkdownCodeCard: View {
    let language: String?
    let code: String
    let onCopy: @Sendable (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                SwiftUI::Text(verbatim: language ?? String(localized: "Code"))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Copy", systemImage: "doc.on.doc") { onCopy(code) }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.45))

            ScrollView(.horizontal, showsIndicators: false) {
                SwiftUI::Text(verbatim: code)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(.background.secondary, in: .rect(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1)
        }
    }
}

struct MarkdownTableGrid: View {
    let table: MarkdownBlock.TableModel

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                if !table.headers.isEmpty {
                    GridRow {
                        ForEach(Array(table.headers.enumerated()), id: \.offset) { col, header in
                            cell(header, column: col).font(.callout.weight(.semibold))
                        }
                    }
                    Divider()
                }
                ForEach(Array(table.rows.enumerated()), id: \.offset) { _, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { col, value in
                            cell(value, column: col).font(.callout)
                        }
                    }
                }
            }
            .padding(10)
        }
        .background(.background.secondary, in: .rect(cornerRadius: 10))
    }

    private func cell(_ text: String, column: Int) -> some View {
        let attributed = (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(text)
        let alignment: Alignment = switch column < table.alignments.count ? table.alignments[column] : .leading {
        case .leading: .leading
        case .center: .center
        case .trailing: .trailing
        }
        return SwiftUI::Text(attributed)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: alignment)
    }
}
