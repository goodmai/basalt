import Foundation

enum OTSLTable {
    enum Token: String {
        case newline = "<nl>", filled = "<fcel>", empty = "<ecel>", left = "<lcel>", up = "<ucel>", cross = "<xcel>"
    }

    struct Cell {
        let token: Token
        var text = ""
    }

    static func html(_ otsl: String) -> String {
        if otsl.hasPrefix("<table"), otsl.hasSuffix("</table>") { return otsl }
        let rows = grid(otsl)
        guard !rows.isEmpty else { return otsl }
        var html = "<table>"
        for (r, row) in rows.enumerated() {
            html += "<tr>"
            for (c, cell) in row.enumerated() where cell.token == .filled || cell.token == .empty {
                let columns = 1 + row[(c + 1)...].prefix { $0.token == .left || $0.token == .cross }.count
                let spanned = rows[(r + 1)...].prefix { $0[c].token == .up || $0[c].token == .cross }.count
                html += "<td" + (spanned > 0 ? " rowspan=\"\(spanned + 1)\"" : "") + (columns > 1 ? " colspan=\"\(columns)\"" : "")
                html += ">" + escaped(cell.text.trimmingCharacters(in: .whitespacesAndNewlines)) + "</td>"
            }
            html += "</tr>"
        }
        return html + "</table>"
    }

    static func grid(_ otsl: String) -> [[Cell]] {
        var rows: [[Cell]] = [[]]
        var cursor = otsl.startIndex
        for match in otsl.matches(of: #/<(?:nl|fcel|ecel|lcel|ucel|xcel)>/#) {
            append(otsl[cursor ..< match.range.lowerBound], to: &rows)
            cursor = match.range.upperBound
            guard let token = Token(rawValue: String(match.output)) else { continue }
            if token == .newline { rows.append([]) } else { rows[rows.count - 1].append(Cell(token: token)) }
        }
        append(otsl[cursor...], to: &rows)
        let filled = rows.filter { !$0.isEmpty }
        let width = filled.map(\.count).max() ?? 0
        return filled.map { $0 + Array(repeating: Cell(token: .empty), count: width - $0.count) }
    }

    static func append(_ text: Substring, to rows: inout [[Cell]]) {
        guard let last = rows[rows.count - 1].indices.last, rows[rows.count - 1][last].token == .filled else { return }
        rows[rows.count - 1][last].text += text
    }

    static func escaped(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#x27;")
    }
}
