import Foundation
import BasaltCore

struct MultipartForm: Sendable {
    struct Part: Sendable, Equatable {
        let name: String
        let filename: String?
        let contentType: String?
        let body: Data
    }

    static let maxParts = 16
    static let maxHeaderBytes = 8 * 1024
    static let maxTextBytes = 4 * 1024

    let parts: [String: Part]

    init(_ body: Data, boundary: String) throws(BasaltError) {
        guard (1 ... 70).contains(boundary.utf8.count), !boundary.contains(where: \.isNewline) else {
            throw Self.malformed("the multipart boundary must be 1 to 70 characters on one line")
        }
        let body = body.startIndex == 0 ? body : Data(body)
        let delimiter = Data("\r\n--\(boundary)".utf8)
        guard let first = body.range(of: delimiter.dropFirst(2)) else { throw Self.malformed("the body holds no multipart boundary") }
        var cursor = first.upperBound
        var parts: [String: Part] = [:]
        while !body[cursor...].starts(with: Data("--".utf8)) {
            guard body[cursor...].starts(with: Data("\r\n".utf8)) else { throw Self.malformed("a multipart boundary is not followed by CRLF") }
            let headerStart = cursor + 2
            let headerWindow = headerStart ..< min(body.endIndex, headerStart + Self.maxHeaderBytes)
            guard let headerEnd = body.range(of: Data("\r\n\r\n".utf8), in: headerWindow) else {
                throw Self.malformed("a multipart part has no header end within \(Self.maxHeaderBytes) bytes")
            }
            guard let next = body.range(of: delimiter, in: headerEnd.upperBound ..< body.endIndex) else {
                throw Self.malformed("a multipart part is not closed by the boundary")
            }
            let part = try Self.part(headers: String(decoding: body[headerStart ..< headerEnd.lowerBound], as: UTF8.self),
                                     body: body.subdata(in: headerEnd.upperBound ..< next.lowerBound))
            guard parts[part.name] == nil else { throw Self.malformed("the form field \(part.name.prefix(32)) appears twice") }
            parts[part.name] = part
            guard parts.count <= Self.maxParts else { throw Self.malformed("a form carries at most \(Self.maxParts) fields") }
            cursor = next.upperBound
        }
        self.parts = parts
    }

    func text(_ name: String) throws(BasaltError) -> String? {
        guard let part = parts[name] else { return nil }
        guard part.body.count <= Self.maxTextBytes, let text = String(data: part.body, encoding: .utf8) else {
            throw Self.malformed("the form field \(name) must be UTF-8 text of at most \(Self.maxTextBytes) bytes")
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func part(headers: String, body: Data) throws(BasaltError) -> Part {
        var disposition: (kind: String, parameters: [String: String])?
        var contentType: String?
        for line in headers.split(whereSeparator: \.isNewline) {
            guard let colon = line.firstIndex(of: ":") else { throw malformed("a multipart header line has no colon") }
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            switch line[..<colon].trimmingCharacters(in: .whitespaces).lowercased() {
            case "content-disposition": disposition = parameters(value)
            case "content-type": contentType = value.lowercased()
            default: break
            }
        }
        guard let disposition, disposition.kind == "form-data", let name = disposition.parameters["name"], !name.isEmpty else {
            throw malformed("every multipart part needs Content-Disposition: form-data with a name")
        }
        return Part(name: name, filename: disposition.parameters["filename"], contentType: contentType, body: body)
    }

    static func parameters(_ value: String) -> (kind: String, parameters: [String: String]) {
        var fields: [String] = [""]
        var quoted = false
        for character in value {
            if character == "\"" { quoted.toggle() }
            if character == ";", !quoted { fields.append("") } else { fields[fields.count - 1].append(character) }
        }
        var parameters: [String: String] = [:]
        for field in fields.dropFirst() {
            guard let equals = field.firstIndex(of: "=") else { continue }
            var value = field[field.index(after: equals)...].trimmingCharacters(in: .whitespaces)
            if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") { value = String(value.dropFirst().dropLast()) }
            parameters[field[..<equals].trimmingCharacters(in: .whitespaces).lowercased()] = value
        }
        return (fields[0].trimmingCharacters(in: .whitespaces).lowercased(), parameters)
    }

    static func malformed(_ details: String) -> BasaltError {
        .invalidRequestStructure(details: details)
    }
}
