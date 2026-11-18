import BasaltCore
import Foundation

#if canImport(FoundationModels, _version: 2)
import FoundationModels
#endif

public enum BasaltFoundationSchemaAdapter {
    public static func validatedJSONSchema(_ rawSchema: String, strict: Bool = false) throws(BasaltError) -> String {
        do {
            return try JSONSchemaGate.prepare(rawSchema, strict: strict).schema
        } catch {
            throw .invalidRequestStructure(details: "invalid JSON schema: \(error.message)")
        }
    }

    public static func encodeToolCallingEnvelope(tools: [ToolDefinition]) throws(BasaltError) -> String {
        guard !tools.isEmpty else {
            throw .invalidRequestStructure(details: "tool-calling envelope requires at least one tool")
        }
        var hoistedDefs: [String: Any] = [:]
        var oneOf: [[String: Any]] = []
        oneOf.reserveCapacity(tools.count)
        for tool in tools {
            let paramsObject = try decodedParametersObject(for: tool)
            var rewritten = rewriteDefsRefs(in: paramsObject, toolName: tool.name)
            if var dictionary = rewritten as? [String: Any],
               let defs = dictionary.removeValue(forKey: "$defs") as? [String: Any]
            {
                for (key, value) in defs {
                    hoistedDefs["\(tool.name)__\(key)"] = value
                }
                rewritten = dictionary
            }
            oneOf.append([
                "type": "object",
                "required": ["name", "arguments"],
                "additionalProperties": false,
                "properties": [
                    "name": ["const": tool.name],
                    "arguments": rewritten,
                ],
            ])
        }
        var envelope: [String: Any] = ["oneOf": oneOf]
        if !hoistedDefs.isEmpty {
            envelope["$defs"] = hoistedDefs
        }
        return try serializeJSONObject(envelope)
    }

    public static func encodeToolCallingStructuralTag(tools: [ToolDefinition]) throws(BasaltError) -> String {
        guard !tools.isEmpty else {
            throw .invalidRequestStructure(details: "tool-calling structural tag requires at least one tool")
        }
        var toolTags: [[String: Any]] = []
        toolTags.reserveCapacity(tools.count)
        for tool in tools {
            let paramsObject = try decodedParametersObject(for: tool)
            let nameData = (try? JSONSerialization.data(withJSONObject: tool.name, options: [.fragmentsAllowed])) ?? Data()
            let escapedName = String(data: nameData, encoding: .utf8) ?? "\"\(tool.name)\""
            toolTags.append([
                "type": "tag",
                "begin": "{\"name\": \(escapedName), \"arguments\": ",
                "content": [
                    "type": "json_schema",
                    "json_schema": paramsObject,
                ],
                "end": ["}"],
            ])
        }
        let perToolOr: [String: Any] = ["type": "or", "elements": toolTags]
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "tag",
                        "begin": "<tool_call>\n",
                        "content": perToolOr,
                        "end": ["\n</tool_call>"],
                    ],
                    perToolOr,
                ],
            ],
        ]
        return try serializeJSONObject(structuralTag)
    }

    public static func parseEmbeddedToolCalls(
        from text: String,
        declaredTools: Set<String>? = nil
    ) -> [ToolCallOutput] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let blocks = extractedToolCallBlocks(from: trimmed)
        let candidates = blocks.isEmpty ? [stripMarkdownFence(trimmed)] : blocks
        var parsed: [ToolCallOutput] = []
        for (index, candidate) in candidates.enumerated() {
            for call in decodeCandidateCalls(candidate, baseIndex: index, declaredTools: declaredTools) {
                parsed.append(call)
            }
        }
        return parsed
    }

    public static func stripMarkdownFence(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```"), let firstNewline = trimmed.firstIndex(of: "\n") else {
            return trimmed
        }
        var body = String(trimmed[trimmed.index(after: firstNewline)...])
        if body.hasSuffix("```") {
            body = String(body.dropLast(3))
        }
        return body.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

extension BasaltFoundationSchemaAdapter {
    private static func decodedParametersObject(for tool: ToolDefinition) throws(BasaltError) -> Any {
        guard let json = tool.parametersJSON, !json.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return ["type": "object", "properties": [String: Any](), "additionalProperties": false] as [String: Any]
        }
        guard let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)) else {
            throw .invalidRequestStructure(details: "tool '\(tool.name)' has invalid parametersJSON")
        }
        return object
    }

    private static func serializeJSONObject(_ object: Any) throws(BasaltError) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else {
            throw .invalidRequestStructure(details: "failed to serialize schema JSON object")
        }
        return text
    }

    private static func rewriteDefsRefs(in value: Any, toolName: String) -> Any {
        switch value {
        case let object as [String: Any]:
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, nested) in object {
                if key == "$ref", let ref = nested as? String, ref.hasPrefix("#/$defs/") {
                    result[key] = "#/$defs/\(toolName)__" + ref.dropFirst("#/$defs/".count)
                } else {
                    result[key] = rewriteDefsRefs(in: nested, toolName: toolName)
                }
            }
            return result
        case let array as [Any]:
            return array.map { rewriteDefsRefs(in: $0, toolName: toolName) }
        default:
            return value
        }
    }

    private static func extractedToolCallBlocks(from text: String) -> [String] {
        var blocks: [String] = []
        var remainder = text[...]
        while let openRange = remainder.range(of: "<tool_call>") {
            let afterOpen = remainder[openRange.upperBound...]
            guard let closeRange = afterOpen.range(of: "</tool_call>") else { break }
            let payload = afterOpen[..<closeRange.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            if !payload.isEmpty { blocks.append(payload) }
            remainder = afterOpen[closeRange.upperBound...]
        }
        return blocks
    }

    private static func decodeCandidateCalls(
        _ candidate: String,
        baseIndex: Int,
        declaredTools: Set<String>?
    ) -> [ToolCallOutput] {
        guard (candidate.hasPrefix("{") || candidate.hasPrefix("[")),
              let object = try? JSONSerialization.jsonObject(with: Data(candidate.utf8))
        else {
            return []
        }
        let entries: [[String: Any]]
        if let single = object as? [String: Any] {
            entries = [single]
        } else if let list = object as? [[String: Any]] {
            entries = list
        } else {
            return []
        }
        var results: [ToolCallOutput] = []
        for (offset, entry) in entries.enumerated() {
            let functionDict = (entry["function"] as? [String: Any]) ?? entry
            guard let name = functionDict["name"] as? String, !name.isEmpty else { continue }
            if let declaredTools, !declaredTools.contains(name) { continue }
            let callID = (entry["id"] as? String).flatMap { $0.isEmpty ? nil : $0 }
                ?? "call_\(baseIndex)_\(offset)"
            let argumentsText = stringifyArguments(functionDict["arguments"])
            results.append(ToolCallOutput(id: callID, name: name, arguments: argumentsText))
        }
        return results
    }

    private static func stringifyArguments(_ rawArguments: Any?) -> String {
        if let text = rawArguments as? String { return text }
        guard let rawArguments, JSONSerialization.isValidJSONObject(rawArguments),
              let data = try? JSONSerialization.data(withJSONObject: rawArguments, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return text
    }
}

#if canImport(FoundationModels, _version: 2)
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension BasaltFoundationSchemaAdapter {
    public static func encodeGenerationSchema(_ schema: GenerationSchema) throws -> String {
        let data = try JSONEncoder().encode(schema)
        guard let rawJSON = String(data: data, encoding: .utf8) else {
            throw LanguageModelError.unsupportedGenerationGuide(
                .init(schemaName: schema.name, debugDescription: "GenerationSchema UTF-8 encoding failed")
            )
        }
        do {
            return try JSONSchemaGate.prepare(rawJSON, strict: false).schema
        } catch {
            throw LanguageModelError.unsupportedGenerationGuide(
                .init(schemaName: schema.name, debugDescription: error.message)
            )
        }
    }

    public static func convertToolDefinition(_ tool: Transcript.ToolDefinition) throws -> ToolDefinition {
        let paramsData = try JSONEncoder().encode(tool.parameters)
        let paramsJSON = String(data: paramsData, encoding: .utf8)
        return ToolDefinition(name: tool.name, description: tool.description, parametersJSON: paramsJSON)
    }

    public static func decodeGenerable<Content: Generable>(
        _ type: Content.Type = Content.self,
        from json: String
    ) throws -> Content {
        let cleaned = stripMarkdownFence(json)
        let generated = try GeneratedContent(json: cleaned)
        return try Content(generated)
    }
}
#endif
