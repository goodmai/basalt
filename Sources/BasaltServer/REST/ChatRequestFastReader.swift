import Foundation
import Hummingbird
import NIOCore
import yyjson
import BasaltCore

struct ChatRequestFastReader {
    struct DecodeError: Error, LocalizedError, Sendable, Equatable {
        let message: String
        var errorDescription: String? {
            DecodingError.dataCorrupted(.init(codingPath: [], debugDescription: message)).localizedDescription
        }
    }

    struct Parsed {
        let dto: OpenAIController.ChatCompletionRequest
        let antiRepetition: Result<ChatPipeline.AntiRepetition, DecodeError>
        let tools: Result<OpenAIController.RequestedTools, OpenAIController.UndeclaredForcedTool>
        let responseFormat: Result<ResponseFormat?, OpenAIController.InvalidResponseFormat>
    }

    static func read(_ bytes: RawSpan) throws(DecodeError) -> Parsed {
        let outcome: Result<Parsed, DecodeError> = bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress, raw.count > 0 else {
                return .failure(DecodeError(message: "empty request body"))
            }
            var err = yyjson_read_err()
            let ptr = UnsafeMutableRawPointer(mutating: base).assumingMemoryBound(to: CChar.self)
            guard let doc = yyjson_read_opts(ptr, raw.count, YYJSON_READ_ALLOW_BOM, nil, &err) else {
                let detail = err.msg.map { String(cString: $0) } ?? "invalid JSON"
                return .failure(DecodeError(message: "\(detail) at byte \(err.pos)"))
            }
            defer { yyjson_doc_free(doc) }
            guard let root = yyjson_doc_get_root(doc), yyjson_is_obj(root) else {
                return .failure(DecodeError(message: "expected JSON object"))
            }
            return Result<Parsed, DecodeError> { () throws(DecodeError) in try parseRoot(root) }
        }
        return try outcome.get()
    }

    static func read(_ bytes: Span<UInt8>) throws(DecodeError) -> Parsed {
        try read(bytes.bytes)
    }

    static func read(_ buffer: ByteBuffer) throws(DecodeError) -> Parsed {
        try read(buffer.readableBytesSpan)
    }

    static func read(_ data: Data) throws(DecodeError) -> Parsed {
        try read(data.span)
    }

    private static func parseRoot(_ root: UnsafeMutablePointer<yyjson_val>) throws(DecodeError) -> Parsed {
        let dto = try decodeDTO(root)
        let antiRepetition = Result<ChatPipeline.AntiRepetition, DecodeError> { () throws(DecodeError) in
            try decodeAntiRepetition(root)
        }
        let tools = Result<OpenAIController.RequestedTools, OpenAIController.UndeclaredForcedTool> {
            () throws(OpenAIController.UndeclaredForcedTool) in
            try requestedTools(in: root)
        }
        let format = Result<ResponseFormat?, OpenAIController.InvalidResponseFormat> {
            () throws(OpenAIController.InvalidResponseFormat) in
            try responseFormat(in: root)
        }
        return Parsed(dto: dto, antiRepetition: antiRepetition, tools: tools, responseFormat: format)
    }

    private static func decodeDTO(
        _ root: UnsafeMutablePointer<yyjson_val>
    ) throws(DecodeError) -> OpenAIController.ChatCompletionRequest {
        var cursor = ObjectCursor(root)
        let model = try cursor.string("model")
        let messages = try decodeMessages(cursor.raw("messages"))
        let maxTokens = try cursor.int("max_tokens")
        let temperature = try cursor.double("temperature")
        let topP = try cursor.double("top_p")
        let minP = try cursor.double("min_p")
        let dynamicTemperature = try cursor.bool("dynamic_temperature")
        let stream = try cursor.bool("stream")
        let reasoningEffort = try cursor.string("reasoning_effort")
        let topK = try cursor.int("top_k")
        let presencePenalty = try cursor.double("presence_penalty")
        let repetitionPenalty = try cursor.double("repetition_penalty")
        let thinkingBudget = try cursor.int("thinking_budget")
        let speculativeLookup = try cursor.bool("speculative_lookup")
        let grammarJumpForward = try cursor.bool("grammar_jump_forward")
        let streamOptions = try decodeStreamOptions(cursor.nonNull("stream_options"))
        let enableThinking = try cursor.bool("enable_thinking")
        return OpenAIController.ChatCompletionRequest(
            model: model, messages: messages, maxTokens: maxTokens, temperature: temperature,
            topP: topP, minP: minP, dynamicTemperature: dynamicTemperature, stream: stream,
            reasoningEffort: reasoningEffort, topK: topK, presencePenalty: presencePenalty,
            repetitionPenalty: repetitionPenalty, thinkingBudget: thinkingBudget,
            speculativeLookup: speculativeLookup, grammarJumpForward: grammarJumpForward,
            streamOptions: streamOptions, enableThinking: enableThinking)
    }

    private static func decodeAntiRepetition(
        _ root: UnsafeMutablePointer<yyjson_val>
    ) throws(DecodeError) -> ChatPipeline.AntiRepetition {
        var cursor = ObjectCursor(root)
        var out = ChatPipeline.AntiRepetition()
        out.dryMultiplier = try cursor.double("dry_multiplier")
        out.dryBase = try cursor.double("dry_base")
        out.dryAllowedLength = try cursor.int("dry_allowed_length")
        out.drySequenceBreakers = try cursor.stringArray("dry_sequence_breakers")
        out.xtcThreshold = try cursor.double("xtc_threshold")
        out.xtcProbability = try cursor.double("xtc_probability")
        out.logprobs = try cursor.bool("logprobs")
        out.topLogprobs = try cursor.int("top_logprobs")
        out.nProbs = try cursor.int("n_probs")
        out.xrayTrace = try cursor.bool("xray_trace")
        return out
    }
}

private struct ObjectCursor {
    var iter: yyjson_obj_iter

    init(_ obj: UnsafeMutablePointer<yyjson_val>) {
        iter = yyjson_obj_iter_with(obj)
    }

    mutating func raw(_ key: StaticString) -> UnsafeMutablePointer<yyjson_val>? {
        let ptr = UnsafeRawPointer(key.utf8Start).assumingMemoryBound(to: CChar.self)
        return yyjson_obj_iter_getn(&iter, ptr, key.utf8CodeUnitCount)
    }

    mutating func nonNull(_ key: StaticString) -> UnsafeMutablePointer<yyjson_val>? {
        guard let val = raw(key), !yyjson_is_null(val) else { return nil }
        return val
    }

    mutating func string(_ key: StaticString) throws(ChatRequestFastReader.DecodeError) -> String? {
        guard let val = nonNull(key) else { return nil }
        guard yyjson_is_str(val) else {
            throw ChatRequestFastReader.DecodeError(message: "expected string for '\(key)'")
        }
        return Self.uncheckedString(val)
    }

    mutating func bool(_ key: StaticString) throws(ChatRequestFastReader.DecodeError) -> Bool? {
        guard let val = nonNull(key) else { return nil }
        guard yyjson_is_bool(val) else {
            throw ChatRequestFastReader.DecodeError(message: "expected boolean for '\(key)'")
        }
        return yyjson_get_bool(val)
    }

    mutating func int(_ key: StaticString) throws(ChatRequestFastReader.DecodeError) -> Int? {
        guard let val = nonNull(key) else { return nil }
        if yyjson_is_uint(val) {
            let u = yyjson_get_uint(val)
            guard u <= UInt64(Int.max) else {
                throw ChatRequestFastReader.DecodeError(message: "integer overflow for '\(key)'")
            }
            return Int(u)
        }
        if yyjson_is_sint(val) {
            let s = yyjson_get_sint(val)
            guard s >= Int64(Int.min), s <= Int64(Int.max) else {
                throw ChatRequestFastReader.DecodeError(message: "integer overflow for '\(key)'")
            }
            return Int(s)
        }
        throw ChatRequestFastReader.DecodeError(message: "expected integer for '\(key)'")
    }

    mutating func double(_ key: StaticString) throws(ChatRequestFastReader.DecodeError) -> Double? {
        guard let val = nonNull(key) else { return nil }
        guard yyjson_is_num(val) else {
            throw ChatRequestFastReader.DecodeError(message: "expected number for '\(key)'")
        }
        return yyjson_get_num(val)
    }

    mutating func stringArray(_ key: StaticString) throws(ChatRequestFastReader.DecodeError) -> [String]? {
        guard let val = nonNull(key) else { return nil }
        guard yyjson_is_arr(val) else {
            throw ChatRequestFastReader.DecodeError(message: "expected string array for '\(key)'")
        }
        var out: [String] = []
        out.reserveCapacity(yyjson_arr_size(val))
        var arrIter = yyjson_arr_iter_with(val)
        while let elem = yyjson_arr_iter_next(&arrIter) {
            guard yyjson_is_str(elem) else {
                throw ChatRequestFastReader.DecodeError(message: "expected string element in '\(key)'")
            }
            out.append(Self.uncheckedString(elem))
        }
        return out
    }

    static func uncheckedString(_ val: UnsafeMutablePointer<yyjson_val>) -> String {
        guard let ptr = yyjson_get_str(val) else { return "" }
        return String(decoding: UnsafeRawBufferPointer(start: ptr, count: yyjson_get_len(val)), as: UTF8.self)
    }
}

extension ChatRequestFastReader {
    fileprivate static func decodeStreamOptions(
        _ val: UnsafeMutablePointer<yyjson_val>?
    ) throws(DecodeError) -> OpenAIController.ChatCompletionRequest.StreamOptions? {
        guard let val else { return nil }
        guard yyjson_is_obj(val) else {
            throw DecodeError(message: "expected object for 'stream_options'")
        }
        var cursor = ObjectCursor(val)
        return OpenAIController.ChatCompletionRequest.StreamOptions(includeUsage: try cursor.bool("include_usage"))
    }

    fileprivate static func decodeMessages(
        _ val: UnsafeMutablePointer<yyjson_val>?
    ) throws(DecodeError) -> [OpenAIController.ChatMessage] {
        guard let val, yyjson_is_arr(val) else {
            throw DecodeError(message: "missing or invalid 'messages' array")
        }
        var out: [OpenAIController.ChatMessage] = []
        out.reserveCapacity(yyjson_arr_size(val))
        var iter = yyjson_arr_iter_with(val)
        while let item = yyjson_arr_iter_next(&iter) {
            out.append(try decodeMessage(item))
        }
        return out
    }

    private static func decodeMessage(
        _ val: UnsafeMutablePointer<yyjson_val>
    ) throws(DecodeError) -> OpenAIController.ChatMessage {
        guard yyjson_is_obj(val) else {
            throw DecodeError(message: "expected object in 'messages'")
        }
        var cursor = ObjectCursor(val)
        let type = try cursor.string("type")
        let role = try cursor.string("role")
        let content = try decodeContent(cursor.nonNull("content"))
        let name = try cursor.string("name")
        let arguments = try cursor.string("arguments")
        let input = try cursor.string("input")
        let output = try decodeContent(cursor.nonNull("output"))
        let toolCalls = try decodeToolCalls(cursor.nonNull("tool_calls"))
        let callId = try cursor.string("call_id")
        let toolCallId = try cursor.string("tool_call_id")
        let reasoning: String?
        if let primary = try cursor.string("reasoning") {
            reasoning = primary
        } else {
            reasoning = try cursor.string("reasoning_content")
        }
        return OpenAIController.ChatMessage(
            type: type, role: role, content: content, name: name, arguments: arguments,
            input: input, output: output, toolCalls: toolCalls,
            callId: callId, toolCallId: toolCallId, reasoning: reasoning)
    }

    private static func decodeContent(
        _ val: UnsafeMutablePointer<yyjson_val>?
    ) throws(DecodeError) -> OpenAIController.MessageContent? {
        guard let val else { return nil }
        if yyjson_is_str(val) {
            return .text(ObjectCursor.uncheckedString(val))
        }
        guard yyjson_is_arr(val) else {
            throw DecodeError(message: "expected string or array for message content")
        }
        var blocks: [OpenAIController.MessageContent.ContentBlock] = []
        blocks.reserveCapacity(yyjson_arr_size(val))
        var iter = yyjson_arr_iter_with(val)
        while let elem = yyjson_arr_iter_next(&iter) {
            blocks.append(try decodeContentBlock(elem))
        }
        return .blocks(blocks)
    }

    private static func decodeContentBlock(
        _ val: UnsafeMutablePointer<yyjson_val>
    ) throws(DecodeError) -> OpenAIController.MessageContent.ContentBlock {
        guard yyjson_is_obj(val) else {
            throw DecodeError(message: "expected content block object")
        }
        var cursor = ObjectCursor(val)
        let type = try cursor.string("type")
        let text = try cursor.string("text")
        let imageUrl = decodeImageURL(cursor.nonNull("image_url"))
        return OpenAIController.MessageContent.ContentBlock(type: type, text: text, imageUrl: imageUrl)
    }

    private static func decodeImageURL(
        _ val: UnsafeMutablePointer<yyjson_val>?
    ) -> OpenAIController.MessageContent.ImageURL? {
        guard let val else { return nil }
        if yyjson_is_str(val) {
            return OpenAIController.MessageContent.ImageURL(url: ObjectCursor.uncheckedString(val))
        }
        if yyjson_is_obj(val) {
            var cursor = ObjectCursor(val)
            return OpenAIController.MessageContent.ImageURL(url: try? cursor.string("url"))
        }
        return OpenAIController.MessageContent.ImageURL(url: nil)
    }

    private static func decodeToolCalls(
        _ val: UnsafeMutablePointer<yyjson_val>?
    ) throws(DecodeError) -> [OpenAIController.ChatToolCall]? {
        guard let val else { return nil }
        guard yyjson_is_arr(val) else {
            throw DecodeError(message: "expected array for 'tool_calls'")
        }
        var calls: [OpenAIController.ChatToolCall] = []
        calls.reserveCapacity(yyjson_arr_size(val))
        var iter = yyjson_arr_iter_with(val)
        while let elem = yyjson_arr_iter_next(&iter) {
            guard yyjson_is_obj(elem) else {
                throw DecodeError(message: "expected object in 'tool_calls'")
            }
            var cursor = ObjectCursor(elem)
            let id = try cursor.string("id")
            let type = try cursor.string("type")
            guard let fnVal = cursor.nonNull("function"), yyjson_is_obj(fnVal) else {
                throw DecodeError(message: "missing 'function' in 'tool_calls'")
            }
            var fnCursor = ObjectCursor(fnVal)
            guard let name = try fnCursor.string("name") else {
                throw DecodeError(message: "missing 'name' in tool call function")
            }
            let arguments = try fnCursor.string("arguments")
            calls.append(OpenAIController.ChatToolCall(
                id: id, type: type, function: .init(name: name, arguments: arguments)))
        }
        return calls
    }
}

extension ChatRequestFastReader {
    fileprivate static func requestedTools(
        in root: UnsafeMutablePointer<yyjson_val>
    ) throws(OpenAIController.UndeclaredForcedTool) -> OpenAIController.RequestedTools {
        let choice = objField(root, "tool_choice")
        let forced = (choice.flatMap { objField($0, "function") }.flatMap { strField($0, "name") })
            ?? choice.flatMap { strField($0, "name") }
        let parallel = boolField(root, "parallel_tool_calls") != false
            && choice.flatMap { boolField($0, "disable_parallel_tool_use") } != true
        let raw = objectArray(root, "tools") + objectArray(root, "additional_tools")
        let tools = raw.compactMap(decodeToolDefinition)
        let offered = tools.isEmpty ? nil : tools
        if (strField(root, "tool_choice") ?? choice.flatMap { strField($0, "type") }) == "none" {
            return OpenAIController.RequestedTools(tools: offered, choice: ToolChoice(callsDisabled: true))
        }
        if let forced, !tools.contains(where: { $0.name == forced }) {
            throw OpenAIController.UndeclaredForcedTool(name: forced)
        }
        return OpenAIController.RequestedTools(
            tools: offered, choice: forced == nil && parallel ? nil : ToolChoice(forcedTool: forced, parallel: parallel))
    }

    private static func decodeToolDefinition(_ entry: UnsafeMutablePointer<yyjson_val>) -> ToolDefinition? {
        guard strField(entry, "type") == "function" || rawField(entry, "input_schema") != nil else { return nil }
        let spec = objField(entry, "function") ?? entry
        guard let name = strField(spec, "name"), !name.isEmpty else { return nil }
        let rawParams = rawField(spec, "parameters") ?? rawField(spec, "input_schema")
        let parameters = rawParams.flatMap { yyjson_is_obj($0) || yyjson_is_arr($0) ? writeSortedJSON($0) : nil }
        return ToolDefinition(name: name, description: strField(spec, "description"), parametersJSON: parameters)
    }

    fileprivate static func responseFormat(
        in root: UnsafeMutablePointer<yyjson_val>
    ) throws(OpenAIController.InvalidResponseFormat) -> ResponseFormat? {
        if let guided = try guidedFormat(root) { return guided }
        guard let format = objField(root, "response_format")
                ?? objField(root, "text").flatMap({ objField($0, "format") }) else { return nil }
        switch strField(format, "type") {
        case "text":
            return nil
        case "json_object":
            return .jsonObject
        case "json_schema":
            return try jsonSchemaFormat(format)
        case "regex":
            return .regex(try grammarText(format, key: "regex"))
        case "grammar":
            return .grammar(try grammarText(format, key: "grammar"))
        case "structural_tag":
            guard let source = writeSortedJSON(format) else {
                throw OpenAIController.InvalidResponseFormat(message: "response_format structural_tag is not a JSON object")
            }
            return .structuralTag(source)
        case let type:
            throw OpenAIController.InvalidResponseFormat(message:
                "response_format type \(type.map { "'\($0)' is not supported" } ?? "is missing"); expected text, json_object, json_schema, regex, grammar or structural_tag")
        }
    }

    private static func guidedFormat(
        _ root: UnsafeMutablePointer<yyjson_val>
    ) throws(OpenAIController.InvalidResponseFormat) -> ResponseFormat? {
        let rawRegex = rawField(root, "guided_regex")
        let rawGrammar = rawField(root, "guided_grammar")
        if let rawRegex, !yyjson_is_null(rawRegex), !yyjson_is_str(rawRegex) {
            throw OpenAIController.InvalidResponseFormat(message: "guided_regex must be a string")
        }
        if let rawGrammar, !yyjson_is_null(rawGrammar), !yyjson_is_str(rawGrammar) {
            throw OpenAIController.InvalidResponseFormat(message: "guided_grammar must be a string")
        }
        let regex = strField(root, "guided_regex")
        let grammar = strField(root, "guided_grammar")
        let count = (regex != nil ? 1 : 0) + (grammar != nil ? 1 : 0)
        let hasResponseFormat = rawField(root, "response_format").map { !yyjson_is_null($0) } ?? false
        guard count <= 1, count == 0 || !hasResponseFormat else {
            throw OpenAIController.InvalidResponseFormat(message: "give one of response_format, guided_regex or guided_grammar")
        }
        return regex.map(ResponseFormat.regex) ?? grammar.map(ResponseFormat.grammar)
    }

    private static func jsonSchemaFormat(
        _ format: UnsafeMutablePointer<yyjson_val>
    ) throws(OpenAIController.InvalidResponseFormat) -> ResponseFormat {
        let declaration = objField(format, "json_schema") ?? format
        guard let schemaVal = rawField(declaration, "schema"), let text = writeJSON(schemaVal) else {
            throw OpenAIController.InvalidResponseFormat(message: "response_format json_schema needs a schema")
        }
        let requested = ResponseFormat.jsonSchema(text, strict: boolField(declaration, "strict") == true)
        let prepared: PreparedJSONSchema
        do {
            prepared = try requested.prepared()
        } catch {
            throw OpenAIController.InvalidResponseFormat(message: "response_format: \(error.message)")
        }
        if !prepared.relaxations.isEmpty {
            BasaltLogger(module: "OpenAIController").info(
                "response_format without strict: \(prepared.relaxations.joined(separator: "; "))")
        }
        return requested
    }

    private static func grammarText(
        _ format: UnsafeMutablePointer<yyjson_val>, key: StaticString
    ) throws(OpenAIController.InvalidResponseFormat) -> String {
        guard let text = strField(format, key), !text.isEmpty else {
            throw OpenAIController.InvalidResponseFormat(message: "response_format \(key) needs a non-empty \"\(key)\" string")
        }
        return text
    }
}

extension ChatRequestFastReader {
    fileprivate static func rawField(
        _ obj: UnsafeMutablePointer<yyjson_val>, _ key: StaticString
    ) -> UnsafeMutablePointer<yyjson_val>? {
        let ptr = UnsafeRawPointer(key.utf8Start).assumingMemoryBound(to: CChar.self)
        return yyjson_obj_getn(obj, ptr, key.utf8CodeUnitCount)
    }

    fileprivate static func objField(
        _ obj: UnsafeMutablePointer<yyjson_val>, _ key: StaticString
    ) -> UnsafeMutablePointer<yyjson_val>? {
        guard let val = rawField(obj, key), yyjson_is_obj(val) else { return nil }
        return val
    }

    fileprivate static func strField(_ obj: UnsafeMutablePointer<yyjson_val>, _ key: StaticString) -> String? {
        guard let val = rawField(obj, key), yyjson_is_str(val) else { return nil }
        return ObjectCursor.uncheckedString(val)
    }

    fileprivate static func boolField(_ obj: UnsafeMutablePointer<yyjson_val>, _ key: StaticString) -> Bool? {
        guard let val = rawField(obj, key), yyjson_is_bool(val) else { return nil }
        return yyjson_get_bool(val)
    }

    fileprivate static func objectArray(
        _ obj: UnsafeMutablePointer<yyjson_val>, _ key: StaticString
    ) -> [UnsafeMutablePointer<yyjson_val>] {
        guard let val = rawField(obj, key), yyjson_is_arr(val) else { return [] }
        var out: [UnsafeMutablePointer<yyjson_val>] = []
        out.reserveCapacity(yyjson_arr_size(val))
        var iter = yyjson_arr_iter_with(val)
        while let elem = yyjson_arr_iter_next(&iter) {
            guard yyjson_is_obj(elem) else { return [] }
            out.append(elem)
        }
        return out
    }

    fileprivate static func writeJSON(_ val: UnsafeMutablePointer<yyjson_val>) -> String? {
        var len = 0
        guard let ptr = yyjson_val_write(val, YYJSON_WRITE_NOFLAG, &len) else { return nil }
        defer { free(ptr) }
        return String(decoding: UnsafeRawBufferPointer(start: ptr, count: len), as: UTF8.self)
    }

    fileprivate static func writeSortedJSON(_ val: UnsafeMutablePointer<yyjson_val>) -> String? {
        var out: [UInt8] = []
        out.reserveCapacity(128)
        guard appendSortedJSON(val, depth: 0, into: &out) else { return nil }
        return String(decoding: out, as: UTF8.self)
    }

    private static func appendSortedJSON(
        _ val: UnsafeMutablePointer<yyjson_val>, depth: Int, into out: inout [UInt8]
    ) -> Bool {
        guard depth <= JSONNestingLimitMiddleware<BasicRequestContext>.depthLimit else { return false }
        if yyjson_is_obj(val) {
            return appendSortedObject(val, depth: depth + 1, into: &out)
        }
        if yyjson_is_arr(val) {
            out.append(UInt8(ascii: "["))
            var iter = yyjson_arr_iter_with(val)
            var first = true
            while let elem = yyjson_arr_iter_next(&iter) {
                if !first { out.append(UInt8(ascii: ",")) }
                first = false
                guard appendSortedJSON(elem, depth: depth + 1, into: &out) else { return false }
            }
            out.append(UInt8(ascii: "]"))
            return true
        }
        var len = 0
        guard let ptr = yyjson_val_write(val, YYJSON_WRITE_NOFLAG, &len) else { return false }
        defer { free(ptr) }
        out.append(contentsOf: UnsafeRawBufferPointer(start: ptr, count: len))
        return true
    }

    private static func appendSortedObject(
        _ obj: UnsafeMutablePointer<yyjson_val>, depth: Int, into out: inout [UInt8]
    ) -> Bool {
        var entries: [(key: String, keyVal: UnsafeMutablePointer<yyjson_val>, val: UnsafeMutablePointer<yyjson_val>)] = []
        entries.reserveCapacity(yyjson_obj_size(obj))
        var iter = yyjson_obj_iter_with(obj)
        while let keyVal = yyjson_obj_iter_next(&iter) {
            guard let itemVal = yyjson_obj_iter_get_val(keyVal) else { return false }
            entries.append((ObjectCursor.uncheckedString(keyVal), keyVal, itemVal))
        }
        entries.sort { $0.key < $1.key }
        out.append(UInt8(ascii: "{"))
        for (index, entry) in entries.enumerated() {
            if index > 0 { out.append(UInt8(ascii: ",")) }
            guard appendSortedJSON(entry.keyVal, depth: depth, into: &out) else { return false }
            out.append(UInt8(ascii: ":"))
            guard appendSortedJSON(entry.val, depth: depth, into: &out) else { return false }
        }
        out.append(UInt8(ascii: "}"))
        return true
    }
}
