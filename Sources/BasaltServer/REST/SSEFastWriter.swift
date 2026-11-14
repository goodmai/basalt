import Foundation
import Hummingbird
import BasaltCore

extension OpenAIController {
    struct ChatCompletionChunk: Codable, Equatable, Sendable {
        struct Choice: Codable, Equatable, Sendable {
            var index: Int
            var delta: Delta
            var finishReason: String?

            enum CodingKeys: String, CodingKey {
                case index, delta
                case finishReason = "finish_reason"
            }
        }

        struct Delta: Codable, Equatable, Sendable {
            var role: String?
            var content: String?
            var reasoning: String?
            var reasoningContent: String?
            var toolCalls: [ToolCall]?

            init(
                role: String? = nil, content: String? = nil, reasoning: String? = nil,
                reasoningContent: String? = nil, toolCalls: [ToolCall]? = nil
            ) {
                self.role = role
                self.content = content
                self.reasoning = reasoning
                self.reasoningContent = reasoningContent
                self.toolCalls = toolCalls
            }

            enum CodingKeys: String, CodingKey {
                case role, content, reasoning
                case reasoningContent = "reasoning_content"
                case toolCalls = "tool_calls"
            }
        }

        struct ToolCall: Codable, Equatable, Sendable {
            struct Function: Codable, Equatable, Sendable {
                var name: String?
                var arguments: String?
            }

            var index: Int
            var id: String?
            var type: String?
            var function: Function?
        }

        struct Usage: Codable, Equatable, Sendable {
            struct PromptTokensDetails: Codable, Equatable, Sendable {
                var cachedTokens: Int

                enum CodingKeys: String, CodingKey {
                    case cachedTokens = "cached_tokens"
                }
            }

            struct CompletionTokensDetails: Codable, Equatable, Sendable {
                var reasoningTokens: Int

                enum CodingKeys: String, CodingKey {
                    case reasoningTokens = "reasoning_tokens"
                }
            }

            var promptTokens: Int
            var completionTokens: Int
            var totalTokens: Int
            var promptTokensDetails: PromptTokensDetails?
            var completionTokensDetails: CompletionTokensDetails?

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
                case promptTokensDetails = "prompt_tokens_details"
                case completionTokensDetails = "completion_tokens_details"
            }
        }

        var id: String
        var object: String
        var created: Int
        var model: String
        var choices: [Choice]
        var usage: Usage?

        init(
            id: String, object: String = "chat.completion.chunk", created: Int, model: String,
            choices: [Choice], usage: Usage? = nil
        ) {
            self.id = id
            self.object = object
            self.created = created
            self.model = model
            self.choices = choices
            self.usage = usage
        }
    }
}

typealias ChatCompletionChunk = OpenAIController.ChatCompletionChunk

extension ChatCompletionChunk.Usage {
    init(
        promptTokens: Int, completionTokens: Int, totalTokens: Int,
        cachedTokens: Int = 0, reasoningTokens: Int = 0
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
        self.completionTokensDetails = CompletionTokensDetails(reasoningTokens: reasoningTokens)
    }

    init(_ result: GenerationResponse) {
        self.promptTokens = result.promptTokens
        self.completionTokens = result.completionTokens
        self.totalTokens = result.promptTokens + result.completionTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: result.cachedPromptTokens)
        self.completionTokensDetails = CompletionTokensDetails(reasoningTokens: result.reasoningTokens ?? 0)
    }
}

struct SSEFastWriter: Sendable {
    static let allocator = ByteBufferAllocator()

    let id: String
    let created: Int
    let model: String
    private let prefix: [UInt8]

    init(id: String, created: Int, model: String) {
        self.id = id
        self.created = created
        self.model = model
        var buffer = Self.allocator.buffer(capacity: 96 + id.utf8.count + model.utf8.count)
        buffer.writeStaticString("data: {\"id\":\"")
        Self.writeEscaped(id, into: &buffer)
        buffer.writeStaticString("\",\"object\":\"chat.completion.chunk\",\"created\":")
        Self.writeDecimal(created, into: &buffer)
        buffer.writeStaticString(",\"model\":\"")
        Self.writeEscaped(model, into: &buffer)
        buffer.writeStaticString("\",")
        self.prefix = Array(buffer.readableBytesView)
    }

    func textDelta(_ text: String) -> ByteBuffer {
        let utf8 = text.utf8Span
        let span = utf8.span
        var buffer = Self.allocator.buffer(capacity: prefix.count + span.count + 112)
        writeTextDelta(span, into: &buffer)
        return buffer
    }

    func writeTextDelta(_ text: Span<UInt8>, into buffer: inout ByteBuffer) {
        writePrefix(into: &buffer)
        buffer.writeStaticString("\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"")
        Self.writeEscaped(text, into: &buffer)
        buffer.writeStaticString("\"},\"finish_reason\":null}]}\n\n")
    }

    func reasoningDelta(_ reasoning: String) -> ByteBuffer {
        let utf8 = reasoning.utf8Span
        let span = utf8.span
        var buffer = Self.allocator.buffer(capacity: prefix.count + span.count * 2 + 144)
        writeReasoningDelta(span, into: &buffer)
        return buffer
    }

    func writeReasoningDelta(_ reasoning: Span<UInt8>, into buffer: inout ByteBuffer) {
        writePrefix(into: &buffer)
        buffer.writeStaticString("\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning\":\"")
        Self.writeEscaped(reasoning, into: &buffer)
        buffer.writeStaticString("\",\"reasoning_content\":\"")
        Self.writeEscaped(reasoning, into: &buffer)
        buffer.writeStaticString("\"},\"finish_reason\":null}]}\n\n")
    }

    func toolCallDelta(index: Int, call: ToolCallOutput) -> ByteBuffer {
        let extra = call.id.utf8.count + call.name.utf8.count + call.arguments.utf8.count
        var buffer = Self.allocator.buffer(capacity: prefix.count + extra + 176)
        writeToolCallDelta(index: index, call: call, into: &buffer)
        return buffer
    }

    func writeToolCallDelta(index: Int, call: ToolCallOutput, into buffer: inout ByteBuffer) {
        writePrefix(into: &buffer)
        buffer.writeStaticString("\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":")
        Self.writeDecimal(index, into: &buffer)
        buffer.writeStaticString(",\"id\":\"")
        Self.writeEscaped(call.id, into: &buffer)
        buffer.writeStaticString("\",\"type\":\"function\",\"function\":{\"name\":\"")
        Self.writeEscaped(call.name, into: &buffer)
        buffer.writeStaticString("\",\"arguments\":\"")
        Self.writeEscaped(call.arguments, into: &buffer)
        buffer.writeStaticString("\"}}]},\"finish_reason\":null}]}\n\n")
    }

    func finish(_ finishReason: String) -> ByteBuffer {
        var buffer = Self.allocator.buffer(capacity: prefix.count + finishReason.utf8.count + 64)
        writeFinish(finishReason, into: &buffer)
        return buffer
    }

    func writeFinish(_ finishReason: String, into buffer: inout ByteBuffer) {
        writePrefix(into: &buffer)
        buffer.writeStaticString("\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"")
        Self.writeEscaped(finishReason, into: &buffer)
        buffer.writeStaticString("\"}]}\n\n")
    }

    func usage(_ result: GenerationResponse) -> ByteBuffer {
        usage(ChatCompletionChunk.Usage(result))
    }

    func usage(_ usage: ChatCompletionChunk.Usage) -> ByteBuffer {
        var buffer = Self.allocator.buffer(capacity: prefix.count + 224)
        writeUsage(usage, into: &buffer)
        return buffer
    }

    func writeUsage(_ usage: ChatCompletionChunk.Usage, into buffer: inout ByteBuffer) {
        writePrefix(into: &buffer)
        buffer.writeStaticString("\"choices\":[],\"usage\":")
        Self.writeUsageObject(usage, into: &buffer)
        buffer.writeStaticString("}\n\n")
    }

    private func writePrefix(into buffer: inout ByteBuffer) {
        let header = prefix.span
        buffer.writeBytes(header.bytes)
    }
}

extension SSEFastWriter {
    static func doneFrame() -> ByteBuffer {
        var buffer = allocator.buffer(capacity: 16)
        buffer.writeStaticString("data: [DONE]\n\n")
        return buffer
    }

    static func errorFrame(message: String) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: message.utf8.count + 64)
        buffer.writeStaticString("event: error\ndata: {\"error\":{\"message\":\"")
        writeEscaped(message, into: &buffer)
        buffer.writeStaticString("\",\"type\":\"server_error\"}}\n\n")
        return buffer
    }

    static func writeUsageObject(_ usage: ChatCompletionChunk.Usage, into buffer: inout ByteBuffer) {
        buffer.writeStaticString("{\"prompt_tokens\":")
        writeDecimal(usage.promptTokens, into: &buffer)
        buffer.writeStaticString(",\"completion_tokens\":")
        writeDecimal(usage.completionTokens, into: &buffer)
        buffer.writeStaticString(",\"total_tokens\":")
        writeDecimal(usage.totalTokens, into: &buffer)
        if let promptDetails = usage.promptTokensDetails {
            buffer.writeStaticString(",\"prompt_tokens_details\":{\"cached_tokens\":")
            writeDecimal(promptDetails.cachedTokens, into: &buffer)
            buffer.writeInteger(UInt8(ascii: "}"))
        }
        if let completionDetails = usage.completionTokensDetails {
            buffer.writeStaticString(",\"completion_tokens_details\":{\"reasoning_tokens\":")
            writeDecimal(completionDetails.reasoningTokens, into: &buffer)
            buffer.writeInteger(UInt8(ascii: "}"))
        }
        buffer.writeInteger(UInt8(ascii: "}"))
    }

    static func writeEscaped(_ string: String, into buffer: inout ByteBuffer) {
        let utf8 = string.utf8Span
        writeEscaped(utf8.span, into: &buffer)
    }

    static func writeEscaped(_ bytes: Span<UInt8>, into buffer: inout ByteBuffer) {
        var runStart = 0
        var index = 0
        while index < bytes.count {
            let byte = bytes[index]
            if byte >= 0x20, byte != UInt8(ascii: "\""), byte != UInt8(ascii: "\\") {
                index += 1
                continue
            }
            if index > runStart {
                let slice = bytes.extracting(runStart..<index)
                buffer.writeBytes(slice.bytes)
            }
            writeEscapeSequence(for: byte, into: &buffer)
            index += 1
            runStart = index
        }
        if runStart < bytes.count {
            let tail = bytes.extracting(runStart..<bytes.count)
            buffer.writeBytes(tail.bytes)
        }
    }

    private static func writeEscapeSequence(for byte: UInt8, into buffer: inout ByteBuffer) {
        switch byte {
        case UInt8(ascii: "\""): buffer.writeStaticString("\\\"")
        case UInt8(ascii: "\\"): buffer.writeStaticString("\\\\")
        case 0x08: buffer.writeStaticString("\\b")
        case 0x09: buffer.writeStaticString("\\t")
        case 0x0A: buffer.writeStaticString("\\n")
        case 0x0C: buffer.writeStaticString("\\f")
        case 0x0D: buffer.writeStaticString("\\r")
        default:
            let escaped: InlineArray<6, UInt8> = [
                UInt8(ascii: "\\"), UInt8(ascii: "u"), UInt8(ascii: "0"), UInt8(ascii: "0"),
                hexDigit(byte >> 4), hexDigit(byte & 0x0F),
            ]
            let span = escaped.span
            buffer.writeBytes(span.bytes)
        }
    }

    static func writeDecimal(_ value: Int, into buffer: inout ByteBuffer) {
        if value == 0 {
            buffer.writeInteger(UInt8(ascii: "0"))
            return
        }
        var scratch = InlineArray<24, UInt8>(repeating: 0)
        var magnitude: UInt = value < 0 ? 0 &- UInt(bitPattern: value) : UInt(bitPattern: value)
        var cursor = 24
        while magnitude > 0 {
            cursor -= 1
            let (quotient, remainder) = magnitude.quotientAndRemainder(dividingBy: 10)
            scratch[cursor] = UInt8(ascii: "0") &+ UInt8(truncatingIfNeeded: remainder)
            magnitude = quotient
        }
        if value < 0 {
            cursor -= 1
            scratch[cursor] = UInt8(ascii: "-")
        }
        let span = scratch.span
        let digits = span.extracting(cursor..<24)
        buffer.writeBytes(digits.bytes)
    }

    private static func hexDigit(_ nibble: UInt8) -> UInt8 {
        nibble < 10 ? (UInt8(ascii: "0") &+ nibble) : (UInt8(ascii: "a") &+ nibble &- 10)
    }
}
