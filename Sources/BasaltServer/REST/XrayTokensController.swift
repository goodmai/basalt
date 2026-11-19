import BasaltCore
import Foundation
import HTTPTypes
import Hummingbird

struct XrayTokensController: Sendable {
    let orchestrator: any ModelOrchestrating

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/xray/tokens", use: attributeTokens)
    }

    @Sendable
    func attributeTokens(request: Request, context: BasicRequestContext) async throws -> Response {
        let buffer = try await request.body.collect(upTo: ChatPipeline.textBodyLimit)
        let data = Data(buffer.readableBytesView)
        let dto: AttributedTokenTraceRequest
        do {
            dto = try JSONDecoder().decode(AttributedTokenTraceRequest.self, from: data)
        } catch {
            return ProviderErrorResponse.response(
                status: .badRequest,
                message: "Invalid /v1/xray/tokens body: \(error.localizedDescription)",
                error: error,
                shape: .openAI
            )
        }
        do {
            let report = try await orchestrator.research { model throws(BasaltError) in
                try await ResearchCall.lab(model).attributeTokens(dto)
            }
            let encoded = try JSONEncoder().encode(report)
            return Self.jsonDataResponse(encoded)
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/xray/tokens")
        }
    }

    static func chatXrayResponse(
        chat: [ChatMessageInput],
        maxTokens: Int?,
        temperature: Double?,
        topLogprobs: Int,
        model: String,
        orchestrator: any ModelOrchestrating
    ) async throws -> Response {
        let clampedMax = min(AttributedTokenTraceRequest.maxGeneratedLimit, max(1, maxTokens ?? 64))
        let temp = Float(temperature ?? 0.0)
        let req = AttributedTokenTraceRequest(
            prompt: ChatMessageInput.plainText(chat),
            messages: chat,
            maxTokens: clampedMax,
            temperature: temp,
            topLogprobs: max(0, min(20, topLogprobs)),
            topHeads: 4
        )
        do {
            let report = try await orchestrator.research { loaded throws(BasaltError) in
                try await ResearchCall.lab(loaded).attributeTokens(req)
            }
            let message = OpenAIController.ChatMessage(
                role: "assistant",
                content: .text(report.generatedText),
                toolCalls: nil,
                reasoning: nil
            )
            let finish = report.completionTokens >= clampedMax ? "length" : "stop"
            let choice = OpenAIController.ChatCompletionResponse.Choice(
                index: 0,
                message: message,
                finishReason: finish,
                logprobs: .init(content: report.tokens, summary: report.summary)
            )
            let usage = OpenAIController.ChatCompletionResponse.Usage(
                promptTokens: report.promptTokens,
                promptTokensDetails: OpenAIController.CachedTokens(cachedTokens: 0),
                completionTokens: report.completionTokens,
                totalTokens: report.promptTokens + report.completionTokens,
                completionTokensDetails: OpenAIController.ReasoningTokens(reasoningTokens: 0)
            )
            let payload = OpenAIController.ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: model,
                choices: [choice],
                usage: usage
            )
            return jsonDataResponse(try JSONEncoder().encode(payload))
        } catch {
            return ProviderErrorResponse.response(for: error, path: "/v1/chat/completions")
        }
    }

    static func streamingLogprobsFinishFrame(
        id: String,
        created: Int,
        model: String,
        finishReason: String,
        telemetry: [OutputTokenTelemetry]
    ) -> ByteBuffer {
        struct LogprobsFinishChunk: Encodable {
            struct Choice: Encodable {
                let index: Int
                let delta: [String: String]
                let logprobs: OpenAIController.ChatCompletionResponse.ChoiceLogprobs
                let finishReason: String
                enum CodingKeys: String, CodingKey {
                    case index, delta, logprobs
                    case finishReason = "finish_reason"
                }
            }
            let id: String
            let object: String
            let created: Int
            let model: String
            let choices: [Choice]
        }
        let chunk = LogprobsFinishChunk(
            id: id,
            object: "chat.completion.chunk",
            created: created,
            model: model,
            choices: [.init(index: 0, delta: [:], logprobs: .init(content: telemetry), finishReason: finishReason)]
        )
        var buffer = ByteBuffer()
        buffer.writeStaticString("data: ")
        if let json = try? JSONEncoder().encode(chunk) {
            buffer.writeBytes(json)
        } else {
            buffer.writeStaticString("{}")
        }
        buffer.writeStaticString("\n\n")
        return buffer
    }

    private static func jsonDataResponse(_ data: Data) -> Response {
        var buf = ByteBuffer()
        buf.writeBytes(data)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: buf)
        )
    }
}
