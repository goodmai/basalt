import BasaltCore
import Foundation

public enum IntentActions {
    public static func ask(_ question: String, model: String?, engine: EngineClient) async throws -> String {
        var conversation = ConversationSnapshot(id: ConversationID(), model: model)
        conversation.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.text(question)])]
        var answer = ""
        for try await delta in engine.chat(ChatRequest(conversation)) {
            if case .text(let piece) = delta { answer += piece }
        }
        return answer
    }

    public static func summarize(_ file: ChatAttachment, engine: EngineClient) async throws -> String {
        let response = try await engine.pipeline(PipelineCall(attachments: [file], task: "Summarize the attached file."))
        return PipelineResult(response, count: 1).answer
    }

    public static func transcribe(_ file: ChatAttachment, engine: EngineClient) async throws -> String {
        guard file.kind == "audio" else { throw EngineError.refused(status: 400, message: "\(file.name) is not an audio file") }
        let response = try await engine.pipeline(PipelineCall(attachments: [file], task: "Repeat the transcript."))
        return PipelineResult(response, count: 1).perceived.first ?? ""
    }

    public static func switchModel(_ id: String, library: LibraryClient) async throws -> String {
        guard LibraryReducer.isRepoID(id) else { throw EngineError.refused(status: 400, message: "\(id) is not a Hugging Face repo id") }
        let fit = try await library.fit(id)
        guard fit.fits else { throw EngineError.refused(status: 409, message: "\(id) does not fit in this Mac's memory" + (fit.warning.map { ": \($0)" } ?? "")) }
        try await library.load(id)
        return id
    }
}
