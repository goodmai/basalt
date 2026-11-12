import AppIntents
import BasaltApp
import BasaltAppCore
import Foundation

@MainActor
enum AppServices {
    static var model: AppModel?

    static func engine() async throws -> (model: AppModel, endpoint: EngineEndpoint) {
        guard let model else { throw EngineError.notRunning }
        guard let endpoint = await model.readyEndpoint() else { throw EngineError.starting }
        return (model, endpoint)
    }

    static func attachment(_ file: IntentFile) throws -> ChatAttachment {
        guard let attachment = ChatAttachment(name: file.filename, data: file.data) else { throw EngineError.refused(status: 415, message: "\(file.filename) is not a PDF, image or audio file of at most 32 MB") }
        return attachment
    }
}

struct AskBasaltIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Basalt"
    static let description = IntentDescription("Ask the local model a question and get its answer as text.")

    @Parameter(title: "Question") var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let (model, endpoint) = try await AppServices.engine()
        return .result(value: try await IntentActions.ask(question, model: model.model, engine: .http(endpoint)))
    }
}

struct SummarizeFileIntent: AppIntent {
    static let title: LocalizedStringResource = "Summarize File"
    static let description = IntentDescription("Read a PDF, image or audio file on this Mac and summarize it with the local model.")

    @Parameter(title: "File") var file: IntentFile

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let (_, endpoint) = try await AppServices.engine()
        return .result(value: try await IntentActions.summarize(try AppServices.attachment(file), engine: .http(endpoint)))
    }
}

struct TranscribeAudioIntent: AppIntent {
    static let title: LocalizedStringResource = "Transcribe Audio"
    static let description = IntentDescription("Transcribe an audio file on this Mac.")

    @Parameter(title: "Audio") var file: IntentFile

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let (_, endpoint) = try await AppServices.engine()
        return .result(value: try await IntentActions.transcribe(try AppServices.attachment(file), engine: .http(endpoint)))
    }
}

struct SwitchModelIntent: AppIntent {
    static let title: LocalizedStringResource = "Switch Model"
    static let description = IntentDescription("Load another downloaded model, if it fits in this Mac's memory.")

    @Parameter(title: "Model", description: "A Hugging Face repo id, such as mlx-community/gemma-4-12B-it-qat-4bit") var model: String

    @MainActor
    func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let (_, endpoint) = try await AppServices.engine()
        let loaded = try await IntentActions.switchModel(model, library: .live(endpoint))
        UserDefaults.standard.set(loaded, forKey: "model")
        return .result(value: loaded)
    }
}

struct BasaltShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: AskBasaltIntent(), phrases: ["Ask \(.applicationName)"], shortTitle: "Ask", systemImageName: "bubble.left")
        AppShortcut(intent: SummarizeFileIntent(), phrases: ["Summarize a file with \(.applicationName)"], shortTitle: "Summarize File", systemImageName: "doc.text")
    }
}
