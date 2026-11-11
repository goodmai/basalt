import BasaltCore
import Foundation
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app conversations in SwiftData behind a model actor: snapshots in and out, edits, search, recovery after a crash (#434 phase 1)")
struct ConversationRepositoryTests {
    static func repository() throws -> ConversationRepository {
        ConversationRepository(modelContainer: try ConversationRepository.container(at: nil))
    }

    static func conversation(_ title: String = "Basalt", texts: [String] = ["What is basalt?", "A volcanic rock."]) -> ConversationSnapshot {
        var conversation = ConversationSnapshot(id: ConversationID(), model: "org/model", systemPrompt: "Be brief.")
        conversation.title = title
        conversation.messages = texts.enumerated().map { index, text in
            var message = MessageSnapshot(id: MessageID(), role: index.isMultiple(of: 2) ? .user : .assistant, parts: [.text(text)])
            if !index.isMultiple(of: 2) {
                message.parts.insert(.reasoning("rocks"), at: 0)
                var receipt = Receipt(model: "org/model")
                (receipt.promptTokens, receipt.completionTokens, receipt.timeToFirstToken) = (12, 5, .milliseconds(250))
                message.receipt = receipt
            }
            return message
        }
        return conversation
    }

    @Test("A saved conversation loads back equal: parts, order, state and receipt")
    func roundTrip() async throws {
        let repository = try Self.repository()
        let conversation = Self.conversation()
        try await repository.save(conversation)
        #expect(try await repository.load(conversation.id) == conversation)
        #expect(try await repository.load(ConversationID()) == nil)
    }

    @Test("Saving again updates edited messages, drops removed ones, appends new ones, and the newest conversation lists first")
    func edits() async throws {
        let repository = try Self.repository()
        var first = Self.conversation("first")
        let second = Self.conversation("second")
        try await repository.save(first, at: Date(timeIntervalSince1970: 100))
        try await repository.save(second, at: Date(timeIntervalSince1970: 200))
        #expect(try await repository.summaries().map(\.title) == ["second", "first"])
        first.messages[1].parts = [.text("Changed.")]
        first.messages.removeFirst()
        first.messages.append(MessageSnapshot(id: MessageID(), role: .user, parts: [.text("More?")]))
        try await repository.save(first, at: Date(timeIntervalSince1970: 300))
        #expect(try await repository.load(first.id) == first)
        #expect(try await repository.summaries().map(\.title) == ["first", "second"])
        try await repository.delete(second.id)
        #expect(try await repository.summaries().map(\.id) == [first.id])
    }

    @Test("Search finds conversations by message text, each once, ignoring case")
    func search() async throws {
        let repository = try Self.repository()
        let rocks = Self.conversation("rocks", texts: ["Tell me about BASALT", "Basalt is dark."])
        let other = Self.conversation("other", texts: ["Granite?", "Light."])
        try await repository.save(rocks)
        try await repository.save(other)
        #expect(try await repository.search("basalt") == [rocks.id])
        #expect(try await repository.search("nothing").isEmpty)
    }

    @Test("At launch, answers left streaming by a crash become stopped, and the count is reported")
    func recovery() async throws {
        let repository = try Self.repository()
        var conversation = Self.conversation()
        conversation.messages[1].state = .streaming
        try await repository.save(conversation)
        #expect(try await repository.recoverInterrupted() == 1)
        #expect(try await repository.load(conversation.id)?.messages[1].state == .stopped)
        #expect(try await repository.recoverInterrupted() == 0)
    }

    @Test("A file-backed store keeps conversations across containers")
    func persists() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "conversations-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appending(path: "Conversations.store")
        let conversation = Self.conversation()
        try await ConversationRepository(modelContainer: try ConversationRepository.container(at: url)).save(conversation)
        #expect(try await ConversationRepository(modelContainer: try ConversationRepository.container(at: url)).load(conversation.id) == conversation)
    }
}
