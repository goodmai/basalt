import BasaltAppCore
import Foundation
import SwiftData

@ModelActor
actor ConversationRepository {
    static let schema = Schema([ConversationRecord.self, MessageRecord.self])

    static func container(at url: URL?) throws -> ModelContainer {
        let configuration = url.map { ModelConfiguration(schema: schema, url: $0) } ?? ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
        return try ModelContainer(for: schema, configurations: [configuration])
    }

    func save(_ conversation: ConversationSnapshot, at now: Date = Date()) throws {
        let record = try fetch(conversation.id) ?? insert(conversation.id, now)
        (record.title, record.model, record.systemPrompt, record.updated) = (conversation.title, conversation.model, conversation.systemPrompt, now)
        let existing = Dictionary(record.messages.map { ($0.id, $0) }) { first, _ in first }
        let kept = Set(conversation.messages.map(\.id.raw))
        for stale in record.messages where !kept.contains(stale.id) { modelContext.delete(stale) }
        for (order, message) in conversation.messages.enumerated() {
            let target = existing[message.id.raw] ?? {
                let created = MessageRecord(id: message.id.raw, order: order)
                created.conversation = record
                modelContext.insert(created)
                return created
            }()
            RecordMapping.fill(target, from: message, order: order)
        }
        try modelContext.save()
    }

    func load(_ id: ConversationID) throws -> ConversationSnapshot? {
        try fetch(id).map(RecordMapping.snapshot)
    }

    func summaries() throws -> [ConversationSummary] {
        let descriptor = FetchDescriptor<ConversationRecord>(sortBy: [SortDescriptor(\.updated, order: .reverse)])
        return try modelContext.fetch(descriptor).map { ConversationSummary(id: ConversationID($0.id), title: $0.title, updated: $0.updated) }
    }

    func delete(_ id: ConversationID) throws {
        guard let record = try fetch(id) else { return }
        modelContext.delete(record)
        try modelContext.save()
    }

    func search(_ text: String) throws -> [ConversationID] {
        let descriptor = FetchDescriptor<MessageRecord>(predicate: #Predicate { $0.text.localizedStandardContains(text) })
        var seen = Set<UUID>()
        return try modelContext.fetch(descriptor).compactMap { $0.conversation?.id }.filter { seen.insert($0).inserted }.map { ConversationID($0) }
    }

    func recoverInterrupted() throws -> Int {
        let streaming = MessageSnapshot.State.streaming.rawValue
        let records = try modelContext.fetch(FetchDescriptor<MessageRecord>(predicate: #Predicate { $0.state == streaming }))
        for record in records { record.state = MessageSnapshot.State.stopped.rawValue }
        try modelContext.save()
        return records.count
    }

    private func fetch(_ id: ConversationID) throws -> ConversationRecord? {
        let raw = id.raw
        return try modelContext.fetch(FetchDescriptor<ConversationRecord>(predicate: #Predicate { $0.id == raw })).first
    }

    private func insert(_ id: ConversationID, _ now: Date) -> ConversationRecord {
        let record = ConversationRecord(id: id.raw, created: now)
        modelContext.insert(record)
        return record
    }
}
