import AppKit
import BasaltApp
import BasaltAppCore
import UserNotifications

@MainActor
final class ServiceProvider: NSObject {
    let model: AppModel
    let show: @MainActor () -> Void

    init(model: AppModel, show: @escaping @MainActor () -> Void) { (self.model, self.show) = (model, show) }

    @objc func askBasalt(_ pasteboard: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString>) {
        guard let text = pasteboard.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return }
        model.startConversation(text: text)
        show()
    }

    func open(_ urls: [URL]) {
        let attachments = urls.compactMap { url in (try? Data(contentsOf: url)).flatMap { ChatAttachment(name: url.lastPathComponent, data: $0) } }
        guard !attachments.isEmpty else { return }
        model.startConversation(text: "", attachments: attachments)
        show()
    }
}

@MainActor
enum AnswerNotifications {
    static var available: Bool { Bundle.main.bundleIdentifier != nil }

    static func request() {
        guard available else { return }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    static func post(_ conversation: ConversationSnapshot) {
        guard available, !NSApplication.shared.isActive, let answer = conversation.messages.last?.text, !answer.isEmpty else { return }
        let content = UNMutableNotificationContent()
        (content.title, content.body) = (conversation.title.isEmpty ? "Basalt answered" : conversation.title, String(answer.prefix(240)))
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: conversation.id.raw.uuidString, content: content, trigger: nil))
    }
}
