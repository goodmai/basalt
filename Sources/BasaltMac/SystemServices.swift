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

    @discardableResult
    func applyLaunch(_ urls: [URL], defaults: UserDefaults = .standard) -> [String]? {
        let fallback = model.model
        guard let launch = urls.lazy.compactMap({ EngineChoice.launchURL($0, fallbackChatModel: fallback) }).first else { return nil }
        defaults.set(launch.model, forKey: EngineChoice.modelKey)
        let compose = launch.arguments.count >= 3 && launch.arguments[1] == "--compose" ? launch.arguments[2] : ""
        defaults.set(compose, forKey: EngineChoice.composeKey)
        show()
        return launch.arguments
    }

    func open(_ urls: [URL]) {
        if let arguments = applyLaunch(urls) {
            Task { await model.switchEngine(arguments) }
        }
        let fileURLs = urls.filter { $0.isFileURL && ChatAttachment.accepts($0.lastPathComponent) }
        guard !fileURLs.isEmpty else { return }
        Task {
            var attachments: [ChatAttachment] = []
            for url in fileURLs {
                if let attachment = await ChatAttachment.loadAsync(from: url) {
                    attachments.append(attachment)
                }
            }
            guard !attachments.isEmpty else { return }
            model.startConversation(text: "", attachments: attachments)
            show()
        }
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
