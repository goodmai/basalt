import Foundation

/// A single message in the chat history
public struct ChatMessage: Identifiable, Sendable {
    public let id = UUID()
    public let role: Role
    public var text: String
    public let timestamp: Date
    
    public enum Role: Sendable {
        case user
        case assistant
        case system
    }
    
    public init(role: Role, text: String) {
        self.role = role
        self.text = text
        self.timestamp = Date()
    }
}

@MainActor
public class RainbowUIState: ObservableObject {
    public enum Mode: Int, Sendable {
        case idle = 0
        case processing = 1
        case streaming = 2
        case error = 3
        case finished = 4
    }
    
    @Published public var currentMode: Mode = .idle
    @Published public var inputText: String = ""
    @Published public var placeholderHint: String = "Введите ваш запрос…"
    @Published public var messages: [ChatMessage] = []
    @Published public var modelName: String = "No model loaded"
    @Published public var tokensPerSecond: Double = 0
    @Published public var scrollOffset: CGFloat = 0
    
    /// State transition log for agentic testing
    @Published public var stateLog: [(Date, Mode, Mode)] = []
    
    public init() {}
    
    public func setMode(_ mode: Mode) {
        let previous = currentMode
        self.currentMode = mode
        stateLog.append((Date(), previous, mode))
        
        switch mode {
        case .idle:
            placeholderHint = "Введите ваш запрос…"
        case .processing:
            placeholderHint = "Генерирую ответ…"
        case .streaming:
            placeholderHint = "Печатаю…"
        case .error:
            placeholderHint = "Ошибка соединения"
        case .finished:
            placeholderHint = "Готово. Спросите ещё"
        }
    }
    
    public func addUserMessage(_ text: String) {
        messages.append(ChatMessage(role: .user, text: text))
    }
    
    public func addAssistantMessage(_ text: String) {
        messages.append(ChatMessage(role: .assistant, text: text))
    }
    
    public func appendToLastAssistant(_ chunk: String) {
        guard let lastIndex = messages.indices.last,
              messages[lastIndex].role == .assistant else {
            messages.append(ChatMessage(role: .assistant, text: chunk))
            return
        }
        messages[lastIndex].text += chunk
    }
    
    /// Export state log for agentic testing verification
    public func exportStateLog() -> String {
        stateLog.map { date, from, to in
            let formatter = ISO8601DateFormatter()
            return "\(formatter.string(from: date)): \(from) -> \(to)"
        }.joined(separator: "\n")
    }
}
