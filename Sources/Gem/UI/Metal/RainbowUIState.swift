import Foundation

@MainActor
public class RainbowUIState: ObservableObject {
    public enum Mode: Int {
        case idle = 0
        case processing = 1
        case streaming = 2
        case error = 3
        case finished = 4
    }
    
    @Published public var currentMode: Mode = .idle
    @Published public var inputText: String = ""
    @Published public var placeholderHint: String = "Введите ваш запрос…"
    
    public init() {}
    
    public func setMode(_ mode: Mode) {
        self.currentMode = mode
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
}
