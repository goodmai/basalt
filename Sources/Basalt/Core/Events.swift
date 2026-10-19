import Foundation

/// Событие генерации токенов (MVI Intent/Event)
public enum TokenEvent: Sendable {
    case partial(text: String, metrics: GenerationMetrics)
    case complete(fullText: String)
    case error(Error)
}

/// Метрики производительности
public struct GenerationMetrics: Sendable {
    public let tokensPerSecond: Double
    public let memoryUsageMB: Float
    
    public init(tokensPerSecond: Double, memoryUsageMB: Float) {
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsageMB = memoryUsageMB
    }
}

/// Иммутабельное состояние для отрисовки Metal (View State)
public struct RenderState: Sendable {
    public let content: String
    public let isGenerating: Bool
    public let fpsTarget: Int
    public let dirtyRects: [CGRect] // Оптимизация: перерисовывать только изменившиеся области
    
    public init(content: String, isGenerating: Bool, fpsTarget: Int = 30, dirtyRects: [CGRect] = []) {
        self.content = content
        self.isGenerating = isGenerating
        self.fpsTarget = fpsTarget
        self.dirtyRects = dirtyRects
    }
}

/// Порт для рендеринга (может быть CLI или Metal)
@MainActor
public protocol RenderPipeline: AnyObject, Sendable {
    func submit(state: RenderState)
}
