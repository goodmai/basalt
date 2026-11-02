import Foundation

public enum TokenEvent: Sendable {
    case partial(text: String, metrics: GenerationMetrics)
    case complete(fullText: String)
    case error(Error)
}

public struct GenerationMetrics: Sendable {
    public let tokensPerSecond: Double
    public let memoryUsageMB: Float
    
    public init(tokensPerSecond: Double, memoryUsageMB: Float) {
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsageMB = memoryUsageMB
    }
}

public struct RenderState: Sendable {
    public let content: String
    public let isGenerating: Bool
    public let fpsTarget: Int
    public let dirtyRects: [CGRect]
    
    public init(content: String, isGenerating: Bool, fpsTarget: Int = 30, dirtyRects: [CGRect] = []) {
        self.content = content
        self.isGenerating = isGenerating
        self.fpsTarget = fpsTarget
        self.dirtyRects = dirtyRects
    }
}

@MainActor
public protocol RenderPipeline: AnyObject, Sendable {
    func submit(state: RenderState)
}
