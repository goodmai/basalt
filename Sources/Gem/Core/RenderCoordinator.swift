import Foundation
import CoreGraphics

/// Координатор событий (Event Bus / Throttle).
/// Обеспечивает MVI-пайплайн и дросселирование до заданного FPS.
public actor RenderCoordinator {
    private weak var renderer: (any RenderPipeline)?
    private var pendingState: RenderState?
    private var lastRenderTime: TimeInterval = 0
    private var renderTask: Task<Void, Never>?
    
    private let logger = GemLogger(module: "RenderCoordinator")
    
    public init() {}
    
    public func setRenderer(_ pipeline: any RenderPipeline) {
        self.renderer = pipeline
        self.logger.debug("Render pipeline attached.")
    }
    
    /// Паттерн Throttling (Coalescing events).
    /// Принимает новое состояние и отправляет в пайплайн только если прошло достаточно времени (33.3мс для 30FPS).
    public func submit(state: RenderState) {
        pendingState = state
        
        let targetInterval = 1.0 / Double(state.fpsTarget)
        let now = Date().timeIntervalSince1970
        
        if now - lastRenderTime >= targetInterval {
            // Можем рендерить сразу
            dispatchRender()
        } else if renderTask == nil {
            // Ждем до следующего тика
            let waitTime = targetInterval - (now - lastRenderTime)
            renderTask = Task {
                try? await Task.sleep(nanoseconds: UInt64(waitTime * 1_000_000_000))
                self.dispatchRender()
            }
        }
    }
    
    private func dispatchRender() {
        renderTask?.cancel()
        renderTask = nil
        
        guard let stateToRender = pendingState, let renderer = renderer else { return }
        
        lastRenderTime = Date().timeIntervalSince1970
        // Очищаем pendingState, так как мы его сейчас отрендерим
        pendingState = nil
        
        Task { @MainActor in
            renderer.submit(state: stateToRender)
        }
    }
}
