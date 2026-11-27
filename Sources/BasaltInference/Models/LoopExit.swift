import Foundation
import MLX
import MLXLMCommon

enum LoopExit {
    static let environmentKey = "BASALT_LOOP_EXIT"

    static var environmentThreshold: Float? {
        ProcessInfo.processInfo.environment[environmentKey].flatMap(Float.init).flatMap { $0 > 0 && $0 < 1 ? $0 : nil }
    }

    static func copyLastPosition(from source: KVCache, to target: KVCache) -> [MLXArray] {
        let state = source.state
        guard state.count == 2 else { return [] }
        let (keys, values) = target.update(keys: state[0][0..., 0..., (-1)..., 0...], values: state[1][0..., 0..., (-1)..., 0...])
        return [keys, values]
    }
}

protocol LoopExitCounting: AnyObject {
    var loopExitThreshold: Float? { get set }
    var exitSteps: Int { get }
    var exitPasses: Int { get }
}

extension OuroInner: LoopExitCounting {}
extension Inkling2TextModel: LoopExitCounting {}
extension HuginnTransformer: LoopExitCounting {}

extension LoopExit {
    static func counter(of model: any LanguageModel) -> (any LoopExitCounting)? {
        if let ouro = model as? OuroModel { return ouro.model }
        if let inkling = model as? Inkling2Model { return inkling.model.languageModel }
        if let huginn = model as? HuginnModel { return huginn.transformer }
        return nil
    }

    static func passesPerToken(_ ids: [Int], model: any LanguageModel) throws -> [Int]? {
        guard let counter = counter(of: model), counter.loopExitThreshold != nil else { return nil }
        let cache = try model.newCache(parameters: nil)
        return ids.map { id in
            let before = counter.exitPasses
            eval(model(MLXArray([Int32(id)])[.newAxis], cache: cache))
            return counter.exitPasses - before
        }
    }
}
