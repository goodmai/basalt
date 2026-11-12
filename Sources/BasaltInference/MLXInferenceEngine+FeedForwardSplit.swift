import BasaltCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SplitFeedForwardBox {
    let split: SplitFeedForward
    let lengths: [Int]

    init(_ split: SplitFeedForward, lengths: [Int]) {
        (self.split, self.lengths) = (split, lengths.sorted())
    }

    func chunk(from position: Int, to end: Int) -> Int {
        lengths.last { $0 <= end - position } ?? end - position
    }
}

extension MLXInferenceEngine {
    func loadFeedForwardSplit(_ loaded: ModelContainer, epoch: ModelEpoch) async {
        guard let path = configuration.feedForwardSplitBundle else { return }
        do {
            let share = try await CoreAIFeedForwardShare.load(URL(fileURLWithPath: path))
            await attachFeedForwardSplit(share, lengths: share.lengths, activation: share.activation, to: loaded, epoch: epoch, source: path)
        } catch {
            logger.warn("--experimental-ane-ffn-split: \(error.localizedDescription); prefill stays on the GPU")
        }
    }

    func attachFeedForwardSplit(
        _ share: any FeedForwardShare, lengths: [Int], activation: FeedForwardActivation, to loaded: ModelContainer, epoch: ModelEpoch, source: String
    ) async {
        let attached = await loaded.performSending(nonSendable: share) { context, share -> Result<Int, BasaltError> in
            guard let model = context.model as? any SplitFeedForwardPrefill & Module else {
                return .failure(.modelConfigurationInvalid(path: source, reason: "the loaded model has no split feed-forward prefill (Gemma 4 dense only)"))
            }
            do throws(BasaltError) {
                let split = try SplitFeedForward(model: model, share: share, activation: activation)
                return .success(split.rest.count)
            } catch {
                return .failure(error)
            }
        }
        switch attached {
        case .success(let layers) where layers > 0 && !lengths.isEmpty:
            await loaded.update { context in
                guard let model = context.model as? any SplitFeedForwardPrefill & Module,
                      let split = try? SplitFeedForward(model: model, share: share, activation: activation) else { return }
                context.feedForwardSplit = SplitFeedForwardBox(split, lengths: lengths)
            }
            record(.learned(epoch) { $0.feedForwardSplitLengths = lengths.sorted() })
            logger.info("Prefill splits the feed-forward of \(layers) layers with the Neural Engine (\(share.columns) columns, chunks of \(lengths.sorted()) tokens) from \(source)")
        case .success:
            logger.warn("--experimental-ane-ffn-split: \(source) splits no dense layer of the loaded model; prefill stays on the GPU")
        case .failure(let error):
            logger.warn("--experimental-ane-ffn-split: \(error.localizedDescription); prefill stays on the GPU")
        }
    }

    func splitPrefill(_ input: LMInput, _ prompt: CachedPrompt, options: LaunchOptions, container: ModelContainer) async -> CachedPrompt {
        let total = input.text.tokens.dim(-1)
        guard let minimum = modelFacts.feedForwardSplitLengths?.first, !options.mtp, options.params.kvBits == nil,
              !Self.carriesMoreThanTokens(input), total - 1 - (prompt.reuse?.reusedTokens ?? 0) >= minimum
        else { return prompt }
        let params = options.params
        do {
            return try await container.performSending(nonSendable: (input, prompt)) { context, values in
                try await Self.splitPrefilled(values.0, values.1, context: context, params: params)
            }
        } catch {
            logger.warn("Split prefill failed (\(error.localizedDescription)); prefilling on the GPU")
            var fallback = prompt
            fallback.reuse = nil
            return fallback
        }
    }

    static func splitPrefilled(_ input: LMInput, _ prompt: CachedPrompt, context: ModelContext, params: GenerateParameters) async throws -> CachedPrompt {
        guard let box = context.feedForwardSplit as? SplitFeedForwardBox, let model = context.model as? any SplitFeedForwardPrefill else { return prompt }
        let tokens = input.text.tokens.reshaped(-1)
        let cache = try prompt.reuse?.cache ?? PromptCache(context.model.newCache(parameters: params))
        var position = prompt.reuse?.reusedTokens ?? 0
        let end = tokens.dim(0) - 1
        while position < end {
            let length = box.chunk(from: position, to: end)
            _ = try await model.prefill(text: tokens[position ..< position + length], cache: cache.layers) { layer, x in try await box.split(layer, x) }
            eval(cache.layers)
            position += length
        }
        var filled = prompt
        filled.reuse = (cache, end)
        return filled
    }
}
