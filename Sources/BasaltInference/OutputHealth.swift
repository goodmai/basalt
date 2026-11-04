import BasaltCore
import MLX
import MLXLMCommon
import Synchronization

final class NonFiniteLogitAlarm: Sendable {
    private let tripped = Atomic<Bool>(false)

    var isTripped: Bool { tripped.load(ordering: .relaxed) }

    func trip() {
        guard !tripped.exchange(true, ordering: .relaxed) else { return }
        BasaltLogger(module: "NonFiniteLogitGuard").error(BasaltError.nonFiniteLogits.localizedDescription)
    }
}

struct NonFiniteLogitGuard: LogitProcessor {
    static let interval = 32

    let alarm: NonFiniteLogitAlarm
    var sampled = 0

    static func guarding(
        _ components: GenerationComponents, temperature: Float
    ) -> (components: GenerationComponents, alarm: NonFiniteLogitAlarm) {
        let alarm = NonFiniteLogitAlarm()
        guard temperature > 0 || components.logitProcessorFactory != nil else { return (components, alarm) }
        return (components.appendingBasaltProcessor { NonFiniteLogitGuard(alarm: alarm) }, alarm)
    }

    static func isUsable(_ logits: MLXArray) -> Bool {
        (logicalNot(isNaN(logits).any()) .&& isFinite(logits.max(axis: -1)).all()).item(Bool.self)
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        if sampled % Self.interval == 0, !alarm.isTripped, !Self.isUsable(logits) { alarm.trip() }
        return logits
    }

    mutating func didSample(token: MLXArray) {
        sampled += 1
    }
}

struct DegenerateOutputDetector: Sendable {
    static let repeatLimit = 64
    static let answerSampleLimit = 4096

    let controlTokens: [String]
    private var last = ""
    private var run = 0
    private var repeated = false
    private var thought = false
    private var answered = false
    private var answer = ""

    init(controlTokens: [String]) {
        self.controlTokens = controlTokens
    }

    mutating func feed(_ segment: ThinkingSplitter.Segment) {
        switch segment {
        case .reasoning(let text):
            count(text)
            thought = thought || !text.isEmpty
        case .answer(let text):
            count(text)
            answered = answered || !text.isEmpty
            if answer.count < Self.answerSampleLimit { answer += text.filter { !$0.isWhitespace } }
        }
    }

    private mutating func count(_ text: String) {
        guard !text.isEmpty else { return }
        run = text == last ? run + 1 : 1
        last = text
        repeated = repeated || run >= Self.repeatLimit
    }

    func verdict(calledTools: Bool) -> DegenerateOutput? {
        if repeated { return .repeatedToken }
        guard !calledTools else { return nil }
        if answered {
            return controlTokens.reduce(answer) { $0.replacingOccurrences(of: $1, with: "") }.isEmpty ? .blankAnswer : nil
        }
        return thought ? .emptyAfterThinking : nil
    }
}

extension GenerationPlan {
    var draftSource: DraftSource? {
        drafts ? .mtp : speculativeLookup ? .lookup : nil
    }
}
