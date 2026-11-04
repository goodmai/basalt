import Foundation

struct ThinkingSplitter: Sendable {
    enum Segment: Equatable, Sendable {
        case reasoning(String)
        case answer(String)
    }

    private(set) var isThinking: Bool
    private var trimsAnswerStart = false
    private var buffer = ""

    init(startsThinking: Bool) {
        isThinking = startsThinking
    }

    mutating func feed(_ text: String, final: Bool = false) -> [Segment] {
        buffer.append(text)
        var segments: [Segment] = []
        while !buffer.isEmpty {
            guard let boundary = nextMarker() else {
                let held = final ? 0 : MLXInferenceEngine.thinkingMarkerHoldback
                guard buffer.count > held else { break }
                let cut = buffer.index(buffer.endIndex, offsetBy: -held)
                append(String(buffer[..<cut]), to: &segments)
                buffer.removeSubrange(..<cut)
                break
            }
            let opensThinking = !isThinking && MLXInferenceEngine.thinkingOpeners.contains(String(buffer[boundary]))
            append(String(buffer[..<boundary.lowerBound]), to: &segments)
            buffer.removeSubrange(..<boundary.upperBound)
            if buffer.hasPrefix("\n") { buffer.removeFirst() }
            if isThinking, !opensThinking { trimsAnswerStart = true }
            isThinking = opensThinking
        }
        return segments
    }

    private func nextMarker() -> Range<String.Index>? {
        if isThinking { return MLXInferenceEngine.firstMarker(MLXInferenceEngine.thinkingClosers, in: buffer) }
        return [MLXInferenceEngine.firstMarker(MLXInferenceEngine.thinkingOpeners, in: buffer),
                MLXInferenceEngine.firstMarker(MLXInferenceEngine.strayThinkingClosers, in: buffer)]
            .compactMap { $0 }.min { $0.lowerBound < $1.lowerBound }
    }

    private mutating func append(_ text: String, to segments: inout [Segment]) {
        guard !text.isEmpty else { return }
        guard !isThinking else { return segments.append(.reasoning(text)) }
        guard trimsAnswerStart else { return segments.append(.answer(text)) }
        let visible = String(text.drop(while: \.isNewline))
        guard !visible.isEmpty else { return }
        trimsAnswerStart = false
        segments.append(.answer(visible))
    }
}
