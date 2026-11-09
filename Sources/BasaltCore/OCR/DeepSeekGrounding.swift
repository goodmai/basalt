import Foundation

enum DeepSeekGrounding {
    static let prefix = "<|grounding|>"
    static let scale = 999.0

    static func parse(_ output: String) -> (markdown: String, blocks: [OCRBlock]) {
        let tag = #/<\|ref\|>(?<type>.*?)<\|/ref\|><\|det\|>(?<boxes>.*?)<\|/det\|>/#
        let matches = output.matches(of: tag)
        guard !matches.isEmpty else {
            let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
            return (text, [OCRBlock(type: "text", bbox: .page, text: text)])
        }
        let blocks = matches.enumerated().map { index, match in
            let end = index + 1 < matches.count ? matches[index + 1].range.lowerBound : output.endIndex
            let text = String(output[match.range.upperBound ..< end]).trimmingCharacters(in: .whitespacesAndNewlines)
            return OCRBlock(type: String(match.output.type), bbox: box(String(match.output.boxes)), text: text)
        }
        return (blocks.map(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n"), blocks)
    }

    static func box(_ text: String) -> OCRBox {
        let numbers = text.matches(of: #/-?\d+(\.\d+)?/#).compactMap { Double($0.output.0) }.map { min(max($0 / scale, 0), 1) }
        guard numbers.count >= 4, numbers.count.isMultiple(of: 4) else { return .page }
        let boxes = stride(from: 0, to: numbers.count, by: 4).map { Array(numbers[$0 ..< $0 + 4]) }
        return OCRBox(
            x1: boxes.map { $0[0] }.min() ?? 0, y1: boxes.map { $0[1] }.min() ?? 0,
            x2: boxes.map { $0[2] }.max() ?? 1, y2: boxes.map { $0[3] }.max() ?? 1)
    }
}
