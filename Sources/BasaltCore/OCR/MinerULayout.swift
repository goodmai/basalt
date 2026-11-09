import Foundation

public struct MinerULayout: Sendable, Equatable {
    public struct Block: Sendable, Equatable {
        public let type: String
        public let box: OCRBox
        public let angle: Int?
        public let continues: Bool
    }

    static let layoutPrompt = "\nLayout Detection:"
    static let textPrompt = "\nText Recognition:"
    static let tablePrompt = "\nTable Recognition:"
    static let formulaPrompt = "\nFormula Recognition:"
    static let imageAnalysisPrompt = "\nImage Analysis:"
    static let layoutSide = 1036
    static let minEdge = 28
    static let maxEdgeRatio = 50.0

    static let blockTypes: Set<String> = [
        "text", "title", "doc_title", "paragraph_title", "table", "equation", "formula_number", "code", "algorithm",
        "aside_text", "ref_text", "index", "phonetic", "list_item", "caption", "table_caption", "image_caption",
        "code_caption", "footnote", "table_footnote", "image_footnote", "header", "footer", "page_number",
        "page_footnote", "image", "chart", "list", "image_block", "equation_block", "unknown",
    ]
    static let skipped: Set<String> = ["list", "equation_block", "image_block", "image", "chart"]
    static let angles = ["<|rotate_up|>": 0, "<|rotate_right|>": 90, "<|rotate_down|>": 180, "<|rotate_left|>": 270]

    public let blocks: [Block]

    public init(parsing output: String) {
        let pattern = #/<\|box_start\|>([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(<\|rotate_(?:up|right|down|left)\|>)?(.*?)(?=<\|box_start\|>|$)/#
            .dotMatchesNewlines()
        var blocks: [Block] = []
        for match in output.matches(of: pattern) {
            let (_, x1, y1, x2, y2, kind, rotation, tail) = match.output
            guard let box = Self.box([x1, y1, x2, y2].compactMap { Int($0) }) else { continue }
            let type = kind.lowercased() == "unknown" ? "image" : kind.lowercased()
            guard Self.blockTypes.contains(type) else { continue }
            let angle = rotation.flatMap { Self.angles[String($0)] }
            blocks.append(Block(type: type, box: box, angle: angle, continues: type == "text" && tail.contains("txt_contd_tgt")))
        }
        self.blocks = Self.withoutCovered(Self.deduplicated(blocks))
    }

    static func box(_ coordinates: [Int]) -> OCRBox? {
        guard coordinates.count == 4, coordinates.allSatisfy({ (0 ... 1000).contains($0) }) else { return nil }
        let (x1, x2) = (min(coordinates[0], coordinates[2]), max(coordinates[0], coordinates[2]))
        let (y1, y2) = (min(coordinates[1], coordinates[3]), max(coordinates[1], coordinates[3]))
        guard x1 != x2, y1 != y2 else { return nil }
        return OCRBox(x1: Double(x1) / 1000, y1: Double(y1) / 1000, x2: Double(x2) / 1000, y2: Double(y2) / 1000)
    }

    static func deduplicated(_ blocks: [Block]) -> [Block] {
        var kept: [Block] = []
        for block in blocks {
            let duplicate = kept.contains { other in
                guard other.type == block.type, other.angle == block.angle else { return false }
                let intersection = block.box.intersection(other.box)
                let union = block.box.area + other.box.area - intersection
                return union > 0 && intersection / union > 0.9
            }
            if !duplicate { kept.append(block) }
        }
        return kept
    }

    static func withoutCovered(_ blocks: [Block]) -> [Block] {
        let rules: [(candidates: Set<String>, containers: Set<String>)] = [
            (["text", "equation", "equation_block"], ["table"]),
            (["image_caption"], ["image", "chart", "image_block"]),
        ]
        return rules.reduce(blocks) { blocks, rule in
            let containers = blocks.filter { rule.containers.contains($0.type) }
            return blocks.filter { block in
                !rule.candidates.contains(block.type) || !containers.contains { block.box.coverage(by: $0.box) >= 0.9 }
            }
        }
    }

    static func prompt(forBlock type: String) -> String? {
        switch type {
        case _ where skipped.contains(type): nil
        case "table": tablePrompt
        case "equation": formulaPrompt
        default: textPrompt
        }
    }

    static func bracketed(_ latex: String) -> String {
        guard !latex.isEmpty else { return latex }
        let opened = latex.hasPrefix("\\[") ? latex : "\\[\n" + latex
        return opened.hasSuffix("\\]") ? opened : opened + "\n\\]"
    }

    static func assemble(_ recognized: [(block: Block, text: String)]) -> (markdown: String, blocks: [OCRBlock]) {
        var paragraphs: [String] = []
        var lastText: Int?
        for (block, text) in recognized where !text.isEmpty {
            if block.continues, let index = lastText {
                paragraphs[index] += (text.unicodeScalars.contains(where: isCJK) ? "" : " ") + text
            } else {
                paragraphs.append(text)
                if block.type == "text" { lastText = paragraphs.count - 1 }
            }
        }
        let blocks = recognized.map { OCRBlock(type: $0.block.type, bbox: $0.block.box, text: $0.text) }
        return (paragraphs.joined(separator: "\n\n"), blocks)
    }

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        (0x4E00 ... 0x9FFF).contains(scalar.value) || (0x3400 ... 0x4DBF).contains(scalar.value)
    }
}
