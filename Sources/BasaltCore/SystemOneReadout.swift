import Foundation

public struct SystemOneReadout: Sendable, Equatable {
    public static let letters = (65 ... 90).map { String(UnicodeScalar($0)!) } + (97 ... 122).map { String(UnicodeScalar($0)!) }
    public static let scoreSuffix = " Rate along the ordered levels below (lowest first)."

    public var temperature: Double
    public var noulTemperature: Double
    public var noulBias: Double

    public init(temperature: Double = 1, noulTemperature: Double = 1, noulBias: Double = 0) {
        (self.temperature, self.noulTemperature, self.noulBias) = (temperature, noulTemperature, noulBias)
    }

    public static let uncalibrated = SystemOneReadout()

    public static func known(_ model: String) -> SystemOneReadout? {
        let name = model.lowercased()
        if name.contains("openjev-flash-9b") { return SystemOneReadout(temperature: 1.07, noulTemperature: 1.074766) }
        if name.contains("openjev/openjev") || name.hasSuffix("/openjev-mlx") || name.hasSuffix("/openjev-mlx-4bit") {
            return SystemOneReadout(temperature: 0.85, noulTemperature: 1.829074)
        }
        return nil
    }

    public static func options(_ question: SystemOneQuestion) -> [(key: String, description: String)] {
        switch (question.type, question.criteria) {
        case (.noul, let criteria):
            let named: [String: String] = if case .named(let map)? = criteria { map } else { [:] }
            return [("yes", named["true"] ?? "The statement is true."), ("no", named["false"] ?? "The statement is false.")]
        default:
            let (keys, descriptions) = SystemOneCore.options(question)
            return Array(zip(keys, descriptions)).map { (key: $0.0, description: $0.1) }
        }
    }

    public static func prefix(state: String) -> String { "State:\n\(state)" }

    public static func tail(_ question: SystemOneQuestion) -> String {
        let instructions = question.type == .score ? question.instructions + scoreSuffix : question.instructions
        let lines = options(question).enumerated().map { "[\(letters[$0.offset])] \($0.element.key): \($0.element.description)" }.joined(separator: "\n")
        return "\n\nQuestion: \(instructions)\nOptions:\n\(lines)\n\nAnswer with the letter of the best option only."
    }

    public func answer(_ question: SystemOneQuestion, letterLogits: [Double]) -> SystemOneAnswer {
        let z = letterLogits.map { $0 / temperature }
        let top = z.max() ?? 0
        let weights = z.map { exp($0 - top) }
        let total = weights.reduce(0, +)
        let p = weights.map { $0 / total }
        guard question.type == .noul else {
            let keys = SystemOneReadout.options(question).map(\.key)
            let scores = zip(keys, p).map { _, value in log(max(value, 1e-300)) }
            return SystemOneCore.answer(question, scores: scores, temperature: 1)
        }
        let yes = min(max(p.first ?? 0.5, 1e-4), 1 - 1e-4)
        var answer = SystemOneAnswer()
        answer.noul = 1 / (1 + exp(-(log(yes / (1 - yes)) / noulTemperature + noulBias)))
        return answer
    }
}
