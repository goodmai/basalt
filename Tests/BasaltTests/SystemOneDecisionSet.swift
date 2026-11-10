import Foundation
@testable import BasaltCore

struct LabelledDecision {
    let state: String
    let name: String
    let question: SystemOneQuestion
    let label: String
}

enum SystemOneDecisionSet {
    static let departments = [
        "billing": ["I was charged twice for order #%d.", "My invoice for order #%d shows the wrong amount."],
        "shipping": ["Where is my parcel for order #%d? Tracking has not moved.", "The courier never delivered order #%d."],
        "technical": ["The app crashes when I open order #%d.", "My device will not pair after the update; order #%d."],
        "account": ["I cannot log in to the account linked to order #%d.", "Please change the e-mail on my profile; last order #%d."],
        "returns": ["I want to return the shoes from order #%d, wrong size.", "How do I exchange the jacket from order #%d?"],
        "sales": ["Can I get a bulk quote similar to order #%d?", "Is there a discount if I buy again what I got in order #%d?"],
    ]
    static let questions = PipelinePreset.triageQuestions

    static func generate(states count: Int, seed: UInt64) -> [LabelledDecision] {
        var generator = SplitMix64(seed: seed)
        let names = departments.keys.sorted()
        return (0 ..< count).flatMap { _ -> [LabelledDecision] in
            let department = names[Int.random(in: 0 ..< names.count, using: &generator)]
            let templates = departments[department]!
            let order = Int.random(in: 1000 ... 9999, using: &generator)
            let ticket = String(format: templates[Int.random(in: 0 ..< templates.count, using: &generator)], order)
            let promised = Int.random(in: 2 ... 7, using: &generator)
            let delivered = Int.random(in: 1 ... 10, using: &generator)
            let users = [1, 1, 40, 5000][Int.random(in: 0 ..< 4, using: &generator)]
            let outage = Int.random(in: 0 ..< 6, using: &generator) == 0
            let level = outage ? 3 : users >= 1000 ? 2 : users > 1 ? 2 : Int.random(in: 0 ... 1, using: &generator)
            let impact = outage ? "The service is completely down." : level == 0 ? "Only a label is misaligned." : "Users affected: \(users)."
            let state = "Ticket: \(ticket)\nOrder: {\"id\":\(order),\"promised_days\":\(promised),\"delivered_days\":\(delivered)}\nIncident: \(impact)"
            return [
                LabelledDecision(state: state, name: "department", question: questions["department"]!, label: department),
                LabelledDecision(state: state, name: "late", question: questions["late"]!, label: delivered > promised ? "true" : "false"),
                LabelledDecision(state: state, name: "severity", question: questions["severity"]!, label: String(level)),
            ]
        }
    }

    static func llmPrompt(_ item: LabelledDecision) -> String {
        let (keys, descriptions) = SystemOneCore.options(item.question)
        let options = zip(keys, descriptions).map { "- \($0): \($1)" }.joined(separator: "\n")
        return "State:\n\(item.state)\n\nQuestion: \(item.question.instructions)\nOptions:\n\(options)\n\nAnswer with exactly one option label and nothing else."
    }

    static func parse(_ reply: String, keys: [String]) -> String? {
        let words = reply.lowercased().split { !$0.isLetter && !$0.isNumber && $0 != "_" }.map(String.init)
        return words.first { keys.contains($0) }
    }

    static func predicted(_ answer: SystemOneAnswer) -> String? {
        answer.choice ?? answer.noul.map { $0 >= 0.5 ? "true" : "false" } ?? answer.probabilities?.max { $0.value < $1.value }?.key
    }
}
