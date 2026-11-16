import Foundation

public enum NemotronRowMapper {
    private static let foreignInlineTokens = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|eom_id|>",
        "<|im_start|>", "<|im_end|>", "<s>", "</s>",
    ]

    public static func sanitizeTemplateLeakage(_ raw: String) -> String {
        var cleaned = raw
        while let start = cleaned.range(of: "<|start_header_id|>"),
              let end = cleaned.range(of: "<|end_header_id|>", range: start.upperBound ..< cleaned.endIndex) {
            cleaned.removeSubrange(start.lowerBound ..< end.upperBound)
        }
        for token in foreignInlineTokens where cleaned.contains(token) {
            cleaned = cleaned.replacingOccurrences(of: token, with: "")
        }
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func splitThinking(_ raw: String) -> (reasoning: String?, answer: String) {
        let cleaned = sanitizeTemplateLeakage(raw)
        guard let close = cleaned.range(of: "</think>") else {
            return (nil, cleaned)
        }
        let after = String(cleaned[close.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        var before = Substring(cleaned[..<close.lowerBound])
        if let open = before.range(of: "<think>") {
            before = before[open.upperBound...]
        }
        let thought = String(before).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !after.isEmpty else {
            return (nil, thought)
        }
        return (thought.isEmpty ? nil : thought, after)
    }

    public static func formatTarget(
        reasoning: String?,
        answer: String,
        includeReasoning: Bool,
        markers: ThinkingMarkers = .none
    ) -> String {
        let cleanAnswer = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard includeReasoning,
              let cleanReasoning = reasoning?.trimmingCharacters(in: .whitespacesAndNewlines),
              !cleanReasoning.isEmpty else {
            return cleanAnswer
        }
        let open = markers.open.first ?? "<think>"
        let close = markers.close.first ?? "</think>"
        return "\(open)\n\(cleanReasoning)\n\(close)\n\n\(cleanAnswer)"
    }

    public static func parseJSONLines(_ jsonl: String) throws(BasaltError) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        let lines = jsonl.split(whereSeparator: \.isNewline)
        objects.reserveCapacity(lines.count)
        for (index, slice) in lines.enumerated() {
            let trimmed = slice.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            guard let parsed = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any] else {
                throw .invalidRequestStructure(details: "train: line \(index + 1) of corpus is not a JSON object")
            }
            if let nested = parsed["row"] as? [String: Any] {
                objects.append(nested)
            } else {
                objects.append(parsed)
            }
        }
        return objects
    }

    public static func normalizeRow(_ object: [String: Any]) -> CorpusExample? {
        let passRate = numericField(in: object, keys: ["pass_rate", "passRate", "score", "reward"])
        if let multiTurn = extractMessagesExample(from: object, passRate: passRate) {
            return multiTurn
        }
        return extractPairExample(from: object, passRate: passRate)
    }

    public static func prepare(
        rows: [[String: Any]],
        spec: CorpusSourceSpec,
        heldOutFraction: Float
    ) throws(BasaltError) -> PreparedCorpusSplit {
        var rejected: [String: Int] = [:]
        var seen = Set<String>()
        var valid: [CorpusExample] = []
        for raw in rows {
            guard let example = normalizeRow(raw) else {
                rejected["unmapped", default: 0] += 1
                continue
            }
            if let minRate = spec.minPassRate {
                guard let actual = example.passRate, actual >= minRate else {
                    rejected["low_pass_rate", default: 0] += 1
                    continue
                }
            }
            let key = signature(of: example)
            guard seen.insert(key).inserted else {
                rejected["duplicate", default: 0] += 1
                continue
            }
            valid.append(example)
        }
        let sampled = deterministicSample(valid, maxRows: spec.maxRows, seed: spec.seed)
        guard sampled.count >= 2 else {
            throw .invalidRequestStructure(details: "train: corpus produced \(sampled.count) usable examples; at least 2 are required (train + held-out)")
        }
        let fraction = min(max(heldOutFraction, 0.05), 0.5)
        let heldOutCount = min(sampled.count - 1, max(1, Int((Float(sampled.count) * fraction).rounded(.up))))
        let splitIndex = sampled.count - heldOutCount
        return PreparedCorpusSplit(
            train: Array(sampled[..<splitIndex]),
            heldOut: Array(sampled[splitIndex...]),
            rejected: rejected
        )
    }

    private static func deterministicSample(_ items: [CorpusExample], maxRows: Int, seed: UInt64) -> [CorpusExample] {
        guard items.count > 1 else { return items }
        var copy = items
        var rng = SplitMix64(seed: seed)
        for index in stride(from: copy.count - 1, through: 1, by: -1) {
            let swapIndex = Int(rng.next() % UInt64(index + 1))
            if swapIndex != index { copy.swapAt(index, swapIndex) }
        }
        return copy.count > maxRows ? Array(copy.prefix(maxRows)) : copy
    }

    private static func signature(of example: CorpusExample) -> String {
        let promptPart = example.contextMessages.map { "\($0["role"] ?? ""):\($0["content"] ?? "")" }.joined(separator: "\u{1F}")
        return "\(promptPart)\u{0}\(example.answer)"
    }
}

extension NemotronRowMapper {
    static func extractMessagesExample(from object: [String: Any], passRate: Double?) -> CorpusExample? {
        let rawList = (object["messages"] as? [[String: Any]]) ?? (object["conversations"] as? [[String: Any]])
        guard let rawList, rawList.count >= 2 else { return nil }
        var normalized: [(role: String, content: String, reasoning: String?)] = []
        for entry in rawList {
            let rawRole = (entry["role"] as? String) ?? (entry["from"] as? String) ?? ""
            let role = canonicalRole(rawRole)
            var text = (entry["content"] as? String) ?? (entry["value"] as? String) ?? ""
            if let toolCalls = entry["tool_calls"],
               let toolData = try? JSONSerialization.data(withJSONObject: toolCalls, options: [.sortedKeys]) {
                let toolText = String(decoding: toolData, as: UTF8.self)
                text = text.isEmpty ? "<tool_call>\(toolText)</tool_call>" : "\(text)\n<tool_call>\(toolText)</tool_call>"
            }
            let explicitReasoning = stringField(in: entry, keys: ["reasoning", "reasoning_content", "thought"])
            normalized.append((role: role, content: text, reasoning: explicitReasoning))
        }
        guard let lastAssistantIndex = normalized.lastIndex(where: { $0.role == "assistant" }),
              lastAssistantIndex > 0 else { return nil }
        var context: [[String: String]] = []
        for turn in normalized[..<lastAssistantIndex] {
            let clean = turn.role == "assistant" ? splitThinking(turn.content).answer : sanitizeTemplateLeakage(turn.content)
            if !clean.isEmpty { context.append(["role": turn.role, "content": clean]) }
        }
        guard context.contains(where: { $0["role"] == "user" }) else { return nil }
        let targetTurn = normalized[lastAssistantIndex]
        let split = splitThinking(targetTurn.content)
        let explicit = targetTurn.reasoning.map(sanitizeTemplateLeakage).flatMap { $0.isEmpty ? nil : $0 }
        let topLevel = stringField(in: object, keys: ["reasoning", "reasoning_content", "thought"]).map(sanitizeTemplateLeakage)
        let finalAnswer = split.answer
        guard !finalAnswer.isEmpty else { return nil }
        return CorpusExample(
            contextMessages: context,
            reasoning: explicit ?? split.reasoning ?? topLevel,
            answer: finalAnswer,
            passRate: passRate
        )
    }

    static func extractPairExample(from object: [String: Any], passRate: Double?) -> CorpusExample? {
        let promptRaw = stringField(in: object, keys: ["input", "prompt", "problem", "question", "instruction"])
        let personaRaw = stringField(in: object, keys: ["persona", "persona_description", "system", "system_prompt"])
        let answerRaw = stringField(in: object, keys: ["output", "response", "generated_solution", "solution", "answer", "completion"])
        if let promptRaw, let answerRaw {
            let userText = sanitizeTemplateLeakage(promptRaw)
            let split = splitThinking(answerRaw)
            guard !userText.isEmpty, !split.answer.isEmpty else { return nil }
            var messages: [[String: String]] = []
            if let sys = personaRaw.map(sanitizeTemplateLeakage), !sys.isEmpty {
                messages.append(["role": "system", "content": sys])
            }
            messages.append(["role": "user", "content": userText])
            let explicit = stringField(in: object, keys: ["reasoning", "reasoning_content", "thought", "cot"]).map(sanitizeTemplateLeakage)
            return CorpusExample(
                contextMessages: messages,
                reasoning: explicit.flatMap({ $0.isEmpty ? nil : $0 }) ?? split.reasoning,
                answer: split.answer,
                passRate: passRate
            )
        }
        if let persona = personaRaw.map(sanitizeTemplateLeakage), !persona.isEmpty,
           let sample = stringField(in: object, keys: ["professional_persona", "sports_persona", "arts_persona", "culinary_persona", "summary", "text"]).map(sanitizeTemplateLeakage),
           !sample.isEmpty {
            let messages = [
                ["role": "system", "content": "Respond concisely in a style consistent with the given persona profile."],
                ["role": "user", "content": persona],
            ]
            return CorpusExample(contextMessages: messages, reasoning: nil, answer: sample, passRate: passRate)
        }
        return nil
    }

    private static func canonicalRole(_ raw: String) -> String {
        switch raw.lowercased() {
        case "system", "developer": "system"
        case "assistant", "gpt", "model", "bot": "assistant"
        case "tool", "observation", "function": "user"
        default: "user"
        }
    }

    private static func stringField(in dict: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let str = dict[key] as? String, !str.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return str
            }
        }
        if let inputList = dict["input"] as? [[String: Any]], !inputList.isEmpty {
            let joined = inputList.compactMap { ($0["content"] as? String) ?? ($0["value"] as? String) }.joined(separator: "\n")
            if !joined.isEmpty { return joined }
        }
        return nil
    }

    private static func numericField(in dict: [String: Any], keys: [String]) -> Double? {
        for key in keys {
            if let num = dict[key] as? Double { return num }
            if let num = dict[key] as? Int { return Double(num) }
            if let str = dict[key] as? String, let parsed = Double(str) { return parsed }
        }
        return nil
    }
}
