import BasaltCore
import Foundation

public struct TranslationProbeItem: Sendable, Codable, Equatable {
    public var id: String
    public var source: String
    public var reference: String
    public var targetLang: String
    public var targetName: String
    public var targetNative: String

    enum CodingKeys: String, CodingKey {
        case id
        case source
        case reference
        case targetLang = "target_lang"
        case targetName = "target_name"
        case targetNative = "target_native"
    }

    public init(
        id: String,
        source: String,
        reference: String,
        targetLang: String,
        targetName: String,
        targetNative: String
    ) {
        self.id = id
        self.source = source
        self.reference = reference
        self.targetLang = targetLang
        self.targetName = targetName
        self.targetNative = targetNative
    }
}

public struct TranslationScore: Sendable, Equatable {
    public var chrfPP: Double
    public var comet: Double
    public var metricX24: Double
    public var techFidelity: Double

    public init(chrfPP: Double, comet: Double, metricX24: Double, techFidelity: Double) {
        self.chrfPP = chrfPP
        self.comet = comet
        self.metricX24 = metricX24
        self.techFidelity = techFidelity
    }
}

public enum TranslationMetrics {
    public static func chrfPP(
        hypothesis: String,
        reference: String,
        charOrder: Int = 6,
        wordOrder: Int = 2,
        beta: Double = 2.0
    ) -> Double {
        let hypNorm = collapseWhitespace(hypothesis)
        let refNorm = collapseWhitespace(reference)
        guard !hypNorm.isEmpty, !refNorm.isEmpty else { return 0.0 }
        let hypChars = Array(hypNorm)
        let refChars = Array(refNorm)
        let hypWords = tokenizeWords(hypNorm)
        let refWords = tokenizeWords(refNorm)

        var precisionSum = 0.0
        var recallSum = 0.0
        var orders = 0

        for n in 1 ... max(1, charOrder) {
            let (p, r) = ngramPR(hypChars, refChars, n: n)
            precisionSum += p
            recallSum += r
            orders += 1
        }
        if wordOrder > 0 {
            for n in 1 ... wordOrder {
                let (p, r) = ngramPR(hypWords, refWords, n: n)
                precisionSum += p
                recallSum += r
                orders += 1
            }
        }
        let avgP = precisionSum / Double(max(1, orders))
        let avgR = recallSum / Double(max(1, orders))
        guard avgP + avgR > 0 else { return 0.0 }
        let b2 = beta * beta
        return 100.0 * (1.0 + b2) * avgP * avgR / (b2 * avgP + avgR)
    }

    public static func techFidelity(source: String, hypothesis: String) -> Double {
        let tokens = protectedTokens(in: source)
        guard !tokens.isEmpty else { return 100.0 }
        let kept = tokens.filter { hypothesis.contains($0) }.count
        return 100.0 * Double(kept) / Double(tokens.count)
    }

    public static func scoreProxy(source: String, hypothesis: String, reference: String) -> TranslationScore {
        let chrf = chrfPP(hypothesis: hypothesis, reference: reference)
        let fidelity = techFidelity(source: source, hypothesis: hypothesis)
        let refLen = max(1, reference.count)
        let ratio = Double(hypothesis.count) / Double(refLen)
        let lenPenalty = ratio < 0.5 || ratio > 1.8 ? 0.85 : 1.0
        let cometRaw = (0.75 * (chrf / 100.0) + 0.25 * (fidelity / 100.0)) * lenPenalty
        let comet = min(1.0, max(0.0, cometRaw))
        let metricX24 = min(25.0, max(0.0, (1.0 - comet) * 25.0))
        return TranslationScore(chrfPP: chrf, comet: comet, metricX24: metricX24, techFidelity: fidelity)
    }

    public static func parseJudgeJSON(_ text: String) -> (comet: Double, metricX24: Double)? {
        guard let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}"), start < end else { return nil }
        let slice = String(text[start ... end])
        guard let data = slice.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let cometVal = (obj["comet"] as? Double) ?? (obj["comet"] as? Int).map(Double.init)
        let mxVal = (obj["metricx24"] as? Double) ?? (obj["metricx24"] as? Int).map(Double.init)
        guard let comet = cometVal, let mx = mxVal, comet.isFinite, mx.isFinite else { return nil }
        return (min(1.0, max(0.0, comet)), min(25.0, max(0.0, mx)))
    }

    private static func ngramPR<T: Hashable>(_ hyp: [T], _ ref: [T], n: Int) -> (Double, Double) {
        guard hyp.count >= n, ref.count >= n else { return (0.0, 0.0) }
        var hypCounts: [[T]: Int] = [:]
        var refCounts: [[T]: Int] = [:]
        for i in 0 ... (hyp.count - n) { hypCounts[Array(hyp[i ..< (i + n)]), default: 0] += 1 }
        for i in 0 ... (ref.count - n) { refCounts[Array(ref[i ..< (i + n)]), default: 0] += 1 }
        var overlap = 0
        for (gram, count) in hypCounts {
            if let refCount = refCounts[gram] { overlap += min(count, refCount) }
        }
        let p = Double(overlap) / Double(max(1, hyp.count - n + 1))
        let r = Double(overlap) / Double(max(1, ref.count - n + 1))
        return (p, r)
    }

    private static func collapseWhitespace(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    private static func tokenizeWords(_ text: String) -> [String] {
        text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
    }

    private static func protectedTokens(in source: String) -> Set<String> {
        var tokens: Set<String> = []
        for pattern in ["`[^`\\n]+`", "https?://[^\\s)\\]>]+"] {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let ns = source as NSString
            for match in regex.matches(in: source, range: NSRange(location: 0, length: ns.length)) {
                tokens.insert(ns.substring(with: match.range))
            }
        }
        return tokens
    }
}

public struct TranslationBenchmarkAdapter: BenchmarkAdapter {
    public let id = "basalt:translation"
    public let tier = BenchmarkAdapterTier.pureSwift
    public let summary = "Pure-Swift translation quality benchmark (chrF++, COMET, MetricX-24, technical Markdown fidelity) with lite and full modes"
    public var defaultSubmissionTarget: SubmissionTarget? { .mtLeaderboard }

    public init() {}

    public static let builtinProbes: [TranslationProbeItem] = [
        TranslationProbeItem(
            id: "tr-ja-1",
            source: "Run `basalt compose up -f docs/examples/compose/translate.yaml` for local translation with zero cloud calls.",
            reference: "`basalt compose up -f docs/examples/compose/translate.yaml` を実行すると、クラウド通信なしでローカル翻訳を利用できます。",
            targetLang: "ja", targetName: "Japanese", targetNative: "日本語"
        ),
        TranslationProbeItem(
            id: "tr-ko-1",
            source: "Run `basalt compose up -f docs/examples/compose/translate.yaml` for local translation with zero cloud calls.",
            reference: "`basalt compose up -f docs/examples/compose/translate.yaml`을 실행하면 클라우드 호출 없이 로컬 번역을 사용할 수 있습니다.",
            targetLang: "ko", targetName: "Korean", targetNative: "한국어"
        ),
        TranslationProbeItem(
            id: "tr-id-1",
            source: "Run `basalt compose up -f docs/examples/compose/translate.yaml` for local translation with zero cloud calls.",
            reference: "Jalankan `basalt compose up -f docs/examples/compose/translate.yaml` untuk terjemahan lokal tanpa panggilan cloud.",
            targetLang: "id", targetName: "Indonesian", targetNative: "Bahasa Indonesia"
        ),
        TranslationProbeItem(
            id: "tr-ru-1",
            source: "Run `basalt compose up -f docs/examples/compose/translate.yaml` for local translation with zero cloud calls.",
            reference: "Запустите `basalt compose up -f docs/examples/compose/translate.yaml` для локального перевода без обращений к облаку.",
            targetLang: "ru", targetName: "Russian", targetNative: "Русский"
        ),
        TranslationProbeItem(
            id: "tr-zh-1",
            source: "Use `basalt fit` to check which MLX models fit your Mac unified memory.",
            reference: "使用 `basalt fit` 检查哪些 MLX 模型适合您的 Mac 统一内存。",
            targetLang: "zh", targetName: "Simplified Chinese", targetNative: "中文"
        ),
        TranslationProbeItem(
            id: "tr-de-1",
            source: "Use `basalt fit` to check which MLX models fit your Mac unified memory.",
            reference: "Verwenden Sie `basalt fit`, um zu prüfen, welche MLX-Modelle in den Unified Memory Ihres Mac passen.",
            targetLang: "de", targetName: "German", targetNative: "Deutsch"
        ),
        TranslationProbeItem(
            id: "tr-es-1",
            source: "Use `basalt fit` to check which MLX models fit your Mac unified memory.",
            reference: "Usa `basalt fit` para comprobar qué modelos MLX caben en la memoria unificada de tu Mac.",
            targetLang: "es", targetName: "Spanish", targetNative: "Español"
        ),
        TranslationProbeItem(
            id: "tr-fr-1",
            source: "Use `basalt fit` to check which MLX models fit your Mac unified memory.",
            reference: "Utilisez `basalt fit` pour vérifier quels modèles MLX tiennent dans la mémoire unifiée de votre Mac.",
            targetLang: "fr", targetName: "French", targetNative: "Français"
        ),
    ]

    public func run(context: BenchmarkTaskContext) async throws -> BenchmarkAdapterOutput {
        let mode = context.task.params["mode"]?.stringValue ?? "lite"
        let isFull = mode == "full"
        let qeMode = context.task.params["qe_mode"]?.stringValue ?? (isFull ? "llm_judge" : "proxy")
        let judgeModel = context.task.params["judge_model"]?.stringValue ?? context.resolvedModelName
        let items = try loadItems(context: context, isFull: isFull)

        var samples: [BenchmarkSampleRecord] = []
        var chrfValues: [Double] = []
        var cometValues: [Double] = []
        var metricXValues: [Double] = []
        var fidelityValues: [Double] = []

        for item in items {
            let (hyp, reply, usage) = try await translate(item: item, context: context)
            var score = TranslationMetrics.scoreProxy(source: item.source, hypothesis: hyp, reference: item.reference)
            if qeMode == "llm_judge", let judged = try? await judgeQE(item: item, hypothesis: hyp, judgeModel: judgeModel, context: context) {
                score.comet = judged.comet
                score.metricX24 = judged.metricX24
            }
            chrfValues.append(score.chrfPP)
            cometValues.append(score.comet)
            metricXValues.append(score.metricX24)
            fidelityValues.append(score.techFidelity)

            let passed = score.chrfPP >= 35.0 && score.techFidelity >= 90.0
            var record = BenchmarkSampleRecord(
                id: item.id, adapter: id, model: context.resolvedModelName,
                prompt: item.source, answer: hyp, score: score.chrfPP / 100.0
            )
            record.dataset = context.task.dataset ?? "builtin-\(mode)"
            record.latencyMs = reply.totalMs
            record.ttftMs = reply.ttftMs
            record.promptTokens = usage.prompt
            record.completionTokens = usage.completion
            record.passed = passed
            record.metadata["target_lang"] = item.targetLang
            record.metadata["lang_pair"] = "en-\(item.targetLang)"
            record.metadata["reference"] = item.reference
            record.metadata["chrf_pp"] = String(format: "%.2f", score.chrfPP)
            record.metadata["comet"] = String(format: "%.4f", score.comet)
            record.metadata["metricx24"] = String(format: "%.2f", score.metricX24)
            record.metadata["tech_fidelity"] = String(format: "%.1f", score.techFidelity)
            samples.append(record)
        }

        let count = Double(max(1, samples.count))
        let metrics: [String: Double] = [
            "chrf_pp": chrfValues.reduce(0, +) / count,
            "comet": cometValues.reduce(0, +) / count,
            "metricx24": metricXValues.reduce(0, +) / count,
            "tech_fidelity": fidelityValues.reduce(0, +) / count,
            "accuracy": Double(samples.filter(\.passed).count) / count,
            "p50_ms": BenchmarkStats.percentile(samples.map(\.latencyMs), 0.5),
        ]
        return BenchmarkAdapterOutput(samples: samples, metrics: metrics)
    }

    private func loadItems(context: BenchmarkTaskContext, isFull: Bool) throws -> [TranslationProbeItem] {
        var items: [TranslationProbeItem] = []
        if let path = context.task.dataset, !path.isEmpty {
            let raw: String
            do {
                raw = try String(contentsOfFile: path, encoding: .utf8)
            } catch {
                throw BenchmarkError.invalidResponse("failed to read translation dataset '\(path)': \(error.localizedDescription)")
            }
            let decoder = JSONDecoder()
            for line in raw.split(separator: "\n") {
                if let data = String(line).data(using: .utf8),
                   let decoded = try? decoder.decode(TranslationProbeItem.self, from: data) {
                    items.append(decoded)
                }
            }
            guard !items.isEmpty else {
                throw BenchmarkError.invalidResponse("translation dataset '\(path)' contained no valid JSONL entries")
            }
        } else {
            items = Self.builtinProbes
        }
        if let filterRaw = context.task.params["locales"]?.stringValue {
            let allowed = Set(filterRaw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
            if !allowed.isEmpty { items = items.filter { allowed.contains($0.targetLang) } }
        }
        let defaultLimit = isFull ? items.count : min(4, items.count)
        let limit = context.task.params["limit"]?.intValue ?? defaultLimit
        return Array(items.prefix(max(1, limit)))
    }

    private func translate(
        item: TranslationProbeItem,
        context: BenchmarkTaskContext
    ) async throws -> (String, BenchmarkHTTPResponse, (prompt: Int, completion: Int)) {
        let sys = "Translate English Markdown into natural \(item.targetName) (\(item.targetNative)). Preserve code spans, CLI flags, URLs, and product names verbatim. Output only the translation."
        let payload: [String: Any] = [
            "model": context.resolvedModelName,
            "messages": [
                ["role": "system", "content": sys],
                ["role": "user", "content": item.source],
            ],
            "max_tokens": context.task.params["max_tokens"]?.intValue ?? 512,
            "temperature": context.temperature,
            "seed": context.seed,
        ]
        let (json, reply) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        return (BenchmarkStats.extractChatContent(json), reply, BenchmarkStats.extractUsage(json))
    }

    private func judgeQE(
        item: TranslationProbeItem,
        hypothesis: String,
        judgeModel: String,
        context: BenchmarkTaskContext
    ) async throws -> (comet: Double, metricX24: Double)? {
        let prompt = """
        Score the translation quality strictly as JSON {"comet": <0.0-1.0 higher is better>, "metricx24": <0.0-25.0 lower is better>}.
        Source (en): \(item.source)
        Reference (\(item.targetLang)): \(item.reference)
        Hypothesis (\(item.targetLang)): \(hypothesis)
        """
        let payload: [String: Any] = [
            "model": judgeModel,
            "messages": [["role": "user", "content": prompt]],
            "max_tokens": 96,
            "temperature": 0.0,
            "seed": context.seed,
        ]
        let (json, _) = try await context.endpoint.postJSON("/v1/chat/completions", payload: payload)
        return TranslationMetrics.parseJudgeJSON(BenchmarkStats.extractChatContent(json))
    }
}
