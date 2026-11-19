import Foundation
import MLX
import MLXLMCommon

enum KVProbeLanguage: String, Codable, Sendable, CaseIterable {
    case en
    case ru
    case ko
    case ja

    var filler: [String] {
        switch self {
        case .en:
            KVHeadClassifier.filler
        case .ru:
            [
                "Река текла мимо старой мельницы, где колесо остановилось много лет назад.",
                "Каждое утро пекарь открывал ставни и ставил свежий хлеб на подоконник.",
                "Во вторник комитет долго обсуждал бюджет ремонта крыши новой библиотеки.",
                "Поезда к побережью отправлялись каждый час, а последний ждал паром.",
                "В архиве книги за три столетия занимали полки вдоль северной стены.",
                "Оркестр долго настраивал инструменты, прежде чем дирижёр поднял палочку.",
            ]
        case .ko:
            [
                "강물은 수년 전에 바퀴가 멈춘 오래된 방앗간 옆을 조용히 흘러갔다.",
                "매일 아침 제빵사는 덧문을 열고 첫 빵을 창틀 위에 올려놓았다.",
                "위원회는 화요일마다 새 도서관 지붕 예산을 두고 길게 논의했다.",
                "해안으로 가는 열차는 매시간 출발했고 마지막 열차는 연락선을 기다렸다.",
                "보관소 북쪽 벽 선반에는 삼 세기에 걸친 장부가 가득 꽂혀 있었다.",
                "지휘자가 마침내 지휘봉을 들기 전까지 오케스트라는 오랫동안 조율했다.",
            ]
        case .ja:
            [
                "川は数年前に水車が止まった古い製粉所のそばを静かに流れていた。",
                "毎朝パン屋は雨戸を開けて最初のパンを窓辺に並べた。",
                "委員会は火曜日に新しい図書館の屋根の予算について議論した。",
                "海岸行きの列車は毎時出発し最終便は連絡船を待っていた。",
                "書庫の北の壁の棚には三世紀にわたる帳簿が並んでいた。",
                "指揮者がついに指揮棒を上げるまでオーケストラは長く調律した。",
            ]
        }
    }

    var places: [String] {
        switch self {
        case .en: KVHeadClassifier.places
        case .ru: ["северных ворот", "восточной башни", "западного моста", "южной гавани"]
        case .ko: ["북문", "동쪽 탑", "서쪽 다리", "남쪽 항구"]
        case .ja: ["北門", "東の塔", "西の橋", "南の港"]
        }
    }

    func needle(place: String, key: Int) -> String {
        switch self {
        case .en: " The pass key for the \(place) is \(key). Remember it: \(key). "
        case .ru: " Код доступа для \(place) равен \(key). Запомните его: \(key). "
        case .ko: " \(place)의 출입 암호는 \(key)입니다. 꼭 기억하세요: \(key). "
        case .ja: " \(place)の暗証番号は \(key) です。覚えておいてください: \(key). "
        }
    }

    func question(place: String) -> String {
        switch self {
        case .en: " What is the pass key for the \(place)? The pass key for the \(place) is"
        case .ru: " Какой код доступа для \(place)? Код доступа для \(place) равен"
        case .ko: " \(place)의 출입 암호는 무엇입니까? \(place)의 출입 암호는"
        case .ja: " \(place)の暗証番号は何ですか？\(place)の暗証番号は"
        }
    }

    func answer(key: Int) -> String { " \(key)." }
}

struct KVHeadCoordinate: Hashable, Codable, Sendable, Comparable {
    let layer: Int
    let head: Int

    static func < (lhs: KVHeadCoordinate, rhs: KVHeadCoordinate) -> Bool {
        (lhs.layer, lhs.head) < (rhs.layer, rhs.head)
    }
}

enum CrossLingualHeadKV {
    static let defaultPersistence = 0.9
    static let zeroShotThreshold = 0.75
    static let defaultRRFConstant = 60.0

    enum TransferMode: String, Codable, Sendable, Equatable {
        case zeroShot
        case reciprocalRankFused
    }

    struct TransferPolicy: Sendable, Equatable {
        var persistence: Double = CrossLingualHeadKV.defaultPersistence
        var threshold: Double = CrossLingualHeadKV.zeroShotThreshold
        var rrfConstant: Double = CrossLingualHeadKV.defaultRRFConstant
        var depth: Int? = nil
    }

    struct SourceSelection: Codable, Sendable, Equatable {
        let mask: KVHeadMask
        let ranking: [KVHeadCoordinate]
    }

    struct TransferResult: Codable, Sendable, Equatable {
        let mask: KVHeadMask
        let rbo: Double
        let mode: TransferMode
    }

    static func deduplicated<Element: Hashable>(_ elements: [Element]) -> [Element] {
        var seen = Set<Element>(minimumCapacity: elements.count)
        return elements.filter { seen.insert($0).inserted }
    }

    static func rankBiasedOverlap<Element: Hashable>(
        _ lhs: [Element], _ rhs: [Element], persistence p: Double = defaultPersistence, depth: Int? = nil
    ) -> Double {
        precondition(p.isFinite && p >= 0 && p <= 1, "RBO persistence must lie in [0, 1]")
        let left = deduplicated(lhs)
        let right = deduplicated(rhs)
        if left.isEmpty && right.isEmpty { return 1.0 }
        let maxDepth = max(left.count, right.count)
        let limit = min(depth ?? maxDepth, maxDepth)
        guard limit > 0, !left.isEmpty, !right.isEmpty else { return 0.0 }
        var seenLeft = Set<Element>(minimumCapacity: limit)
        var seenRight = Set<Element>(minimumCapacity: limit)
        var overlap = 0
        var weightedAgreement = 0.0
        var weightSum = 0.0
        var weight = 1.0
        for index in 0 ..< limit {
            if index < left.count, seenLeft.insert(left[index]).inserted, seenRight.contains(left[index]) {
                overlap += 1
            }
            if index < right.count, seenRight.insert(right[index]).inserted, seenLeft.contains(right[index]) {
                overlap += 1
            }
            weightedAgreement += weight * (Double(overlap) / Double(index + 1))
            weightSum += weight
            weight *= p
        }
        return weightSum > 0 ? weightedAgreement / weightSum : 0.0
    }

    static func reciprocalRankScores(
        _ rankings: [[KVHeadCoordinate]], k: Double = defaultRRFConstant
    ) -> [(head: KVHeadCoordinate, score: Double)] {
        precondition(k.isFinite && k >= 0, "RRF constant k must be non-negative")
        var scores: [KVHeadCoordinate: Double] = [:]
        for ranking in rankings {
            for (index, coord) in deduplicated(ranking).enumerated() {
                scores[coord, default: 0] += 1.0 / (k + Double(index + 1))
            }
        }
        return scores.map { (head: $0.key, score: $0.value) }.sorted {
            $0.score != $1.score ? $0.score > $1.score : $0.head < $1.head
        }
    }

    static func reciprocalRankFusion(
        _ rankings: [[KVHeadCoordinate]], heads: [Int], keep: Int, k: Double = defaultRRFConstant
    ) -> KVHeadMask {
        precondition(k.isFinite && k >= 0, "RRF constant k must be non-negative")
        let allCoords = heads.enumerated().flatMap { layer, count in (0 ..< count).map { KVHeadCoordinate(layer: layer, head: $0) } }
        guard keep > 0 else { return KVHeadMask(retrieval: heads.map { _ in [] }) }
        guard keep < allCoords.count else { return KVHeadMask(retrieval: heads.map { Array(0 ..< $0) }) }
        let valid = Set(allCoords)
        let filtered = rankings.map { $0.filter { valid.contains($0) } }
        let scored = Dictionary(uniqueKeysWithValues: reciprocalRankScores(filtered, k: k))
        let ordered = allCoords.sorted {
            let left = scored[$0, default: 0]
            let right = scored[$1, default: 0]
            return left != right ? left > right : $0 < $1
        }
        var retrieval = heads.map { _ in [Int]() }
        for coord in ordered.prefix(keep) {
            retrieval[coord.layer].append(coord.head)
        }
        return KVHeadMask(retrieval: retrieval.map { $0.sorted() })
    }

    static func resolveMask(
        cachedMask: KVHeadMask,
        sourceRanking: [KVHeadCoordinate],
        targetRanking: [KVHeadCoordinate],
        heads: [Int],
        keep: Int? = nil,
        policy: TransferPolicy = TransferPolicy()
    ) -> TransferResult {
        let rbo = rankBiasedOverlap(sourceRanking, targetRanking, persistence: policy.persistence, depth: policy.depth)
        let fits = KVHeadSelection(mask: cachedMask, meanKL: 0, answersKept: true).fits(heads)
        if rbo >= policy.threshold && fits {
            return TransferResult(mask: cachedMask, rbo: rbo, mode: .zeroShot)
        }
        let budget = keep ?? cachedMask.retrieval.map(\.count).reduce(0, +)
        let fused = reciprocalRankFusion([sourceRanking, targetRanking], heads: heads, keep: budget, k: policy.rrfConstant)
        return TransferResult(mask: fused, rbo: rbo, mode: .reciprocalRankFused)
    }
}

extension CrossLingualHeadKV {
    static func rankHeads(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer,
        language: KVProbeLanguage, length: Int = 1024
    ) throws -> (heads: [Int], ranking: [KVHeadCoordinate]) {
        let probe = KVHeadClassifier.probe(
            tokenizer: tokenizer, length: length, depths: KVHeadClassifier.calibrationDepths[0],
            seed: 0, language: language)
        let ref = try KVHeadClassifier.reference(model, probe)
        let heads = KVHeadClassifier.attentionHeads(ref.dense)
        let candidates = heads.enumerated().flatMap { layer, count in
            (0 ..< count).map { KVHeadCoordinate(layer: layer, head: $0) }
        }
        let empty = heads.map { _ in [Int]() }
        let scored = try candidates.map { coord -> (coord: KVHeadCoordinate, loss: Float) in
            try Task.checkCancellation()
            var trial = empty
            trial[coord.layer] = [coord.head]
            let value = KVHeadClassifier.divergence(model, ref, trial).total
            return (coord, value.isFinite ? value : .infinity)
        }
        let ordered = scored.sorted {
            $0.loss != $1.loss ? $0.loss < $1.loss : $0.coord < $1.coord
        }.map(\.coord)
        return (heads, ordered)
    }

    static func transferSelection(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer,
        cached: SourceSelection,
        targetLanguage: KVProbeLanguage, length: Int = 1024,
        policy: TransferPolicy = TransferPolicy()
    ) throws -> (transfer: TransferResult, selection: KVHeadSelection) {
        let (heads, targetRanking) = try rankHeads(
            model: model, tokenizer: tokenizer, language: targetLanguage, length: length)
        let transfer = resolveMask(
            cachedMask: cached.mask, sourceRanking: cached.ranking, targetRanking: targetRanking,
            heads: heads, policy: policy)
        let references = try KVHeadClassifier.calibrationDepths.enumerated().map { seed, depths in
            try Task.checkCancellation()
            let probe = KVHeadClassifier.probe(
                tokenizer: tokenizer, length: length, depths: depths, seed: seed, language: targetLanguage)
            return try KVHeadClassifier.reference(model, probe)
        }
        let gates = references.map { KVHeadClassifier.divergence(model, $0, transfer.mask.retrieval) }
        let selection = KVHeadSelection(
            mask: transfer.mask, meanKL: KVHeadClassifier.worst(gates.map(\.mean)),
            answersKept: gates.allSatisfy(\.answersKept))
        return (transfer, selection)
    }

    static func fuseMultilingual(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer,
        languages: [KVProbeLanguage] = KVProbeLanguage.allCases,
        ratio: Double, length: Int = 1024,
        policy: TransferPolicy = TransferPolicy()
    ) throws -> (mask: KVHeadMask, mode: TransferMode, minRBO: Double) {
        let activeLanguages = languages.isEmpty ? [.en] : deduplicated(languages)
        var rankings: [[KVHeadCoordinate]] = []
        var heads: [Int] = []
        for lang in activeLanguages {
            try Task.checkCancellation()
            let ranked = try rankHeads(model: model, tokenizer: tokenizer, language: lang, length: length)
            heads = ranked.heads
            rankings.append(ranked.ranking)
        }
        let keep = KVHeadClassifier.kept(heads, ratio: ratio)
        let primaryMask = reciprocalRankFusion([rankings[0]], heads: heads, keep: keep, k: policy.rrfConstant)
        guard rankings.count > 1 else {
            return (primaryMask, .zeroShot, 1.0)
        }
        let overlaps = rankings.dropFirst().map {
            rankBiasedOverlap(rankings[0], $0, persistence: policy.persistence, depth: policy.depth)
        }
        let minOverlap = overlaps.min() ?? 1.0
        if minOverlap >= policy.threshold {
            return (primaryMask, .zeroShot, minOverlap)
        }
        let fused = reciprocalRankFusion(rankings, heads: heads, keep: keep, k: policy.rrfConstant)
        return (fused, .reciprocalRankFused, minOverlap)
    }
}
