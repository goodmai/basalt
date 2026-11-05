import Foundation
import Testing
import MLX
import MLXGuidedGeneration
@testable import BasaltCore
@testable import BasaltInference

@Suite("Per-step cost of the JSON constraint on a large vocabulary (#141)", .serialized)
struct ConstraintCostBenchmarkTests {
    static let vocabularySize = 151_936
    static let schema = #"""
        {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"}, "hourly": {"type": "boolean"}},
         "required": ["city", "unit", "days", "hourly"]}
        """#
    static let document = #"{"city": "Reykjavik", "unit": "celsius", "days": 3, "hourly": false}"#
    static let enabled = ProcessInfo.processInfo.environment["BASALT_BENCH"] != nil

    static func syntheticVocabulary() throws -> GrammarVocabulary {
        let alphabet = Array("etaoinshrdlucmfwypvbgkqjxz0123456789 _\"{}:,.-")
        var pieces = (0..<128).map { String(UnicodeScalar(UInt8($0))) }
        var serial = 0
        while pieces.count < vocabularySize - 1 {
            var piece = "", rest = serial
            repeat {
                piece.append(alphabet[rest % alphabet.count])
                rest /= alphabet.count
            } while rest > 0
            pieces.append(piece)
            serial += 1
        }
        pieces.append("<eos>")
        return try GrammarVocabulary(pieces: pieces, vocabType: .raw, stopTokens: [vocabularySize - 1], markers: ReasoningMarkers())
    }

    static func micros(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1e6 + Double(duration.components.attoseconds) / 1e12
    }

    static func median(_ samples: [Double]) -> Double { samples.sorted()[samples.count / 2] }

    static func report(_ label: String, _ samples: [Double]) {
        let sorted = samples.sorted()
        print(String(format: "[bench #141] %@: median %.1f µs, p90 %.1f µs, n=%d",
                     label, median(samples), sorted[sorted.count * 9 / 10], samples.count))
    }

    @Test("Per decode step: grammar fill, mask post-processing, logit masking", .enabled(if: enabled, "set BASALT_BENCH=1"))
    func perStepMaskLatency() throws {
        let clock = ContinuousClock()
        let vocabulary = try Self.syntheticVocabulary()
        let prepared = try JSONSchemaGate.prepare(Self.schema, strict: false).schema
        var grammar: GrammarConstraint?
        Self.report("grammar compile (miss)", [Self.micros(try clock.measure {
            grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: prepared)
        })])
        let session = GrammarSession(schema: prepared, grammar: try #require(grammar))
        Self.report("processor construction per request", (0..<20).map { _ in
            Self.micros(clock.measure { _ = JSONConstraintProcessor(session: session, vocabulary: vocabulary, phase: .constrained) })
        })
        var processor = JSONConstraintProcessor(session: session, vocabulary: vocabulary, phase: .constrained)
        let logits = MLXRandom.normal([1, Self.vocabularySize]).asType(.bfloat16)
        eval(logits)
        var fill: [Double] = [], allowed: [Double] = [], bare: [Double] = []
        var build: [Bool: [Double]] = [:], applied: [Bool: [Double]] = [:]
        for byte in Array(Self.document.utf8) {
            _ = processor.phase
            fill.append(Self.micros(clock.measure { _ = session.mask() }))
            allowed.append(Self.micros(clock.measure { _ = processor.allowedTokens() }))
            let current = processor
            var masked: [Bool: MLXArray] = [:]
            for fused in [false, true] {
                build[fused, default: []].append(Self.micros(clock.measure {
                    masked[fused] = current.allowedTokens().map { current.masked(logits, words: $0, fused: fused) } ?? logits
                }))
                applied[fused, default: []].append(Self.micros(clock.measure { eval(masked[fused]!) }))
            }
            let shifted = logits + 0
            bare.append(Self.micros(clock.measure { eval(shifted) }))
            #expect(masked[true]![0, Int(byte)].item(Float.self) > -Float.infinity, "byte \(byte) must stay allowed")
            #expect(arrayEqual(masked[true]!, masked[false]!).item(Bool.self), "byte \(byte): the kernel must equal the ops path")
            processor.didSample(token: MLXArray(Int32(byte)))
        }
        Self.report("xgrammar fill", fill)
        Self.report("allowedTokens (fill + post-processing)", allowed)
        Self.report("process() graph build, CPU, take/shift/where", build[false]!)
        Self.report("process() graph build, CPU, fused kernel", build[true]!)
        Self.report("eval of the masked logits, take/shift/where", applied[false]!)
        Self.report("eval of the masked logits, fused kernel", applied[true]!)
        Self.report("eval of unmasked logits (baseline)", bare)
    }

    @Test("Schema preparation on every request", .enabled(if: enabled, "set BASALT_BENCH=1"))
    func preparationLatency() throws {
        let clock = ContinuousClock()
        let samples = try (0..<200).map { _ in
            Self.micros(try clock.measure { _ = try ResponseFormat.jsonSchema(Self.schema).prepared() })
        }
        Self.report("JSONSchemaGate.prepare", samples)
    }

    @Test("Schema preparation plus grammar acquisition when the pool already holds the schema", .enabled(if: enabled, "set BASALT_BENCH=1"))
    func pooledAcquisitionLatency() async throws {
        let clock = ContinuousClock()
        let engine = try await ConstrainedModel.engine()
        let request = ConstrainedModel.request(format: .jsonSchema(Self.schema), maxTokens: 120)
        var samples: [Double] = []
        for _ in 0..<10 {
            _ = try await engine.generate(request: request)
            #expect(await engine.idleGrammars.values.joined().count == 1)
            samples.append(Self.micros(try await clock.measure { _ = try await engine.outputConstraint(for: request) }))
            #expect(await engine.idleGrammars.values.joined().isEmpty)
        }
        Self.report("prepare + pooled grammar acquisition (hit)", samples)
    }
}

@Suite("Pool of compiled grammars across requests (#141)", .serialized)
struct GrammarPoolTests {
    static func schema(_ index: Int) -> String { #"{"enum": ["v\#(index)"]}"# }

    static func pooledSchemas(_ engine: MLXInferenceEngine) async -> Set<String> {
        Set(await engine.idleGrammars.filter { !$0.value.isEmpty }.keys)
    }

    @Test("A new schema past the pool's limit evicts only the schema used longest ago, so the schemas an agent loop keeps reusing stay compiled")
    func evictsLeastRecentlyUsed() async throws {
        let engine = try await ConstrainedModel.engine()
        func generate(_ index: Int) async throws {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(Self.schema(index)), maxTokens: 20))
        }
        let limit = MLXInferenceEngine.grammarSchemaLimit
        for index in 0..<limit { try await generate(index) }
        try await generate(0)
        try await generate(limit)
        let prepared = try [0, 1, limit].map { "json_schema:" + (try JSONSchemaGate.prepare(Self.schema($0), strict: false).schema) }
        let pooled = await Self.pooledSchemas(engine)
        #expect(pooled.count == limit)
        #expect(pooled.contains(prepared[0]))
        #expect(!pooled.contains(prepared[1]))
        #expect(pooled.contains(prepared[2]))
    }
}

@Suite("Logit masking from the grammar bitmask (#141)")
struct BitmaskMaskingTests {
    @Test("Tokens on the sign bit of a mask word stay allowed and their neighbours a string refuses stay masked")
    func signBitTokens() throws {
        let vocabulary = try GrammarVocabulary(
            pieces: (0..<128).map { String(UnicodeScalar(UInt8($0))) }, vocabType: .raw, stopTokens: [127], markers: ReasoningMarkers())
        var processor = try GrammarVocabularyTests.processor(vocabulary)
        processor.didSample(token: MLXArray(Int32(UInt8(ascii: "\""))))
        let masked = processor.process(logits: MLXArray.zeros([1, 128]))
        let allowed = (0..<128).filter { masked[0, $0].item(Float.self) == 0 }
        #expect(allowed.contains(31 + 32))
        #expect(allowed.contains(31 + 64))
        #expect(!allowed.contains(31))
        #expect(allowed == (32..<127).map { $0 }, "\(allowed)")
    }

    @Test("The fused mask kernel equals the take/shift/where path bit for bit on random words, sign-bit words and widths off a multiple of 32",
          arguments: [(100, 100), (100, 128), (1000, 977), (1000, 1013), (4097, 4097)], [DType.float32, .bfloat16, .float16])
    func fusedKernelMatchesOps(shape: (Int, Int), dtype: DType) throws {
        let (vocabularySize, width) = shape
        let vocabulary = try GrammarVocabulary(
            pieces: (0..<vocabularySize).map { "t\($0)" }, vocabType: .raw, stopTokens: [vocabularySize - 1], markers: ReasoningMarkers())
        let processor = try GrammarVocabularyTests.processor(vocabulary)
        var generator = SplitMix64(seed: UInt64(vocabularySize * 7919 + width))
        var words = (0..<(vocabularySize + 31) / 32).map { _ in Int32(truncatingIfNeeded: generator.next()) }
        words[0] = Int32.min
        words[words.count / 2] = -1
        words[words.count - 1] |= Int32.min
        let logits = MLXRandom.normal([2, width], key: MLXRandom.key(UInt64(width))).asType(dtype)
        let fused = processor.masked(logits, words: words, fused: true)
        let ops = processor.masked(logits, words: words, fused: false)
        eval(fused, ops)
        #expect(fused.dtype == dtype && fused.shape == ops.shape)
        #expect(arrayEqual(fused, ops).item(Bool.self))
        #expect(fused[0, 31].item(Float.self) == logits[0, 31].item(Float.self))
        #expect((fused .== -Float.infinity).sum().item(Int.self) > 0)
    }
}

@Suite("Fused grammar mask kernel: release decode tok/s with a JSON schema, kernel on vs off, interleaved (#346)", .serialized)
struct MaskKernelABRealModelTests {
    static let env = ProcessInfo.processInfo.environment
    static let rounds = env["BASALT_MASK_AB_ROUNDS"].flatMap(Int.init) ?? 8
    static let schema = #"""
        {"type": "object", "properties": {"cities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "country": {"type": "string"}, "population": {"type": "integer"},
            "summary": {"type": "string"}}, "required": ["name", "country", "population", "summary"]}}},
         "required": ["cities"]}
        """#

    static var models: [String] {
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_MASK_MODELS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static func median(_ samples: [Double]) -> Double { samples.sorted()[samples.count / 2] }

    @Test("Greedy output is the same with the kernel on and off; the median tok/s gain is logged against the 2% bar",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_MASK_MODELS"), arguments: models)
    func interleavedDecode(entry: String) async throws {
        let path = GoldenLogitsRealModelTests.path(entry)
        let log = try await ModelRunLog("mask-kernel-ab", modelPath: path)
        defer {
            JSONConstraintProcessor.fusedMask.store(true, ordering: .relaxed)
            log.releaseGPU()
        }
        let engine = MLXInferenceEngine()
        try await log.step("load") { try await engine.load(modelPath: path) }
        let request = GenerationRequest(
            prompt: "Describe twelve European capital cities, one sentence each.", maxTokens: 384, temperature: 0
        ).with {
            $0.enableThinking = false
            $0.responseFormat = .jsonSchema(Self.schema)
        }
        _ = try await log.step("warm-up") { try await engine.generate(request: request) }
        var rates: [Bool: [Double]] = [:], texts: [Bool: Set<String>] = [:]
        for round in 0..<Self.rounds {
            for fused in round.isMultiple(of: 2) ? [false, true] : [true, false] {
                JSONConstraintProcessor.fusedMask.store(fused, ordering: .relaxed)
                let response = try await engine.generate(request: request)
                rates[fused, default: []].append(response.tokensPerSecond)
                texts[fused, default: []].insert(response.generatedText)
                log.record("round \(round) kernel \(fused ? "on" : "off")",
                           String(format: "%ld tokens, %.2f tok/s, %@", response.completionTokens, response.tokensPerSecond, log.host()))
            }
        }
        let on = Self.median(rates[true] ?? [0]), off = Self.median(rates[false] ?? [0])
        let gain = (on / off - 1) * 100
        log.record("result", String(format: "MASK-KERNEL on %.2f tok/s, off %.2f tok/s, gain %+.2f%%, %@",
                                    on, off, gain, gain >= 2 ? "clears the 2% bar" : "below the 2% bar"))
        #expect(texts[true] == texts[false], "\(log.model): greedy text differs between kernel on and off")
    }
}
