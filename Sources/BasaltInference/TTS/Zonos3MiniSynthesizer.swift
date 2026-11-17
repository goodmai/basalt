import Foundation
import MLX
import MLXNN
import Tokenizers
import Hub
import BasaltCore

final class Zonos3PipelineModule: Module {
    let config: Zonos3MiniConfig
    @ModuleInfo(key: "lm") var lm: Zonos3MiniLMModule
    @ModuleInfo(key: "codec") var codec: Zonos3MiniCodecModule

    init(_ config: Zonos3MiniConfig) {
        self.config = config
        self._lm.wrappedValue = Zonos3MiniLMModule(config.lm)
        self._codec.wrappedValue = Zonos3MiniCodecModule(config.codec)
        super.init()
    }
}

enum Zonos3WeightLoader {
    static func remapKey(_ key: String, semanticDepth: Int) -> String? {
        if key.hasPrefix("speaker_encoder.") { return nil }
        var mapped = key
        if mapped.hasPrefix("codec.semantic_head.") {
            let rest = mapped.dropFirst("codec.semantic_head.".count)
            if let dot = rest.firstIndex(of: "."), let idx = Int(rest[..<dot]) {
                let suffix = rest[rest.index(after: dot)...]
                mapped = idx == semanticDepth
                    ? "codec.semantic_proj.\(suffix)"
                    : "codec.semantic_blocks.\(idx).\(suffix)"
            }
        } else if mapped.hasPrefix("codec.stages.") {
            let rest = mapped.dropFirst("codec.stages.".count)
            if let dot = rest.firstIndex(of: "."), let rawStage = Int(rest[..<dot]), rawStage.isMultiple(of: 2) {
                let suffix = String(rest[rest.index(after: dot)...])
                    .replacingOccurrences(of: ".ffn.0.", with: ".ffn_in.")
                    .replacingOccurrences(of: ".ffn.3.", with: ".ffn_out.")
                mapped = "codec.moss_stages.\(rawStage / 2).\(suffix)"
            }
        }
        return mapped
    }

    static func needsConv1dTranspose(_ mappedKey: String, ndim: Int) -> Bool {
        guard ndim == 3 else { return false }
        if mappedKey == "codec.to_external_kernel" { return true }
        if mappedKey == "codec.head.mixer.weight" { return true }
        if mappedKey.hasSuffix(".dwconv.weight") { return true }
        if mappedKey.hasPrefix("lm.layers.") && mappedKey.hasSuffix(".attention.conv.weight") { return true }
        return false
    }

    @discardableResult
    static func load(directory: URL, into module: Zonos3PipelineModule) throws(BasaltError) -> Int {
        let files = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent != "voices.safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { return 0 }
        let existing = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        var updates: [(String, MLXArray)] = []
        let semDepth = module.config.codec.semanticDepth
        do {
            for file in files {
                let loaded = try MLX.loadArrays(url: file)
                for (rawKey, rawTensor) in loaded {
                    guard let key = remapKey(rawKey, semanticDepth: semDepth),
                          let target = existing[key] else { continue }
                    let tensor = needsConv1dTranspose(key, ndim: rawTensor.ndim)
                        ? rawTensor.transposed(0, 2, 1)
                        : rawTensor
                    guard tensor.shape == target.shape else {
                        throw BasaltError.weightsCorrupted(
                            path: file.path,
                            reason: "shape mismatch on '\(key)': expected \(target.shape), got \(tensor.shape)"
                        )
                    }
                    let keepDtype: DType = key.hasPrefix("codec.") || key.hasSuffix("balancing_biases")
                        ? .float32
                        : rawTensor.dtype
                    updates.append((key, tensor.asType(keepDtype)))
                }
            }
            if !updates.isEmpty {
                module.update(parameters: ModuleParameters.unflattened(updates))
            }
            eval(module)
            return updates.count
        } catch let error as BasaltError {
            throw error
        } catch {
            throw .weightsCorrupted(path: directory.path, reason: error.localizedDescription)
        }
    }
}

public actor Zonos3MiniSynthesizer: SpeechSynthesizer {
    public nonisolated let config: Zonos3MiniConfig
    public nonisolated let aneVocoder: ANEAcousticVocoder?
    public nonisolated let loadedTensorCount: Int
    public nonisolated let loadedVoiceCount: Int
    public nonisolated let hasTokenizer: Bool
    private var module: Zonos3PipelineModule?
    private let tokenizer: (any Tokenizers.Tokenizer)?
    private let storedVoices: [String: Zonos3VoicePrompt]
    private var epoch = 0

    public init(
        config: Zonos3MiniConfig = .tiny,
        aneVocoder: ANEAcousticVocoder? = nil,
        seed: UInt64 = 3_030
    ) {
        self.config = config
        self.aneVocoder = aneVocoder
        self.loadedTensorCount = 0
        self.loadedVoiceCount = 0
        self.hasTokenizer = false
        self.tokenizer = nil
        self.storedVoices = [:]
        let built = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Zonos3PipelineModule(config)
        }
        eval(built)
        self.module = built
    }

    public init(
        directory: URL,
        modelID: String? = nil,
        aneVocoder: ANEAcousticVocoder? = nil
    ) async throws(BasaltError) {
        let resolved = try Zonos3MiniConfig.from(
            directory: directory,
            fallbackID: modelID ?? directory.lastPathComponent
        )
        self.config = resolved
        self.aneVocoder = aneVocoder
        let built = withRandomState(MLXRandom.RandomState(seed: 3_030)) {
            Zonos3PipelineModule(resolved)
        }
        self.loadedTensorCount = try Zonos3WeightLoader.load(directory: directory, into: built)
        let loadedTok = try? await Self.loadTokenizerIfPresent(in: directory)
        self.tokenizer = loadedTok
        self.hasTokenizer = loadedTok != nil
        let voices = Zonos3PromptBuilder.loadVoiceTable(in: directory)
        self.storedVoices = voices
        self.loadedVoiceCount = voices.count
        self.module = built
    }

    public var voices: [VoiceSpec] {
        get async { TTSVoiceCatalog.zonos3Voices(model: config.modelID, compute: .mlx) }
    }

    public func resolveRoute(_ request: SpeechRequest) async -> SpeechRouteDecision {
        SpeechRouteDecision(
            selected: SpeechCandidate(
                id: config.modelID,
                model: config.modelID,
                compute: .mlx,
                isAvailable: module != nil,
                estimatedTTFAMs: TTSModelFamily.zonos3Mini.estimatedTTFAMs
            )
        )
    }

    public func unload() async {
        epoch &+= 1
        module = nil
        Memory.clearCache()
    }

    public func synthesize(_ request: SpeechRequest) async throws(BasaltError) -> AsyncThrowingStream<AudioChunk, any Error> {
        try request.validate()
        guard module != nil else { throw .modelNotLoaded }
        guard Zonos3Catalog.normalizeLanguage(request.language) != nil else {
            throw .invalidRequestStructure(
                details: "speech: ZONOS3-Mini does not support language '\(request.language ?? "")'"
            )
        }
        if request.input.unicodeScalars.contains(where: { $0.value == 0xE000 || $0.value == 0xE001 }) {
            throw .invalidRequestStructure(details: "speech: ZONOS3-Mini rejects U+E000/U+E001 phoneme brackets")
        }
        let plan = try preparePlan(for: request)
        return Self.makeStream(owner: self, plan: plan)
    }

    nonisolated private static func makeStream(
        owner: Zonos3MiniSynthesizer,
        plan: Zonos3SynthesisPlan
    ) -> AsyncThrowingStream<AudioChunk, any Error> {
        let rate = owner.config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try Task.checkCancellation()
                    let samples = try await owner.generateWaveform(plan: plan)
                    let chunkSamples = max(rate / 5, Zonos3Catalog.frameHopSamples * 3)
                    var offset = 0
                    while offset < samples.count {
                        try Task.checkCancellation()
                        let end = min(samples.count, offset + chunkSamples)
                        continuation.yield(
                            AudioChunk(samples: Array(samples[offset ..< end]), sampleRate: rate, isFinal: end == samples.count)
                        )
                        offset = end
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

extension Zonos3MiniSynthesizer {
    private static func loadTokenizerIfPresent(in directory: URL) async throws -> (any Tokenizers.Tokenizer)? {
        let tokJSON = directory.appending(component: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokJSON.path) else { return nil }
        let hubConfig = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard let tokConfig = try await hubConfig.tokenizerConfig else { return nil }
        return try await PreTrainedTokenizer(tokenizerConfig: tokConfig, tokenizerData: hubConfig.tokenizerData, strict: false)
    }

    private func preparePlan(for request: SpeechRequest) throws(BasaltError) -> Zonos3SynthesisPlan {
        let normalized = Zonos3Catalog.collapseLineBreaks(request.input)
        let lang = Zonos3Catalog.normalizeLanguage(request.language) ?? "en_us"
        let textIDs = encodeText(normalized)
        let voicePrompt = try resolveVoicePrompt(voice: request.voice, language: lang, referenceAudio: request.referenceAudio)
        let exag = Float(request.exaggeration ?? 0.5)
        let temp = request.exaggeration == nil
            ? config.generation.temperature
            : max(Float(0.2), config.generation.temperature * (0.85 + 0.3 * exag))
        let cfg: Float = if voicePrompt.embedding != nil && voicePrompt.codes == nil {
            request.exaggeration != nil ? (1.2 + 0.6 * exag) : 1.5
        } else {
            1.0
        }
        let estimatedFrames = loadedTensorCount > 0
            ? min(450, max(24, textIDs.count * 4 + 18))
            : min(16, max(12, textIDs.count / 2 + 11))
        let seed = UInt64(request.voice.unicodeScalars.reduce(3_030) { ($0 &* 31 &+ Int($1.value)) & 0x7fff_ffff })
        return Zonos3SynthesisPlan(
            textIDs: textIDs,
            voicePrompt: voicePrompt,
            conditionBuckets: config.lm.conditionBucketCounts,
            temperature: temp,
            cfgScale: cfg,
            maxNewFrames: estimatedFrames,
            speed: request.speed,
            seed: seed,
            epoch: epoch
        )
    }

    private func encodeText(_ text: String) -> [Int32] {
        if let tokenizer {
            let encoded = tokenizer.encode(text: text, addSpecialTokens: true)
            let cap = config.lm.textVocab
            var ids = encoded.map { Int32(max(0, min(cap - 1, $0))) }
            if ids.first != 2 { ids.insert(2, at: 0) }
            if ids.last != 3 { ids.append(3) }
            return ids
        }
        let usable = max(8, config.lm.textVocab - 6)
        var out: [Int32] = [2]
        for byte in text.utf8.prefix(512) {
            out.append(Int32(6 + (Int(byte) % usable)))
        }
        out.append(3)
        return out
    }

    private func resolveVoicePrompt(voice: String, language: String, referenceAudio: String?) throws(BasaltError) -> Zonos3VoicePrompt {
        let trimmed = voice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let voiceKey = (trimmed == "alloy" || trimmed == "default") ? "savannah" : trimmed
        if let referenceAudio {
            let seed = voiceKey.unicodeScalars.reduce(17) { ($0 &* 31 &+ Int($1.value)) & 0x7fff }
            let emb = try ReferenceAudioConditioner.speakerEmbedding(
                from: referenceAudio, voiceSeed: seed, dimension: config.lm.speakerEmbeddingDim, maxSeconds: 10.0
            )
            return Zonos3VoicePrompt(codes: nil, embedding: emb)
        }
        if let stored = storedVoices[voiceKey] ?? storedVoices["savannah"] {
            let nativeLang = Zonos3Catalog.defaultVoices.first { $0.id.lowercased() == voiceKey }?.language ?? "en-US"
            let normalizedNative = Zonos3Catalog.normalizeLanguage(nativeLang) ?? "en_us"
            let sameLanguage = normalizedNative.split(separator: "_").first == language.split(separator: "_").first
            if !sameLanguage, let emb = stored.embedding {
                return Zonos3VoicePrompt(codes: nil, embedding: emb)
            }
            if let codes = stored.codes {
                return Zonos3VoicePrompt(codes: Array(codes.prefix(config.lm.referenceMaxFrames)), embedding: nil)
            }
            return stored
        }
        let seed = voiceKey.unicodeScalars.reduce(23) { ($0 &* 31 &+ Int($1.value)) & 0x7fff }
        let emb = try ReferenceAudioConditioner.speakerEmbedding(
            from: nil, voiceSeed: seed, dimension: config.lm.speakerEmbeddingDim
        )
        return Zonos3VoicePrompt(codes: nil, embedding: emb)
    }
}

extension Zonos3MiniSynthesizer {
    private func generateWaveform(plan: Zonos3SynthesisPlan) throws(BasaltError) -> [Float] {
        guard let module, epoch == plan.epoch else { throw .modelNotLoaded }
        let built = Zonos3PromptBuilder.build(plan: plan, config: config)
        let (rows, eosFrame) = decodeRows(module: module, built: built, plan: plan)
        return renderAudio(module: module, rows: rows, eosFrame: eosFrame, silenceRows: built.silenceRows, plan: plan)
    }

    private struct TwinStep {
        let input: MLXArray
        let cache: Zonos3MiniCache
        let conditioning: Zonos3Conditioning
        let scale: Float
    }

    private func stepLogits(
        module: Zonos3PipelineModule,
        condInput: MLXArray,
        condCache: Zonos3MiniCache,
        conditioning: Zonos3Conditioning,
        twin: TwinStep?
    ) -> MLXArray {
        let condOut = module.lm.forward(
            ids: condInput, cache: condCache, conditioning: conditioning, logitsToKeep: 1
        )[0]
        guard let twin else {
            eval(condOut)
            return condOut
        }
        let twinOut = module.lm.forward(
            ids: twin.input, cache: twin.cache, conditioning: twin.conditioning, logitsToKeep: 1
        )[0]
        let guided = twinOut + twin.scale * (condOut - twinOut)
        eval(guided)
        return guided
    }

    private func decodeRows(
        module: Zonos3PipelineModule,
        built: Zonos3BuiltPrompt,
        plan: Zonos3SynthesisPlan
    ) -> (rows: [[Int32]], eosFrame: Int?) {
        let c = config.lm
        let n = c.nCodebooks
        let pad = Int32(c.audioPadId), textPad = Int32(c.textVocab), eoa = Int32(c.eoaId)
        let cache = Zonos3MiniCache()
        let twinCache = built.twinRows != nil ? Zonos3MiniCache() : nil
        let hasAudioPrefix = built.rows.last?.prefix(n).contains(where: { $0 != pad }) ?? false
        let promptTensor = MLXArray(built.rows.flatMap { $0 }).reshaped(built.rows.count, n + 1)
        let initialTwin = built.twinRows.flatMap { tRows in
            twinCache.map { tc in
                TwinStep(
                    input: MLXArray(tRows.flatMap { $0 }).reshaped(tRows.count, n + 1),
                    cache: tc,
                    conditioning: built.twinConditioning ?? Zonos3Conditioning(),
                    scale: plan.cfgScale
                )
            }
        }
        var logits = stepLogits(
            module: module, condInput: promptTensor, condCache: cache,
            conditioning: built.conditioning, twin: initialTwin
        )
        var history = built.rows.map { Array($0.prefix(n)) }
        var generatedRows: [[Int32]] = []
        var rng = plan.seed == 0 ? 0x9E37_79B9_7F4A_7C15 : plan.seed
        var eosFrame: Int? = nil
        var countdown = -1
        var remain = plan.maxNewFrames
        while true {
            var row = Zonos3Sampler.sampleRow(
                logits: logits, history: history, generated: generatedRows.count,
                config: config, temperature: plan.temperature, rng: &rng
            )
            if !hasAudioPrefix, generatedRows.count < n - 1 {
                for col in (generatedRows.count + 1) ..< n { row[col] = pad }
            }
            generatedRows.append(row)
            history.append(Array(row.prefix(n)))
            remain -= 1
            if eosFrame == nil, let maxCol = (0 ..< n).filter({ row[$0] == eoa }).max() {
                eosFrame = max(0, (generatedRows.count - 1) - maxCol)
                countdown = n + 1
            }
            if countdown > 0 { countdown -= 1 }
            if remain <= 0 || countdown == 0 { break }
            let stepInput = MLXArray(row + [textPad]).reshaped(1, n + 1)
            let nextTwin = twinCache.map { tc in
                TwinStep(input: stepInput, cache: tc, conditioning: Zonos3Conditioning(), scale: plan.cfgScale)
            }
            logits = stepLogits(
                module: module, condInput: stepInput, condCache: cache,
                conditioning: Zonos3Conditioning(), twin: nextTwin
            )
        }
        return (generatedRows, eosFrame)
    }

    private func renderAudio(
        module: Zonos3PipelineModule,
        rows: [[Int32]],
        eosFrame: Int?,
        silenceRows: [[Int32]]?,
        plan: Zonos3SynthesisPlan
    ) -> [Float] {
        let c = config.lm, g = config.generation, n = c.nCodebooks
        var (take, _) = Zonos3DelaySchedule.takeCodes(
            rows: rows, eosFrame: eosFrame, prefillRows: silenceRows, lm: c, overlapFrames: g.overlapFrames
        )
        let validFrames = take.map { frame in
            frame.map { tok in (tok >= 0 && tok < Int32(c.codebookSize)) ? tok : 0 }
        }
        take = validFrames.isEmpty ? [Array(g.silenceFrame.prefix(n))] : validFrames
        let codeTensor = MLXArray(take.flatMap { $0 }).reshaped(take.count, n)
        let decoded = module.codec.decode(codes: codeTensor)
        eval(decoded)
        let rawAudio = decoded.asType(.float32).asArray(Float.self)
        var staged = Zonos3DelaySchedule.outputStage(
            rawAudio, headroomDB: g.headroomDB, outputLimitDBFS: g.outputLimitDBFS, fadeOutMs: 8.0, sampleRate: config.sampleRate
        )
        if loadedTensorCount == 0 && TTSWaveformMetrics.rms(staged) < 1e-3 {
            staged = TTSWaveformMetrics.normalizeSoft(rawAudio, peakTarget: 0.5)
        }
        if abs(plan.speed - 1.0) > 1e-3 {
            let srcRate = max(4_000, Int((Double(config.sampleRate) * plan.speed).rounded()))
            staged = TTSWaveformMetrics.resampleLinear(staged, from: srcRate, to: config.sampleRate)
        }
        return staged
    }
}
