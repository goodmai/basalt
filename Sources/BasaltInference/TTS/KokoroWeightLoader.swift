import Foundation
import MLX
import MLXNN
import BasaltCore

package enum KokoroWeightLoader {
    package struct StrictLoadReport: Sendable, Equatable {
        package let checkpointKeyCount: Int
        package let modelParameterCount: Int
        package let fusedWeightNormCount: Int
    }

    package static func findModelSafetensors(in directory: URL) -> URL? {
        let preferred = ["kokoro-v1_0.safetensors", "model.safetensors"]
        for name in preferred {
            let candidate = directory.appending(component: name)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }

    package static func loadStrict(
        from fileURL: URL,
        into model: KokoroStyleTTS2Model
    ) throws(BasaltError) -> StrictLoadReport {
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: fileURL)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "kokoro: failed to read safetensors: \(error.localizedDescription)")
        }
        let (mapped, fusedCount) = try collectMappedTensors(raw, path: fileURL.path)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let updates = try buildStrictUpdates(expected: expected, mapped: mapped, path: fileURL.path)
        do {
            try model.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        } catch {
            throw .weightsCorrupted(path: fileURL.path, reason: "kokoro: strict update failed: \(error.localizedDescription)")
        }
        eval(model)
        return StrictLoadReport(
            checkpointKeyCount: raw.count,
            modelParameterCount: expected.count,
            fusedWeightNormCount: fusedCount
        )
    }

    private static func collectMappedTensors(
        _ raw: [String: MLXArray],
        path: String
    ) throws(BasaltError) -> (mapped: [String: MLXArray], fusedCount: Int) {
        var consumed = Set<String>()
        var mapped: [String: MLXArray] = [:]
        var fusedCount = 0
        for key in raw.keys.sorted() where !consumed.contains(key) {
            if key.hasSuffix(".weight_g") {
                let prefix = String(key.dropLast(9))
                guard let g = raw[key], let v = raw[prefix + ".weight_v"] else {
                    throw .weightsCorrupted(path: path, reason: "kokoro: missing weight_v for '\(key)'")
                }
                consumed.insert(key)
                consumed.insert(prefix + ".weight_v")
                mapped[remapKey(prefix + ".weight")] = fuseWeightNorm(g: g, v: v, keyPrefix: prefix)
                fusedCount += 1
            } else if key.hasSuffix(".weight_v") {
                let prefix = String(key.dropLast(9))
                guard raw[prefix + ".weight_g"] != nil else {
                    throw .weightsCorrupted(path: path, reason: "kokoro: missing weight_g for '\(key)'")
                }
            } else {
                consumed.insert(key)
                mapped[remapKey(key)] = convertTensor(raw[key]!, originalKey: key)
            }
        }
        let unconsumed = Set(raw.keys).subtracting(consumed)
        guard unconsumed.isEmpty else {
            throw .weightsCorrupted(path: path, reason: "kokoro: \(unconsumed.count) unconsumed checkpoint tensors")
        }
        return (mapped, fusedCount)
    }

    private static func buildStrictUpdates(
        expected: [String: MLXArray],
        mapped: [String: MLXArray],
        path: String
    ) throws(BasaltError) -> [(String, MLXArray)] {
        let expectedKeys = Set(expected.keys)
        let mappedKeys = Set(mapped.keys)
        let missing = expectedKeys.subtracting(mappedKeys)
        let extra = mappedKeys.subtracting(expectedKeys)
        guard missing.isEmpty && extra.isEmpty else {
            let detail = "missing=\(missing.count) [\(missing.sorted().prefix(5).joined(separator: ", "))], extra=\(extra.count) [\(extra.sorted().prefix(5).joined(separator: ", "))]"
            throw .weightsCorrupted(path: path, reason: "kokoro: strict parameter mismatch: \(detail)")
        }
        var updates: [(String, MLXArray)] = []
        updates.reserveCapacity(expected.count)
        for key in expectedKeys.sorted() {
            let target = expected[key]!
            let source = mapped[key]!
            guard target.shape == source.shape else {
                throw .weightsCorrupted(path: path, reason: "kokoro: shape mismatch for '\(key)': expected \(target.shape), got \(source.shape)")
            }
            updates.append((key, source.asType(.float32)))
        }
        return updates
    }

    package static func loadVoiceStyle(
        voice: String,
        in directory: URL
    ) throws(BasaltError) -> MLXArray {
        let voicesDir = directory.appending(component: "voices")
        let candidate = voicesDir.appending(component: "\(voice).safetensors")
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            throw .weightsCorrupted(
                path: candidate.path,
                reason: "kokoro: missing voice pack '\(voice).safetensors' in \(voicesDir.path)"
            )
        }
        do {
            let arrays = try MLX.loadArrays(url: candidate)
            guard let tensor = arrays["voice"] ?? arrays.values.first else {
                throw BasaltError.weightsCorrupted(path: candidate.path, reason: "kokoro: voice file has no tensors")
            }
            let f32 = tensor.asType(.float32)
            eval(f32)
            return f32
        } catch let err as BasaltError {
            throw err
        } catch {
            throw .weightsCorrupted(path: candidate.path, reason: "kokoro: failed to load voice '\(voice)': \(error.localizedDescription)")
        }
    }

    private static func fuseWeightNorm(g: MLXArray, v: MLXArray, keyPrefix: String) -> MLXArray {
        let g32 = g.asType(.float32)
        let v32 = v.asType(.float32)
        let norm = sqrt(v32.square().sum(axes: [1, 2], keepDims: true) + Float(1e-12))
        let fused = g32 * (v32 / norm)
        if isConvTranspose(keyPrefix) {
            return fused.transposed(1, 2, 0)
        }
        return fused.transposed(0, 2, 1)
    }

    private static func isConvTranspose(_ keyPrefix: String) -> Bool {
        keyPrefix.contains(".ups.")
    }

    private static func convertTensor(_ tensor: MLXArray, originalKey: String) -> MLXArray {
        let f32 = tensor.asType(.float32)
        if (originalKey.contains(".noise_convs.") || originalKey.hasSuffix("F0_proj.weight") || originalKey.hasSuffix("N_proj.weight"))
            && originalKey.hasSuffix(".weight") && f32.ndim == 3 {
            return f32.transposed(0, 2, 1)
        }
        if originalKey.contains(".alpha1.") || originalKey.contains(".alpha2.") {
            return f32.transposed(0, 2, 1)
        }
        return f32
    }

    private static func remapKey(_ key: String) -> String {
        if key.hasPrefix("bert.encoder.albert_layer_groups.0.albert_layers.0.") {
            let suffix = key.dropFirst("bert.encoder.albert_layer_groups.0.albert_layers.0.".count)
            return "bert.encoder.layer.\(suffix)"
        }
        if key.hasPrefix("decoder.asr_res.0.") {
            return "decoder.asr_res." + key.dropFirst("decoder.asr_res.0.".count)
        }
        if key.hasPrefix("predictor.duration_proj.linear_layer.") {
            return "predictor.duration_proj." + key.dropFirst("predictor.duration_proj.linear_layer.".count)
        }
        if key.hasPrefix("text_encoder.cnn.") {
            let parts = key.split(separator: ".")
            if parts.count >= 5, let layerIdx = Int(parts[2]), let subIdx = Int(parts[3]) {
                let tail = parts[4...].joined(separator: ".")
                return subIdx == 0 ? "text_encoder.cnn_conv.\(layerIdx).\(tail)" : "text_encoder.cnn_norm.\(layerIdx).\(tail)"
            }
        }
        if key.hasPrefix("predictor.text_encoder.lstms.") {
            let parts = key.split(separator: ".")
            if parts.count >= 5, let rawIdx = Int(parts[3]) {
                let tail = parts[4...].joined(separator: ".")
                return rawIdx.isMultiple(of: 2)
                    ? "predictor.text_encoder.lstms.\(rawIdx / 2).\(tail)"
                    : "predictor.text_encoder.norms.\(rawIdx / 2).\(tail)"
            }
        }
        return key
    }
}

package enum KokoroPhonemizer {
    package static let maxPhonemeTokens = 510

    package static func requireEspeakBinary() throws(BasaltError) -> String {
        guard let binary = findEspeakBinary() else {
            throw .invalidRequestStructure(
                details: "kokoro: espeak-ng binary not found (install via 'brew install espeak-ng')"
            )
        }
        return binary
    }

    package static func phonemize(
        _ text: String,
        voice: String,
        language: String?
    ) throws(BasaltError) -> String {
        let binary = try requireEspeakBinary()
        let espeakLang = resolveEspeakVoice(voice: voice, language: language)
        guard let ipa = runEspeak(binary: binary, lang: espeakLang, text: text), !ipa.isEmpty else {
            throw .invalidRequestStructure(details: "kokoro: espeak-ng failed to phonemize input")
        }
        return normalizeEspeakIPA(ipa)
    }

    private static func normalizeEspeakIPA(_ raw: String) -> String {
        var s = String(raw.unicodeScalars.filter { $0 != "\u{200D}" && $0 != "\u{0361}" && $0 != "\u{035C}" })
        let replacements: [(String, String)] = [
            ("tʃ", "ʧ"), ("dʒ", "ʤ"),
            ("aɪ", "I"), ("aʊ", "W"), ("eɪ", "A"), ("oʊ", "O"), ("əʊ", "O"), ("ɔɪ", "Y"),
            ("ɝ", "ɜɹ"), ("ɚ", "əɹ"), ("r", "ɹ"), ("g", "ɡ"), ("ː", ""),
        ]
        for (from, to) in replacements {
            s = s.replacingOccurrences(of: from, with: to)
        }
        return s
    }

    package static func tokenize(_ phonemes: String, vocab: [String: Int]) throws(BasaltError) -> [Int32] {
        var ids: [Int32] = [0]
        ids.reserveCapacity(min(maxPhonemeTokens + 2, phonemes.count + 2))
        for ch in phonemes {
            guard let id = vocab[String(ch)] else { continue }
            if ids.count > maxPhonemeTokens {
                throw .invalidRequestStructure(
                    details: "kokoro: phonemized input exceeds maximum of \(maxPhonemeTokens) tokens"
                )
            }
            ids.append(Int32(id))
        }
        ids.append(0)
        return ids
    }

    private static func resolveEspeakVoice(voice: String, language: String?) -> String {
        let lang = language?.lowercased() ?? ""
        if lang.hasPrefix("ru") { return "ru" }
        if lang.hasPrefix("en-gb") || voice.hasPrefix("bf_") || voice.hasPrefix("bm_") { return "en-gb" }
        if lang.hasPrefix("fr") || voice.hasPrefix("ff_") { return "fr-fr" }
        if lang.hasPrefix("es") || voice.hasPrefix("ef_") || voice.hasPrefix("em_") { return "es" }
        return "en-us"
    }

    private static func findEspeakBinary() -> String? {
        for path in ["/opt/homebrew/bin/espeak-ng", "/usr/local/bin/espeak-ng", "/usr/bin/espeak-ng"] {
            if FileManager.default.isExecutableFile(atPath: path) { return path }
        }
        return nil
    }

    private static func runEspeak(binary: String, lang: String, text: String) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: binary)
        proc.arguments = ["-q", "--ipa=3", "-v", lang, "--", text]
        let outPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = Pipe()
        do {
            try proc.run()
            let data = outPipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0, let raw = String(data: data, encoding: .utf8) else { return nil }
            return raw
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "_", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }
}
