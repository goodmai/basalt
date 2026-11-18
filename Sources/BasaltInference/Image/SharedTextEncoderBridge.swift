import Foundation
import MLX
import MLXLMCommon
import MLXNN
import BasaltCore

extension MLXInferenceEngine {
    public func sharedImagePromptEmbedding(_ prompt: String, dimension: Int) async throws -> [Float]? {
        try await encodePromptForImage(prompt, dimension: dimension)
    }

    public func encodePromptForImage(_ prompt: String, dimension: Int = 256) async throws -> [Float] {
        let targetDim = max(16, dimension)
        try Task.checkCancellation()
        guard let container else {
            return FourierTextEmbedder.embed(prompt, dimension: targetDim)
        }
        let rawFeatures: [Float]? = await container.performEvaluating(nonSendable: ()) { context, _ in
            let tokenIds = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            guard !tokenIds.isEmpty else { return nil }
            let leaves = context.model.leafModules().flattened()
            let preferred = leaves.first { path, mod in
                mod is Embedding && (path.contains("embed") || path.contains("tok") || path.contains("wte"))
            }
            let fallback = leaves.first { _, mod in mod is Embedding }
            guard let chosen = (preferred ?? fallback)?.1 as? Embedding else { return nil }
            let vocab = chosen.weight.dim(0)
            guard vocab > 0 else { return nil }
            let clipped = tokenIds.prefix(512).compactMap { (0 ..< vocab).contains($0) ? Int32($0) : nil }
            guard !clipped.isEmpty else { return nil }
            let hidden = chosen(MLXArray(clipped)).asType(.float32)
            let rms = rsqrt(mean(hidden * hidden, axis: -1, keepDims: true) + 1e-6)
            let normed = hidden * rms
            let pooled = 0.7 * mean(normed, axis: 0) + 0.3 * max(normed, axis: 0)
            eval(pooled)
            return pooled.asArray(Float.self)
        }
        guard let rawFeatures, !rawFeatures.isEmpty else {
            return FourierTextEmbedder.embed(prompt, dimension: targetDim)
        }
        return FourierTextEmbedder.projectAndNormalize(rawFeatures, to: targetDim, salt: prompt.count)
    }
}

package enum FourierTextEmbedder {
    package static func embed(_ text: String, dimension: Int = 256) -> [Float] {
        let dim = max(16, dimension)
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else {
            var empty = [Float](repeating: 0, count: dim)
            empty[0] = 1.0
            return empty
        }
        let words = normalized.split(whereSeparator: { $0.isWhitespace || $0.isPunctuation })
        var accum = [Double](repeating: 0, count: dim)
        for (wIdx, word) in words.prefix(128).enumerated() {
            let h = fnv1a64(Substring(word))
            let weight = 1.0 / Double(1 + wIdx / 8)
            accumulateHash(h, position: wIdx + 1, weight: weight, into: &accum)
        }
        let scalars = Array(normalized.unicodeScalars.prefix(512))
        for idx in scalars.indices {
            let trigram = trigramHash(scalars, at: idx)
            accumulateHash(trigram, position: idx + 1, weight: 0.35, into: &accum)
        }
        var out = accum.map { Float($0) }
        normalizeL2(&out)
        return out
    }

    package static func embedTokenSequence(_ text: String, maxTokens: Int = 16, dimension: Int = 64) -> [Float] {
        let count = max(1, maxTokens)
        let dim = max(8, dimension)
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        var flat: [Float] = []
        flat.reserveCapacity(count * dim)
        for slot in 0 ..< count {
            let tokenText = words.isEmpty ? "" : words[slot % words.count]
            var vec = embed(tokenText, dimension: dim)
            for d in 0 ..< dim {
                let freq = Float(d + 1) * 0.19
                let posBias = 0.15 * sin(Float(slot + 1) * freq)
                vec[d] += posBias
            }
            normalizeL2(&vec)
            flat.append(contentsOf: vec)
        }
        return flat
    }

    package static func embedBoundingBox(_ box: ImageBoundingBox, dimension: Int = 64) -> [Float] {
        let dim = max(16, dimension)
        let half = dim / 2
        let coords = [box.x, box.y, box.width, box.height].map { min(1.0, max(0.0, $0)) }
        var spatial = [Float](repeating: 0, count: half)
        for idx in 0 ..< half {
            let c = coords[idx % 4]
            let band = Double((idx / 4) + 1)
            let angle = c * .pi * pow(1.6, band - 1.0)
            spatial[idx] = idx.isMultiple(of: 2) ? Float(sin(angle)) : Float(cos(angle))
        }
        let labelVec = embed(box.label, dimension: dim - half)
        var combined = spatial + labelVec
        normalizeL2(&combined)
        return combined
    }

    package static func embedVGLScene(_ vglJSON: String, maxTokens: Int = 4, dimension: Int = 64) -> [Float] {
        let count = max(1, maxTokens)
        let dim = max(16, dimension)
        let segments = extractVGLSegments(vglJSON, targetCount: count)
        var flat: [Float] = []
        flat.reserveCapacity(count * dim)
        for seg in segments {
            flat.append(contentsOf: embed(seg, dimension: dim))
        }
        return flat
    }
}

extension FourierTextEmbedder {
    package static func parseInlineJSONBoundingBoxes(_ text: String) -> [ImageBoundingBox] {
        guard let data = text.data(using: .utf8),
              let raw = try? JSONSerialization.jsonObject(with: data) else {
            return []
        }
        let items: [[String: Any]]
        if let array = raw as? [[String: Any]] {
            items = array
        } else if let dict = raw as? [String: Any], let boxes = dict["boxes"] as? [[String: Any]] {
            items = boxes
        } else {
            return []
        }
        var parsed: [ImageBoundingBox] = []
        for item in items.prefix(16) {
            let label = (item["label"] as? String) ?? (item["caption"] as? String) ?? "object"
            if let bbox = item["bbox"] as? [Double], bbox.count == 4 {
                parsed.append(ImageBoundingBox(label: label, x: bbox[0], y: bbox[1], width: bbox[2], height: bbox[3]))
            } else if let x = item["x"] as? Double, let y = item["y"] as? Double,
                      let w = item["width"] as? Double, let h = item["height"] as? Double {
                parsed.append(ImageBoundingBox(label: label, x: x, y: y, width: w, height: h))
            }
        }
        return parsed
    }

    package static func projectAndNormalize(_ source: [Float], to targetDim: Int, salt: Int = 0) -> [Float] {
        let dim = max(1, targetDim)
        guard !source.isEmpty else { return embed("", dimension: dim) }
        var out = [Float](repeating: 0, count: dim)
        for (idx, val) in source.enumerated() where val.isFinite {
            let bucket = (idx &* 7 &+ salt) % dim
            let sign: Float = ((idx / dim) + salt).isMultiple(of: 2) ? 1.0 : -1.0
            out[bucket >= 0 ? bucket : bucket + dim] += val * sign
        }
        normalizeL2(&out)
        return out
    }

    package static func scoreImageAlignment(
        pixels: [Float],
        dimensions: ImageDimensions,
        channels: Int,
        promptVector: [Float]
    ) -> Float {
        guard !pixels.isEmpty, !promptVector.isEmpty, channels >= 3 else { return 0 }
        let dim = promptVector.count
        var descriptor = [Float](repeating: 0, count: dim)
        let totalPixels = min(pixels.count / channels, dimensions.width * dimensions.height)
        let strideStep = max(1, totalPixels / 256)
        var sampleIdx = 0
        var pixelIndex = 0
        while pixelIndex < totalPixels {
            let base = pixelIndex * channels
            let r = pixels[base]
            let g = pixels[base + 1]
            let b = pixels[base + 2]
            let luma = 0.299 * r + 0.587 * g + 0.114 * b
            let chromaRG = r - g
            let chromaYB = 0.5 * (r + g) - b
            let slot0 = (sampleIdx * 3) % dim
            let slot1 = (sampleIdx * 3 + 1) % dim
            let slot2 = (sampleIdx * 3 + 2) % dim
            descriptor[slot0] += luma - 0.5
            descriptor[slot1] += chromaRG
            descriptor[slot2] += chromaYB
            sampleIdx += 1
            pixelIndex += strideStep
        }
        normalizeL2(&descriptor)
        var dot: Float = 0
        for i in 0 ..< dim { dot += descriptor[i] * promptVector[i] }
        return dot
    }

    package static func normalizeL2(_ vector: inout [Float]) {
        var normSq: Double = 0
        for v in vector where v.isFinite { normSq += Double(v) * Double(v) }
        if normSq <= 1e-12 {
            if !vector.isEmpty { vector[0] = 1.0 }
            return
        }
        let inv = Float(1.0 / normSq.squareRoot())
        for idx in vector.indices {
            vector[idx] = vector[idx].isFinite ? vector[idx] * inv : 0
        }
    }

    private static func fnv1a64(_ sub: Substring) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in sub.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return hash
    }

    private static func trigramHash(_ scalars: [Unicode.Scalar], at index: Int) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for offset in 0 ..< 3 where index + offset < scalars.count {
            hash ^= UInt64(scalars[index + offset].value)
            hash &*= 1_099_511_628_211
        }
        return hash
    }

    private static func accumulateHash(_ hash: UInt64, position: Int, weight: Double, into accum: inout [Double]) {
        let dim = accum.count
        let phaseBase = Double(hash & 0xffff) / 65_535.0 * 2.0 * .pi
        let freqBase = 1.0 + Double((hash >> 16) & 0xff) / 64.0
        let startBucket = Int((hash >> 24) % UInt64(dim))
        let span = min(dim, 16)
        for offset in 0 ..< span {
            let idx = (startBucket + offset * 7) % dim
            let angle = phaseBase + freqBase * Double(offset + 1) + Double(position) * 0.13
            accum[idx] += weight * cos(angle)
        }
    }

    private static func extractVGLSegments(_ vglJSON: String, targetCount: Int) -> [String] {
        if let data = vglJSON.data(using: .utf8),
           let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            let sortedKeys = dict.keys.sorted()
            if !sortedKeys.isEmpty {
                return (0 ..< targetCount).map { idx in
                    let key = sortedKeys[idx % sortedKeys.count]
                    return "\(key):\(String(describing: dict[key] ?? ""))"
                }
            }
        }
        return (0 ..< targetCount).map { idx in "\(vglJSON)#\(idx)" }
    }
}

package struct TextConditioningInput: Sendable {
    package let prompt: String
    package let sharedVector: [Float]
    package let bboxLayout: [ImageBoundingBox]
    package let vglSceneJSON: String?

    package init(
        prompt: String,
        sharedVector: [Float],
        bboxLayout: [ImageBoundingBox] = [],
        vglSceneJSON: String? = nil
    ) {
        self.prompt = prompt
        self.sharedVector = sharedVector
        self.bboxLayout = bboxLayout
        self.vglSceneJSON = vglSceneJSON
    }
}

package final class TextConditioningEncoder: Module {
    package let hiddenDim: Int
    package let textTokenCount: Int
    package let sharedInputDim: Int
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ModuleInfo(key: "shared_proj") var sharedProj: Linear
    @ModuleInfo(key: "bbox_proj") var bboxProj: Linear
    @ModuleInfo(key: "vgl_proj") var vglProj: Linear
    @ModuleInfo(key: "cond_norm") var condNorm: RMSNorm

    package init(hiddenDim: Int, sharedInputDim: Int = 256, textTokenCount: Int = 12) {
        let d = max(16, hiddenDim)
        let sDim = max(16, sharedInputDim)
        self.hiddenDim = d
        self.sharedInputDim = sDim
        self.textTokenCount = max(4, textTokenCount)
        self._textProj.wrappedValue = Linear(d, d)
        self._sharedProj.wrappedValue = Linear(sDim, d)
        self._bboxProj.wrappedValue = Linear(d, d)
        self._vglProj.wrappedValue = Linear(d, d)
        self._condNorm.wrappedValue = RMSNorm(dimensions: d)
    }

    package func encode(_ input: TextConditioningInput) -> (tokens: MLXArray, pooled: MLXArray) {
        let alignedShared = input.sharedVector.count == sharedInputDim
            ? input.sharedVector
            : FourierTextEmbedder.projectAndNormalize(input.sharedVector, to: sharedInputDim)
        let sharedEmb = sharedProj(MLXArray(alignedShared).reshaped(1, 1, sharedInputDim))
        let seqFlat = FourierTextEmbedder.embedTokenSequence(
            input.prompt,
            maxTokens: textTokenCount,
            dimension: hiddenDim
        )
        var pieces: [MLXArray] = [
            textProj(MLXArray(seqFlat).reshaped(1, textTokenCount, hiddenDim)) + sharedEmb
        ]
        let allBoxes = input.bboxLayout + (input.vglSceneJSON.map(FourierTextEmbedder.parseInlineJSONBoundingBoxes) ?? [])
        if !allBoxes.isEmpty {
            let boxFlat = allBoxes.prefix(16).flatMap {
                FourierTextEmbedder.embedBoundingBox($0, dimension: hiddenDim)
            }
            let boxCount = min(16, allBoxes.count)
            pieces.append(bboxProj(MLXArray(boxFlat).reshaped(1, boxCount, hiddenDim)))
        }
        if let vgl = input.vglSceneJSON, !vgl.isEmpty {
            let vglFlat = FourierTextEmbedder.embedVGLScene(vgl, maxTokens: 4, dimension: hiddenDim)
            pieces.append(vglProj(MLXArray(vglFlat).reshaped(1, 4, hiddenDim)))
        }
        let sequence = condNorm(concatenated(pieces, axis: 1))
        let pooled = mean(sequence, axis: 1)
        return (sequence, pooled)
    }
}
