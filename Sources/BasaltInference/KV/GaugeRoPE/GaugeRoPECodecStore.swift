import BasaltCore
import CryptoKit
import Foundation
import MLX

extension GaugeRoPECodec {
    static func stored(basis: MLXArray, projection: MLXArray, layout: GaugeRoPELayout, keyDim: Int, valueDim: Int) -> GaugeRoPECodec? {
        guard basis.ndim == 3, projection.ndim == 2, basis.dim(0) * 2 == layout.rotaryDims,
              basis.dim(2) == min(max(1, layout.carrierRank), basis.dim(1)), keyDim >= layout.rotaryDims, valueDim > 0,
              projection.dim(0) == basis.dim(1) * (keyDim + valueDim), projection.dim(1) > 0 else { return nil }
        return GaugeRoPECodec(
            carrier: GaugeRoPECarrier(
                basis: basis.asType(.float32).materialized(), rotaryDims: layout.rotaryDims, numKVHeads: basis.dim(1), carrierHeads: basis.dim(2)),
            projection: projection.asType(.float32).materialized(), ropeBase: layout.ropeBase, keyDim: keyDim, valueDim: valueDim)
    }
}

enum GaugeRoPECodecStore {
    static let fileName = "gaugerope.safetensors"
    static let formatVersion = "1"

    static func fingerprint(modelDirectory: URL) -> String {
        var hasher = SHA256()
        let files = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        where file.pathExtension == PromptCacheDiskStore.fileExtension && file.lastPathComponent != fileName {
            let resolved = file.resolvingSymlinksInPath()
            let size = ((try? FileManager.default.attributesOfItem(atPath: resolved.path))?[.size] as? Int) ?? -1
            let line = "\(file.lastPathComponent) \(size) \(PromptCacheDiskStore.sampledContentHash(resolved, size: size))\n"
            hasher.update(data: Data(line.utf8))
        }
        return PromptCacheDiskStore.hex(hasher.finalize())
    }

    static func save(
        _ codecs: [Int: GaugeRoPECodec], layout: GaugeRoPELayout, fingerprint: String, to directory: URL
    ) throws(BasaltError) {
        let url = directory.appending(component: fileName)
        guard let first = codecs.values.first else {
            throw .modelConfigurationInvalid(path: url.path, reason: "no full-attention layer was calibrated, so there is no GaugeRoPE codec to store (#647)")
        }
        var arrays: [String: MLXArray] = [:]
        for (index, codec) in codecs {
            arrays["layers.\(index).carrier_basis"] = codec.carrier.basis
            arrays["layers.\(index).projection"] = codec.projection
        }
        let metadata = [
            "format_version": formatVersion, "fingerprint": fingerprint,
            "rope_base": String(layout.ropeBase), "rotary_dims": String(layout.rotaryDims),
            "key_dim": String(first.keyDim), "value_dim": String(first.valueDim),
            "carrier_rank": String(layout.carrierRank), "calibration_tokens": String(max(layout.calibrationTokens, GaugeRoPECalibration.storedCalibrationTokens)),
        ]
        do {
            try MLX.save(arrays: arrays, metadata: metadata, url: url)
        } catch {
            throw .modelConfigurationInvalid(path: url.path, reason: "the GaugeRoPE codec could not be written: \(error.localizedDescription)")
        }
    }

    static func load(from directory: URL, layout: GaugeRoPELayout, fingerprint: String) throws(BasaltError) -> [Int: GaugeRoPECodec]? {
        let url = directory.appending(component: fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        func refusal(_ why: String) -> BasaltError {
            .modelConfigurationInvalid(
                path: url.path,
                reason: "the stored GaugeRoPE codec \(why); run basalt models convert --to gaugerope again, or delete \(fileName) to calibrate on each request (#647)")
        }
        guard let (arrays, metadata) = try? loadArraysAndMetadata(url: url) else { throw refusal("cannot be read") }
        guard metadata["format_version"] == formatVersion else {
            throw refusal("has format version \(metadata["format_version"] ?? "none") and this build reads version \(formatVersion)")
        }
        guard metadata["fingerprint"] == fingerprint else { throw refusal("was calibrated on other weights (the weight fingerprint differs)") }
        guard let keyDim = metadata["key_dim"].flatMap({ Int($0) }), let valueDim = metadata["value_dim"].flatMap({ Int($0) }),
              metadata["rotary_dims"].flatMap({ Int($0) }) == layout.rotaryDims,
              metadata["rope_base"].flatMap({ Float($0) }) == layout.ropeBase,
              metadata["carrier_rank"].flatMap({ Int($0) }) == layout.carrierRank else {
            throw refusal("does not match this model's RoPE layout (\(layout.rotaryDims) rotary dims, base \(layout.ropeBase), carrier rank \(layout.carrierRank))")
        }
        var codecs: [Int: GaugeRoPECodec] = [:]
        for (key, projection) in arrays where key.hasSuffix(".projection") {
            let parts = key.split(separator: ".")
            guard parts.count == 3, parts[0] == "layers", let index = Int(parts[1]), let basis = arrays["layers.\(index).carrier_basis"],
                  let codec = GaugeRoPECodec.stored(basis: basis, projection: projection, layout: layout, keyDim: keyDim, valueDim: valueDim) else {
                throw refusal("has a layer entry '\(key)' whose tensors do not fit the stored dimensions")
            }
            codecs[index] = codec
        }
        guard !codecs.isEmpty else { throw refusal("holds no layer") }
        return codecs
    }
}
