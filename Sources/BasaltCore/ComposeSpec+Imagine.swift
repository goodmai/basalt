import Foundation

extension ComposeSpec {
    public struct Imagine: Sendable, Codable, Equatable {
        public var chain: [String]
        public var upsampleVia: String?
        public var shareEncoder: String?
        public var decoderVia: String?
        public var steps: Int?
        public var size: String?
        public var format: ImageOutputFormat?
        public var teacacheThreshold: Double?
        public var previewEverySteps: Int?
        public var candidates: Int?
        public var judgeVia: String?
        public var onGPUBusy: ImageOnGPUBusy?
        public var shareTextEncoderWith: String?
        public var kvCacheReferences: Bool?
        public var rgba: Bool?
        public var upscale: ImagineUpscaleSpec?

        public init(
            chain: [String],
            upsampleVia: String? = nil,
            shareEncoder: String? = nil,
            steps: Int? = nil,
            size: String? = nil,
            onGPUBusy: ImageOnGPUBusy? = nil
        ) {
            self.chain = chain
            self.upsampleVia = upsampleVia
            self.shareEncoder = shareEncoder
            self.decoderVia = nil
            self.steps = steps
            self.size = size
            self.format = nil
            self.teacacheThreshold = nil
            self.previewEverySteps = nil
            self.candidates = nil
            self.judgeVia = nil
            self.onGPUBusy = onGPUBusy
        }

        enum CodingKeys: String, CodingKey {
            case chain, steps, size, format, candidates, rgba, upscale
            case upsampleVia = "upsample_via"
            case shareEncoder = "share_encoder"
            case shareTextEncoderWith = "share_text_encoder_with"
            case kvCacheReferences = "kv_cache_references"
            case decoderVia = "decoder_via"
            case teacacheThreshold = "teacache_threshold"
            case previewEverySteps = "preview_every_steps"
            case judgeVia = "judge_via"
            case onGPUBusy = "on_gpu_busy"
        }
    }
}

extension ComposeSpec {
    public var imageGeneratorModels: Set<String> {
        Set(views.flatMap { $0.spec.imagine?.chain ?? [] }.filter { models[$0]?.isCloud == false })
    }

    public var imagineDecoderModels: Set<String> {
        let names = views.compactMap { $0.spec.imagine?.decoderVia }
        return Set(names.filter { models[$0]?.isCloud == false })
    }

    public var imagineAuxModels: Set<String> {
        let names = views.flatMap { view -> [String] in
            guard let imagine = view.spec.imagine else { return [] }
            return [imagine.upsampleVia, imagine.shareEncoder, imagine.judgeVia].compactMap { $0 }
        }
        return Set(names.filter { models[$0]?.isCloud == false })
    }

    var imagineRoleNames: [String] {
        imagine?.chain ?? []
    }

    var imagineDecoderRoleNames: [String] {
        [imagine?.decoderVia].compactMap { $0 }
    }

    var imagineAuxRoleNames: [String] {
        guard let imagine else { return [] }
        return [imagine.upsampleVia, imagine.shareEncoder, imagine.judgeVia].compactMap { $0 }
    }

    var imagineIssues: [Issue] {
        guard let imagine else { return [] }
        var issues = imagineChainIssues(imagine.chain) + imagineModelRefIssues(imagine)
        if let size = imagine.size {
            issues += imagineSizeIssues(size)
        }
        if let steps = imagine.steps, !(1 ... 100).contains(steps) {
            issues.append(Issue("pipeline.imagine.steps", "must lie between 1 and 100"))
        }
        if let candidates = imagine.candidates, !(1 ... 64).contains(candidates) {
            issues.append(Issue("pipeline.imagine.candidates", "must lie between 1 and 64"))
        }
        if let preview = imagine.previewEverySteps, preview < 0 {
            issues.append(Issue("pipeline.imagine.preview_every_steps", "must be 0 or a positive integer"))
        }
        if let threshold = imagine.teacacheThreshold, !threshold.isFinite || threshold < 0 {
            issues.append(Issue("pipeline.imagine.teacache_threshold", "must be 0 or positive"))
        }
        return issues
    }

    private func imagineChainIssues(_ chain: [String]) -> [Issue] {
        var issues: [Issue] = []
        if chain.isEmpty {
            issues.append(Issue("pipeline.imagine.chain", "an imagine chain names at least one declared image model"))
        }
        if Set(chain).count != chain.count {
            issues.append(Issue("pipeline.imagine.chain", "an image model appears twice"))
        }
        for name in chain where models[name] == nil {
            issues.append(Issue("pipeline.imagine.chain", "\(name) is not a model declared under models"))
        }
        return issues
    }

    private func imagineModelRefIssues(_ imagine: Imagine) -> [Issue] {
        var issues: [Issue] = []
        for (key, value) in [
            ("upsample_via", imagine.upsampleVia),
            ("share_encoder", imagine.shareEncoder),
            ("decoder_via", imagine.decoderVia),
            ("judge_via", imagine.judgeVia),
        ] {
            guard let name = value else { continue }
            if models[name] == nil {
                issues.append(Issue("pipeline.imagine.\(key)", "\(name) is not a model declared under models"))
            } else if key == "share_encoder", models[name]?.isCloud == true {
                issues.append(Issue("pipeline.imagine.share_encoder", "\(name) is a cloud model; shared text encoders must be local"))
            } else if key == "decoder_via", models[name]?.isCloud == true {
                issues.append(Issue("pipeline.imagine.decoder_via", "\(name) is a cloud model; latent VAE decoders must be local"))
            }
        }
        return issues
    }

    private func imagineSizeIssues(_ size: String) -> [Issue] {
        do {
            let dims = try ImageGenerationRequest.parseSize(size)
            let range = ImageGenerationRequest.minDimension ... ImageGenerationRequest.maxDimension
            guard range.contains(dims.width), range.contains(dims.height),
                  dims.width.isMultiple(of: 16), dims.height.isMultiple(of: 16) else {
                return [Issue(
                    "pipeline.imagine.size",
                    "dimensions \(dims.width)x\(dims.height) must be multiples of 16 within \(range.lowerBound)...\(range.upperBound)"
                )]
            }
            return []
        } catch {
            return [Issue("pipeline.imagine.size", error.localizedDescription)]
        }
    }
}
